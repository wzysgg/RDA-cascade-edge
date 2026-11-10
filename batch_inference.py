"""Batch-at-most-ten eager inference with per-sample Fast early exit.

This is an inference adapter, not a batching scheduler. It never waits to fill
a batch and never reads labels or cached routes. The caller sends the original
input rows listed in accurate_indices to Accurate after this call returns.
"""
import torch

from live_runtime_v20_two_gpu import GpuEarlyHeads
from physical_probe import fast_prefix, fast_suffix
from quality_need_v24 import quality_order

MAX_BATCH_SIZE = 10


class MixedFastBatch:
    """One shared Prefix, heads only on unknown rows, one same-card Suffix.

    The mask is public Local routing history, never a pool label or predicted
    future route. Known Fast rows keep their original route unconditionally.
    """
    def __init__(self,fast,heads):
        self.fast=fast;self.heads=heads

    def __call__(self,inputs,needs_probe,on_routes=None,suffix_filter=None,on_stage=None):
        if inputs.ndim!=4 or not inputs.is_cuda or not 1<=len(needs_probe)==len(inputs)<=10:
            raise ValueError('Expected one to ten CUDA inputs and a public probe mask')
        probe=tuple(i for i,needed in enumerate(needs_probe) if needed)
        features=fast_prefix(self.fast,inputs)
        if on_stage:on_stage('prefix')
        roles=['Fast']*len(inputs);public=[]
        if probe:
            indices=torch.tensor(probe,dtype=torch.long,device=inputs.device)
            selected=features if len(probe)==len(inputs) else features.index_select(0,indices)
            pooled=torch.nn.functional.adaptive_avg_pool2d(selected,1).flatten(1)
            public=self.heads.predict_batch(pooled).cpu().tolist()
            for i,row in zip(probe,public):
                order=quality_order(dict(zip(self.heads.roles,row[:-1])),'B',row[-1],True)
                roles[i]=order[0] if order else 'Abstain'
        if on_routes and probe:
            on_routes(dict(probe_indices=probe,roles=tuple(roles[i] for i in probe),public_decisions=public))
        if on_stage:on_stage('probe')
        fast_indices=tuple(i for i,r in enumerate(roles) if r=='Fast')
        expired=()
        if suffix_filter is not None:
            allowed=tuple(suffix_filter(fast_indices))
            if len(set(allowed))!=len(allowed) or any(i not in fast_indices for i in allowed):
                raise ValueError('Invalid Suffix subset')
            if any(not needs_probe[i] and i not in allowed for i in fast_indices):
                raise ValueError('Known Fast rows may not be silently dropped')
            expired=tuple(i for i in fast_indices if i not in allowed);fast_indices=allowed
        if fast_indices:
            indices=torch.tensor(fast_indices,dtype=torch.long,device=inputs.device)
            selected=features if len(fast_indices)==len(inputs) else features.index_select(0,indices)
            predictions=fast_suffix(self.fast,selected).argmax(1)
        else:predictions=torch.empty(0,dtype=torch.long,device=inputs.device)
        return dict(roles=tuple(roles),probe_indices=probe,public_decisions=public,
            fast_indices=fast_indices,fast_predictions=predictions,expired_fast_indices=expired,
            accurate_indices=tuple(i for i,r in enumerate(roles) if r=='Accurate'),
            rejected_indices=tuple(i for i,r in enumerate(roles) if r=='Abstain'))


class BatchedEarlyHeads(GpuEarlyHeads):
    """Evaluate the existing FP64 coefficients independently for every row."""

    def logistic_batch(self, coefficients, x):
        if not isinstance(coefficients, dict):
            return coefficients.expand(x.shape[0])
        ones = self.one.expand(x.shape[0], 1)
        xx = torch.cat((ones, (x-coefficients['mean'])/coefficients['std']), dim=1)
        return torch.sigmoid(torch.clamp(xx @ coefficients['beta'], -35, 35))

    def predict_batch(self, pooled):
        if pooled.ndim != 2 or not 1 <= pooled.shape[0] <= MAX_BATCH_SIZE:
            raise ValueError('EARLY expects one to ten pooled feature rows')
        vector = pooled.to(torch.float64)
        probabilities = []
        for role in self.roles:
            p = self.logistic_batch(self.coefficients[role, 'base'], vector)
            p = torch.clamp(p, 1e-12, 1-1e-12)
            probabilities.append(self.logistic_batch(
                self.coefficients[role, 'calibrator'],
                torch.log(p/(1-p)).reshape(-1, 1)))
        probabilities = torch.stack(probabilities, dim=1)
        clipped = torch.clamp(probabilities[:, :2], 1e-6, 1-1e-6)
        need_features = torch.cat((vector, torch.log(clipped/(1-clipped))), dim=1)
        need = self.logistic_batch(self.need, need_features)
        return torch.cat((probabilities, need.reshape(-1, 1)), dim=1)


class FastBatchEarlyExit:
    """Prefix all rows; execute Suffix only for rows routed to Fast.

    GPU quality evaluation is followed by the existing CPU quality_order rule.
    Its decision transfer, CPU branch and compaction are real costs, not zero
    latency. No CNN weights, quality thresholds, or Need rule are changed.
    """

    def __init__(self, fast, heads, quality_need_enabled=True):
        self.fast = fast
        self.heads = heads
        self.quality_need_enabled = quality_need_enabled

    def __call__(self, inputs, on_routes=None, suffix_filter=None,suffix_scheduler=None):
        if inputs.ndim != 4 or not 1 <= inputs.shape[0] <= MAX_BATCH_SIZE:
            raise ValueError('Fast accepts one to ten input images')
        if not inputs.is_cuda:
            raise ValueError('This adapter measures CUDA eager inference')
        features = fast_prefix(self.fast, inputs)
        pooled = torch.nn.functional.adaptive_avg_pool2d(features, 1).flatten(1)
        decisions = self.heads.predict_batch(pooled)
        # This synchronizes the actual public EARLY result before routing.
        public = decisions.cpu().tolist()
        roles = []
        for row in public:
            order = quality_order(dict(zip(self.heads.roles, row[:-1])), 'B',
                                  row[-1], self.quality_need_enabled)
            roles.append(order[0] if order else 'Abstain')
        fast_indices = tuple(i for i, role in enumerate(roles) if role == 'Fast')
        accurate_indices = tuple(i for i, role in enumerate(roles) if role == 'Accurate')
        rejected_indices = tuple(i for i, role in enumerate(roles) if role == 'Abstain')
        if on_routes is not None:
            # Publish A immediately, before this card executes any Fast Suffix.
            on_routes(dict(roles=tuple(roles),public_decisions=public,
                           fast_indices=fast_indices,accurate_indices=accurate_indices,
                           rejected_indices=rejected_indices))
        expired_fast_indices = ()
        suffix_groups=[]
        if suffix_scheduler is not None and suffix_filter is not None:
            raise ValueError('Choose one Suffix execution guard')
        if suffix_filter is not None:
            allowed = tuple(suffix_filter(fast_indices))
            if len(set(allowed))!=len(allowed) or any(i not in fast_indices for i in allowed):
                raise ValueError('Suffix filter must return a subset of routed Fast rows')
            expired_fast_indices = tuple(i for i in fast_indices if i not in allowed)
            fast_indices = allowed
        if suffix_scheduler is not None:
            pending=list(fast_indices);executed=[];expired=[];outputs=[]
            while pending:
                group,late=suffix_scheduler(tuple(pending))
                group=tuple(group);late=tuple(late)
                if (not group and not late) or len(set(group+late))!=len(group+late):
                    raise ValueError('Suffix scheduler must make unique progress')
                if any(i not in pending for i in group+late):
                    raise ValueError('Suffix scheduler selected an unavailable row')
                if group!=tuple(i for i in pending if i in group):
                    raise ValueError('Suffix groups must preserve EDF row order')
                if group:
                    index=torch.tensor(group,dtype=torch.long,device=inputs.device)
                    continuation=features if len(group)==len(roles) else features.index_select(0,index)
                    # Synchronize/copy each result before asking the next group
                    # for its actual remaining budget. No saved-feature queue,
                    # no model interleaving, and no repeated Prefix computation.
                    outputs.append(fast_suffix(self.fast,continuation).argmax(1).cpu())
                    executed.extend(group);suffix_groups.append(group)
                expired.extend(late);done=set(group+late)
                pending=[i for i in pending if i not in done]
            fast_indices=tuple(executed);expired_fast_indices=tuple(expired)
            predictions=torch.cat(outputs) if outputs else torch.empty(0,dtype=torch.long)
        elif fast_indices:
            if len(fast_indices) == len(roles):
                continuation = features
            else:
                index = torch.tensor(fast_indices, dtype=torch.long, device=inputs.device)
                continuation = features.index_select(0, index)
            predictions = fast_suffix(self.fast, continuation).argmax(1)
            suffix_groups=[fast_indices]
        else:
            predictions = torch.empty(0, dtype=torch.long, device=inputs.device)
        return dict(roles=tuple(roles), decisions=decisions,
                    fast_indices=fast_indices, accurate_indices=accurate_indices,
                    rejected_indices=rejected_indices, fast_predictions=predictions,
                    expired_fast_indices=expired_fast_indices,
                    suffix_groups=tuple(suffix_groups),
                    suffix_samples=len(fast_indices))
