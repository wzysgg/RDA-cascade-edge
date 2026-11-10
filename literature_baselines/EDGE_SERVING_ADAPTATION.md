# EdgeServing-Stability-2GPU — adaptation and fidelity record

| Field | Value |
|---|---|
| Arm | `edge_serving` (task-1, highest-priority literature arm) |
| Paper | *EdgeServing: Deadline-Aware Multi-DNN Serving at the Edge*, arXiv:2605.05527v1 [cs.DC], 07 May 2026 |
| Authoritative resolution of the paper | [`EDGE_SERVING_METHOD_AUDIT.md`](EDGE_SERVING_METHOD_AUDIT.md) (25-row fidelity ledger, every equation diffed against the raw HTML) |
| Code | [`edge_serving_adapter.py`](edge_serving_adapter.py) (`edge_serving_adapter.py:102-583`), [`config_edge_serving.json`](config_edge_serving.json) |
| Tests | [`edge_serving_unit_tests.py`](edge_serving_unit_tests.py) — 16 checks, local, stdlib only: `D:\wzy\python\python.exe literature_baselines\edge_serving_unit_tests.py` → `16/16 checks passed`, exit 0 |
| Fidelity tally | **ORIGINAL 18 · ADAPTED 6 · UNSUPPORTED 1** (the audit's own tags, reproduced in §6) |
| Runs on the server | none; the Lead owns every experiment and the single output root `results/T016/literature_comparison_v1/` |

This arm is a **two-GPU adaptation** of the paper's Stability score. Everything the
paper states is implemented as written; everything it leaves open, or assumes away
with a single GPU, is an explicitly labelled adaptation with its own rationale
below. Row numbers refer to the audit's fidelity ledger (§6).

---

## 1. The decision rule as implemented

`choose_action(now_ns, view)` may dispatch **at most one component per call**; the
runtime calls it repeatedly inside one scheduling step until it returns `None`
(`lit_runtime.py:237-317`), and a returned action is executed on the card we name.
The rule is:

1. **Queue heads (rows 1, 2, 9).** Every executable component kind is one of the
   paper's fragment queues, represented by its FIFO head only: Fast components in
   the order the frozen kernel orders them (arrival, task id), Accurate components
   by (enqueue, task id), a pinned Suffix by (enqueue, task id) inside its own card
   list. Candidates = `{FAST_FULL, FAST_PREFIX, ACCURATE, pinned(g)}`.
   — `edge_serving_adapter.py:279-312`
2. **Placement (row 13, ADAPTED).** A ready candidate is placed on the eligible free
   card minimising `R_g + L(m,e,B*)` (ties: smallest card index). A pinned Suffix is
   eligible only on its own card — structural, `JobView.card_affinity`.
   — `edge_serving_adapter.py:344-358`
3. **Duration (rows 3, 11, 20).** `L` = the frozen public service table for the
   component + the card's IPC leg. A FAST_PREFIX commits its FAST_SUFFIX on the same
   card, so its candidate duration is the composite **409954 + 1182290 = 1592244 ns**
   (plus IPC) — exactly the occupancy the frozen kernel charges at dispatch
   (`group_runtime_t011.py:1486-1487`, `:2822-2824`). — `:360-377`
4. **Aggregate (Eqs. 3–5).** All candidates of a round are compared at the shared
   horizon `H = max_a (R_a + L_a)`, the worst-case candidate completion of the round.
   Every pending component of both roles and both cards is projected to that horizon,
   `w_i → w_i + H`, the served component is deleted (`B* = min(|Q|, B_max) = 1`, and
   `τ` is the task's own budget `deadline_abs_ns − arrival_ns`):

       T(H) = Σ_i f(w_i + H)          f(w) = min(exp(w/τ − 1), C)      (Eq. 3, 4)
       S_a  = T(H) − f(w_a + H)                                          (Eq. 4, 7)

   — `_f` at `:403-409`, `_projection_total` at `:411-443`, scoring at `:224-233`
5. **Feasibility (Eq. 6, ADAPTED for residual).** A candidate is admissible only
   while `w_max + R_g + L ≤ τ` (`w_max` = its own queue's maximum waiting time, the
   paper's `w_{m,max}`). If none is admissible the arm either serves the score's
   argmin anyway (default, counted) or leaves the cards idle.
   — `_queue_heads` `:279-305`, `slo_ok` `:338`, filter `:236-249`
6. **Selection (Eq. 7) and tie-break (row 16).** `argmin_a S_a`; ties broken by the
   frozen kernel's own public order `(predicted service work, arrival_ns for FAST_* /
   enqueue_ns for ACCURATE, card index)` and, only if all of those are equal, the
   kernel job id. Counted in `ledger()`. — `:251-257`, `:447-463`

---

## 2. Equation → code map

| Paper element | Text | Where implemented |
|---|---|---|
| Eq. 1 `T=w+t`, `w` = elapsed since **arrival** | §V-B p.1; Alg. 1 line 1 | `lit_api.py:93-95` (`JobView.waiting_ns`) applied at `edge_serving_adapter.py:388-397`; never since enqueue |
| Eq. 3 `f(w)=min(exp(w/τ−1), C)` | §V-C | `edge_serving_adapter.py:403-409`; clip constants `:80-87`; projected terms `:411-443` |
| Eq. 4 `S=Σ_m Σ_{i∈Q_m} f(w_{m,i})` | §V-C | `edge_serving_adapter.py:411-443` (all pending work, both roles, both cards) |
| Eq. 5 `B*=min(|Q_m|,B_max)` | §V-C | `edge_serving_adapter.py:123-125` (`batch_max=1`), deletion of the served term at `:231-233` |
| Eq. 6 `e*=argmax e s.t. w_{m,max}+L(m,e,B*)≤τ` | §V-C | `edge_serving_adapter.py:279-305` (`w_max`), `:322` (`Δ=R+L`), `:338`, `:236-249` |
| Eq. 7 `m*=argmin_m S_m` | §V-C | `edge_serving_adapter.py:251-257`; tie key `:447-455` |
| Alg. 1 lines 2–8 (candidate loop, score, min) | §V-D | `edge_serving_adapter.py:204-262` (candidates `:314-342`, ranking `:251-257`) |
| Event-driven replanning at completion (row 15) | §III, §V-A | no timer: the runtime calls `choose_action` after every completion (`on_control_tick` is bookkeeping only, `:466-467`) |
| P95 latency profile consumed by the scheduler (row 3) | §III, §IV-B | partially: the public view publishes one service table (the frozen measured mean), so `L` is mean-based; P95/occupancy enters only through `busy_remaining_ns` (`group_runtime_t011.py:2822-2824`). Documented deviation, see §8 |

---

## 3. The dual-GPU adaptation (audit row 13) — the required adaptation

The paper assumes **one** GPU: "serving model `m` … monopolizes the GPU" and no
device index appears anywhere in sections I–VIII. Our runtime has two cards that can
execute concurrently, so the paper's single-increment queue update must be
reinterpreted. As specified by the audit's row 13 and the Lead's dispatch, the
adaptation is:

    Δ_a = min over eligible cards g of [ max(0, R_g) + L_a ]

* `R_g` = the card's residual busy time, read as `max(0, card.busy_remaining_ns)`
  (`edge_serving_adapter.py:379-386`). The kernel publishes that field as
  `max(0, budget_finish_ns − now)` where
  `budget_finish_ns = now + gate_ns[g][kind] + gpu_ipc_budget[g]`
  (plus `gate_ns[g]['FAST_SUFFIX']` when the running job is an inline prefix) —
  `group_runtime_t011.py:2822-2824`. So `R_g` is **occupancy** time: card service +
  IPC, and the committed suffix for an inline prefix.
* `L_a` is the candidate's own service table row + the card's IPC leg
  (`_duration_ns` `:360-377`); composite for a FAST_PREFIX.
* **Placement rule**: the candidate goes to the card minimising `R_g + L_a`, i.e. the
  earlier-available card (row 13's "earliest available GPU's projected availability").
  A pinned Fast Suffix is constrained to its own card and simply waits
  (`edge_serving_adapter.py:344-350`), matching the paper's non-preemptive exclusive
  discipline per GPU (row 18).
* **Unit disclosure** (mixed units are deliberate): `R_g` includes the IPC leg and is
  gate(P95)-based, `L_a` is the frozen measured service table + IPC. Both are the
  quantities the frozen kernel books and the public API publishes; we do not re-derive
  either from private state.

---

## 4. Common-horizon rule and the collapse identity (must-read caveat)

The audit's *Consistent evaluation horizon requirement* forbids scoring different
candidates at different absolute times. We implement its "cleaner and defensible"
variant (items 1+2), which is also the structure of V-C's own three bullets:

* **One common increment per round** — `H = max_a (R_a + L_a)`, the worst-case
  candidate completion of the round (it reduces to the audit item 4 permitted
  fallback `max_a L_a` when both cards are free).
* Two-stage projection: (i) the candidate's own completion deletes the served
  component and adds the row-13 delay; (ii) residual work is carried forward to `H`.
  Because `H` is common and arrivals are frozen, stage (ii) collapses to adding the
  same increment `H` to every surviving component of every candidate — which is
  exactly what `_projection_total` does (`edge_serving_adapter.py:411-443`).
* Same `H` and the same frozen-arrival assumption for all candidates; `H` is recorded
  per round in `ledger()['last_round']['horizon_ns']`; the config names the rule
  (`horizon_rule: common_horizon_common_increment`). Per-candidate absolute evaluation
  times are never used.

**Consequence, stated openly.** With a common `H` and `B*=1`, the projection total
`T(H)` does not depend on the candidate and exactly one term is deleted:

    S_a = T(H) − f(w_a + H)     ⟹     argmin_a S_a = argmax_a f(w_a + H) = argmax_a w_a

because `f` is strictly increasing. **The score therefore reduces exactly to serving
the longest-waiting component** (earliest arrival across queues), with exact ties
whenever two candidates have equal `w`. The arm still computes and records the honest
Eq. 3/4 value for every candidate (`ledger()['last_round']['candidates'][*]['score']`),
and `check_score_matches_independent_paper_computation` proves the recorded numbers
equal an independently coded Eq. 3/4 evaluation, but no reader should credit this arm
with more than a deadline-blind oldest-first rule at `B=1`.

That is not a defect of the port; it is the audit's own degeneracy finding, and it has
three consequences the Lead must carry into the write-up:

1. With `B_max=1`, Eq. 5 is a no-op and the deletion term is a single job, so every
   non-empty queue is an exact tie at any common horizon. **The decision falls
   entirely to the tie-break.**
2. The paper never validated the score at `B=1`: its own `Ours+bs=1` arm is a
   **negative ablation** ("single-task execution underutilizes GPU parallelism and
   reduces throughput"). Our `B=1` result therefore tests an extrapolation, not the
   paper's validated operating point (`B_max=10`).
3. The rule is **not EDF** (it ignores slack) and **not plain SJF** (it ignores
   duration); at equal waits the frozen tie-break's first element makes it choose the
   cheaper component, which is the closest resemblance to LQF-style reasoning.

---

## 5. Tie-breaking (audit row 16 = the single UNSUPPORTED mechanism)

The paper's `argmin` has **no** tie rule ("tie"/"tie-breaking" appear zero times in
the HTML), yet row 14's own analysis shows ties are the norm at `B=1`. We therefore
supply one, and it is the frozen kernel's own public order (Lead decision), so it can
never be tuned to favour this arm:

    (S_m, predicted service work, arrival_ns for FAST_* / enqueue_ns for ACCURATE, card index)

and, only when all four are equal, the kernel job id as a deterministic total-order
breaker (Python's `min` is otherwise iteration-order dependent).
Implemented at `edge_serving_adapter.py:447-455`; the same order appears in
`quota_forecast_v8.dispatch_choice` (`selected = min(candidates, key=lambda x: x[:3])`
with `x = (work, arrival_ns, card)`), i.e. it is the kernel's own arm-neutral ordering,
not ours.

**Why it cannot favour `edge_serving`:** (a) the order is computed purely from the
public cost table and public arrival/enqueue stamps, identical for every arm; (b) the
chosen candidate is `argmax w`, which is arm-invariant under the collapse; (c) the
extra `job_id` element is arbitrary and only reachable after every frozen element ties;
(d) the counts are reported.

`ledger()` reports `ties_broken` and a breakdown of which element decided
(`tie_break_used`). The enumerated conformity grid in the unit tests produced
**24 score ties over 54 states**, each resolved by the frozen order.

---

## 6. Fidelity ledger (the audit's 25 rows, reproduced)

| # | Mechanism | Tag | How it maps to this arm |
|---|---|---|---|
| 1 | Queue model (one FIFO queue per model, single GPU, exclusive non-preemptive) | ORIGINAL | one queue per executable component kind, per card for pinned suffixes; exclusive per card |
| 2 | `w_{m,i}` = elapsed since arrival | ORIGINAL | `JobView.waiting_ns`; `tau` uses the same stamp |
| 3 | Offline profile table `L(m,e,B)`, scheduler consumes P95 | ORIGINAL | frozen public service table (mean) + IPC; P95 only via `R_g` — documented deviation, §8 |
| 4 | Urgency `f` with clip `C` | ORIGINAL | `_f` `:403-409`, projected `:411-443`, verbatim `min(exp(w/τ−1), C)` |
| 5 | Numeric clip constant `C` | **ADAPTED** | `C = 10e = 27.18281828459045` (Lead-locked); arithmetic in §7 |
| 6 | Stability score `S` = sum over all queues | ORIGINAL | `_projection_total` over every pending component of both roles and both cards |
| 7 | Score semantics: lower = more stable, minimise | ORIGINAL | `min`-selection on `S_a` |
| 8 | Batch-size rule `B*=min(|Q_m|,B_max)` | **ADAPTED** | `batch_max=1` applied, not dropped — therefore a no-op (§4) |
| 9 | Candidate enumeration over non-empty queues | ORIGINAL | `_queue_heads` over the component-kind queues |
| 10 | Exit selection `e*=argmax e` s.t. `w_max+L≤τ` | **ADAPTED** | two branches only (Fast / Accurate); for a Prefix the constraint is evaluated on the composite duration |
| 11 | Prefix → reveal → Suffix execution | **ADAPTED** | composite candidate duration `409954+1182290=1592244` (+IPC); no paper counterpart |
| 12 | Post-decision update (served removed; own +L; every other queue +L; no arrivals) | ORIGINAL | `T(H)` deletes exactly the served term and adds the common increment |
| 13 | Multi-GPU residual work | **ADAPTED** | `Δ_a = min_g [max(0,R_g) + L_a]`, §3 |
| 14 | `m*=argmin S_m`, served batch excluded, no arrivals | ORIGINAL | `:231-233`, `:251-257`; behaviour at `B=1` documented in §4 |
| 15 | Event-driven decision at batch completion; frozen-arrival round | ORIGINAL | no timer; the runtime re-invokes us per completion; arrivals never modelled |
| 16 | Tie-breaking | **UNSUPPORTED** | supplied from the frozen kernel's public order, §5 |
| 17 | Exhaustive fixed profile, no re-profiling | ORIGINAL | the public service table is fixed per run |
| 18 | Non-preemptive exclusive GPU | ORIGINAL | one component per card at a time, enforced by the runtime |
| 19 | Spatial sharing rejected (time division) | ORIGINAL | one execution component per card; no co-tenancy |
| 20 | Preemption (none in the paper) | **ADAPTED** | our non-preemptive two-component executor is a confirmed match, not a port |
| 21 | Violation ratio `V` over a serving window | ORIGINAL | measured by the frozen harness, not by the arm |
| 22 | P95 tail latency | ORIGINAL | same |
| 23 | Effective accuracy via exit-point table lookup | ORIGINAL | procedure ports; Table I (CIFAR-100) is not usable for our quality model |
| 24 | Fixed window + warmup discipline | ORIGINAL | the frozen harness enforces 20 s window / 100-task warmup |
| 25 | Contrast set (LQF, EDF, All-Final, bs=1) | ORIGINAL | the Lead's matrix: the other literature arms + the frozen neutral axis |

**Tally: ORIGINAL 18 (rows 1,2,3,4,6,7,9,12,14,15,17,18,19,21,22,23,24,25) ·
ADAPTED 6 (rows 5,8,10,11,13,20) · UNSUPPORTED 1 (row 16).**

---

## 7. The clip constant, arithmetically resolved

The paper writes `f(w)=min(exp(w/τ−1), C)` (`_paper_cache/edgeserving_full.txt:213`)
and explains the normalization "it equals 0 at `w=τ` … so `f(τ)=exp(0)=1`" (line 218)
— the `−1` is inside the exponent. It then illustrates the clip's purpose: "tasks
already far beyond the SLO (e.g., `w>τ(1+ln 10)≈3.3τ`) … cannot be rescued" (line
219). **`C`'s numeric value is never stated anywhere in the paper.**

Two arithmetically consistent readings exist, and they must not be confused:

* `C = 10`: then `min()` saturates exactly at the illustrated point `τ(1+ln 10)≈3.3τ`,
  where `exp(w/τ−1)=10`.
* `C = 10e = 27.18281828459045` (**the Lead-locked value**): then saturation happens at
  `w ≥ τ(1+ln C) = τ(2+ln 10) ≈ 4.3026τ`.

We keep the locked `10e` and implement the paper's `min()` exactly as written, i.e. the
clipped branch is taken when `exp(w/τ−1) ≥ C`. The difference is immaterial at runtime
(§ below), and the constant is reported in the ledger (`clip_C`, `clip_log_C`) and the
config so no reader has to guess.

**Why it never activates in this simulator.** The kernel's `_expire_ready()`
(`group_runtime_t011.py:2744-2762`) removes a ready component the moment
`now > deadline_abs_ns`, and `τ` is the *whole* 50 ms per-task budget measured from
generation. Every component we can ever observe therefore has `w ≤ τ`, so the
unprojected weight is `f ≤ 1`; the projection adds at most the round horizon `H`
(≈3.1 ms), leaving `w+H ≤ 1.07τ`, four times below the 4.3026τ saturation point. The
ledger records `clip_activated_terms = 0` and the observed projected `f` range
(`f_range_projected`), and the unit tests exercise the clipped branch *arithmetically*
(`check_clip_constants_and_clipped_branch_arithmetic`: `f(3.3τ)=10` unclipped,
`f(4.3026τ)= C` clipped, `f(50τ)=f(1000τ)=C` without overflow).

---

## 8. Kernel facts this arm relies on (and deliberate non-uses)

* **Ingress classification and FIFO order** — `group_runtime_t011.py:1418-1497`
  (`push`): FAST_SUFFIX to the card's pinned list, anything else to
  `ready['Accurate' if kind=='ACCURATE' else 'Fast']`; document order per role is
  arrival / enqueue (`lit_api.py:240-316`).
* **Inline Fast Prefix** — the formal path forces it on
  (`pure_simulation.py:344`, `:841`): `group_runtime_t011.py:1486-1487` marks the
  prefix, `:2822-2824` charges `gate(prefix) + ipc + gate(suffix)` as one occupancy,
  and the suffix is executed inside the kernel without a second scheduling call
  (`:2710` only pins suffixes in the non-inline mode; `:2715-2741` records it as stage
  telemetry). Our composite `L` mirrors exactly that occupancy.
* **Expiry** — `group_runtime_t011.py:2744-2762` (ready) and `:2764-2790` (pinned):
  the reason `w ≤ τ` and the clip is unreachable (§7).
* **IPC leg** — `group_runtime_t011.py:780` `gpu_ipc_budget = [400_000] * CARD_COUNT`,
  folded into the booked cost at `:829`.
* **Kernel-side admission** — a late Accurate continuation can be rejected inside the
  kernel (`_route_b`, "remaining deadline insufficient" → Abstain). The arm never
  re-implements admission; it only ever sees published components.
* **Deliberate non-uses.** No network legs in `L` (6.5 ms up / 0.5 ms back are outside
  the paper's scope and are the runner's business; the config states this explicitly).
  No completion feedback is consumed (counted only, `on_completion` `:469-482`). No
  future arrivals, no `raw`, no label/probability/phase field, no view mutation — the
  arm is a pure function of the public view, and the frozen static leakage audit
  returns `[]` for this file (`check_static_leakage_audit_is_clean`).
* **Row 3 deviation, restated**: the paper has its scheduler consume the P95 profile;
  the frozen public view publishes a single service table (the measured mean) plus
  `ipc_ns`, so our `L` is mean-based and P95-like information reaches us only through
  the occupancy residual `R_g`. We use the public quantities as published rather than
  reconstructing a P95 table from private state.

---

## 9. Configuration (`config_edge_serving.json`)

| Key | Value | Meaning |
|---|---|---|
| `batch_max` | 1 | Eq. 5 applied; the environment locks batch 1 |
| `gpu_cards` | 2 | dual-GPU adaptation (row 13) |
| `deadline_ns` | 50000000 | τ when a job carries no per-task deadline |
| `clip_C` / `clip_log_C` | 27.18281828459045 / 3.302585092994046 | Eq. 3 clip and `ln C` (§7) |
| `eq6_deadline_filter` | true | apply Eq. 6 residual-feasibility filter |
| `eq6_fallback_serve` | true | when *no* candidate is feasible, serve the score's argmin and count it (false = idle the cards) |
| `horizon_rule` | `common_horizon_common_increment` | audit items 1+2, collapsing to item 4 |
| `gpu_service_ns` / `gpu_ipc_budget_ns` | 1422843 / 409954 / 1182290 / 2735549, 400000 | documented fallbacks; the live values come from the public view |
| `network_upload_ns` / `network_return_ns` | 6500000 / 500000 | recorded, **not** part of `L` |

The runner merges `lit_config.json['common']` with this file's flat keys
(`run_literature.py:_adapter_config`), so the values above are the arm parameters in
force; the adapter also accepts the nested `{'common':…, 'arms':{…}}` shape.

---

## 10. Local verification (no server, no kernel)

```
D:\wzy\python\python.exe literature_baselines\edge_serving_unit_tests.py
-> 16/16 checks passed, exit 0
```

| Check | Proves |
|---|---|
| `module_registers_through_the_runners_path` | `lit_api.load_adapter('edge_serving')` resolves the arm (module-level `register`, no main guard) |
| `static_leakage_audit_is_clean` | `lit_api.audit_adapter_source` → `[]`; this doc exists |
| `clip_constants_and_clipped_branch_arithmetic` | Eq. 3 verbatim; `f(τ)=1`, `f(3.3τ)=10`, clipped branch at `4.3026τ`, no overflow |
| `clip_is_unreachable_for_legal_components` | `w ≤ τ` for every observed candidate; `clip_activated_terms = 0` |
| `picks_longest_waiting_component` | the §4 identity on a real candidate set |
| `placement_uses_least_loaded_free_card` | row 13: the busier card is skipped, `Δ = R + L` |
| `inline_prefix_carries_its_suffix_commitment` | row 11: one action, composite `1592244 + ipc` |
| `pinned_suffix_only_on_its_own_card` | row 18/13 affinity; idle until its card frees |
| `tie_break_is_the_frozen_public_order` | all four tie elements exercised in order, counts recorded |
| `eq6_deadline_filter_changes_the_choice` | Eq. 6 skips a longer-waiting but infeasible candidate; fallback and idle modes |
| `score_matches_independent_paper_computation` | recorded `S_a` == independently coded Eq. 3/4 (4 candidates, `H=3135549`) |
| `choice_is_argmin_score_and_argmax_wait` | 54 enumerated states: `argmin S == argmax w` every time (24 ties) |
| `never_uses_future_arrivals_mutations_or_feedback` | determinism, no view mutation, feedback ignored, queue-order independent |
| `waits_use_arrival_and_tau_is_the_task_budget` | Eq. 1 semantics; per-task `τ = deadline − arrival` |
| `idle_and_empty_rounds_return_none` | `None` on empty queue / no free card / no view |
| `ledger_is_json_serialisable_and_complete` | the ledger is JSON-safe and complete for the arm audit |

---

## 11. Open items handed to the Lead

1. **§4 degeneracy is the headline finding**: at `B=1` this port is a
   longest-waiting-component rule decided by the tie-break. The paper's validated
   configuration is `B_max=10`; its `bs=1` arm is a negative ablation. Attribute the
   arm's measured result accordingly.
2. **Row 5 / §7**: `C=10e` is our derived value, not the paper's; the paper's own
   illustrated point corresponds to `C=10`. No behavioural difference here (the clip is
   unreachable), but the write-up must say "derived, never stated".
3. **Row 3 deviation (§8)**: our `L` is the mean service table the frozen view
   publishes, not the paper's P95 profile number.
4. No server run was performed by this task; the Lead schedules the formal run and owns
   the single output root.
