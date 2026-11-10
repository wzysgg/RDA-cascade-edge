# RDA：基于可兑现密度的边缘级联推理 GPU 分配与批调度

论文《基于可兑现密度的边缘级联推理GPU分配与批调度》（投稿《计算机工程》）的源代码：可兑现密度分配方法（RDA）、5 种对比方法的边缘适配，以及两张 GPU 上的真实批执行器。

Source code of *GPU Allocation and Batch Scheduling for Cascaded Edge Inference Based on Realizable Density* (RDA), the edge adaptations of five baselines and the real two-GPU batch executor.

## 从哪里读起

| 论文 | 代码 |
|---|---|
| 2.2 可兑现准入（RA）：释放已来不及的请求、开工前复检 | `realizable_admission.py`（`NeedTable.floor`、`median`）；`live_batch_runtime.py` 中的 `purge_expired`、`batch_unrealizable`、`schedule` |
| 2.2 返回端测时（RT）：所需时间计到结果真实返回 | `live_batch_runtime.py` 中 `need.add(..., returned_ns - dispatch_ns, ...)` |
| 2.3 流体背包分配（FKA）：需求工作量率、容量检验、过载时的分数背包 | `queue_feedback_controller.py` 中的 `overload_unit_cost_target` 与 `_direct_demand_tick`；检查时机见 `dual_clock_state.py` |
| 2.3 份额记分卡（式 10） | `quota_forecast_v8.py` 中的 `_service_turn`、`_debt_service`；分卡见 `resource_axis.card_shares` |
| 2.3 可行性保持交换（FPS） | `live_batch_policies.py` 中的 `OursBatchPolicy.urgency_swap` |
| 2.4 最晚可行派发（LFD） | `OursBatchPolicy.priority_turn`、`priority_due`、`next_opportunity` |
| 2.4 密度组批（DBP，式 12） | `OursBatchPolicy.density_pack` |
| 2.4 安全-乐观所需时间（SON，式 13、14） | `realizable_admission.py` 中的 `NeedTable.upper`、`lower`；`OursBatchPolicy.safe_need`、`choice_need` |
| 算法 1：单卡派发决策 | `OursBatchPolicy.choose` 与 `live_batch_runtime.py` 中的 `schedule` |
| 3.3 对比方法 | `literature_baselines/`：`edge_serving_adapter.py`（EdgeServing）、`serving_baselines.py`（PARD、Proteus、SlackFit、RAMSIS）、`ramsis_policies.py` |
| 表 2 参数 | `config/live_gpu_batch.json`；`run_experiment.py` |

`run_experiment.py` 给出论文中每个方法与消融、敏感性配置的名称和对应开关，例如：

```bash
python run_experiment.py --method RDA --manifest <scene>/manifest.json --output out/
python run_experiment.py --method PARD --calibration <calibration.json> --manifest <scene>/manifest.json --output out/
```

## 目录说明

- 根目录：执行器（`live_batch_runtime.py`）、RDA 调度策略（`live_batch_policies.py`）、所需时间表（`realizable_admission.py`）、分卡控制（`queue_feedback_controller.py`、`dual_clock_state.py`）及其依赖模块。
- `literature_baselines/`：5 种对比方法的边缘适配。所有方法共用同一执行器、最大批大小 10 与同一离线标定，只有调度策略不同。
- `tools/run_live_batch.py`：启动一次真实 GPU 运行；`tools/analyze_live_batch.py`：运行结束后离线计算按时正确率等指标。
- `config/`、`evidence/calibration/`：参数与离线标定（离线标定只用于冷启动和对比方法）。
- `evidence/evaluation_private_labels.json`：评测图像的真实标签，只由 `live_batch_statistics.py` 在运行结束后离线统计正确率时读取，在线调度不读取。
- 根目录中文件名带 `t011`、`t012`、`v8`、`v20`、`v24` 的模块是执行器早期版本留下的公共部件（模型加载、质量路由、份额记分等），被上述文件导入，因此保留。

## 运行环境

- Linux，2×NVIDIA RTX 2080 Ti；论文实验为 Ubuntu 20.04.6、Python 3.8.10、PyTorch 2.4.1、CUDA 12.1。依赖见 `requirements.txt`。
- 未包含：NEU 热轧钢带表面缺陷数据集（请从官方渠道获取）、由 ImageNet 预训练权重在 NEU 开发集上微调的模型权重、冻结的场景输入与运行日志（体积过大）。
- RAMSIS 的离线策略表已提供（`literature_baselines/ramsis_policies_20261021.json`）；重新求解策略需要作者代码 <https://github.com/dmmendo/RAMSIS>（commit `9b74967`），放到 `literature_baselines/vendor/ramsis/`。其余对比方法的原文与代码来源见 `literature_baselines/AUTHOR_SOURCES.json`。

## 许可证

MIT，见 `LICENSE`。
