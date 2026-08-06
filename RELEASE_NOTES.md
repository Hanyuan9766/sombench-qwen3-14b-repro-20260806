# SoMBench 可复现备份 — 2026-08-06

本 Release 是 SoMBench Qwen3-14B Stage3 v2 的私有、不可变恢复快照。

## 资产

- 三卷归档：`SoMBench_repro_backup_20260806.tar.part-00`、`part-01`、`part-02`；
- 分卷 SHA-256 清单：`SoMBench_repro_backup_20260806.parts.sha256`；
- 合并归档 SHA-256 清单：`SoMBench_repro_backup_20260806.tar.sha256`。

归档包含项目代码、数据集快照、评测结果与证据、日志、交付材料、三阶段最终 adapters，以及 Stage3 `checkpoint-66`。基础模型、merged/verified 副本、缓存、环境、下载目录、smoke 产物和旧检查点未纳入。

恢复步骤见 `RESTORE.md`。正式模型继续由 ModelScope `Bithyhy/sombench-qwen3-14b-a-v1` 的固定标签 `sombench-a-20260805-v2` 提供，对应提交 `41f6400b07a4f3003bcaf20bf0c9267f921ded88`。

兼容性已在 `Transformers 4.51.3` 与 `vLLM 0.8.5.post1` 上验证。本 Release 不应包含任何访问令牌、密码或私钥。
