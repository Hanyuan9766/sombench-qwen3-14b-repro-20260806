# SoMBench Qwen3-14B Stage3 v2 可复现备份

本私有仓库用于保存 SoMBench Qwen3-14B Stage3 v2 的训练代码、数据、评测证据和可恢复训练产物。Git 仓库只保存适合版本控制的代码、文档与校验清单；约数 GB 的备份归档存放在本仓库的私有 GitHub Release 中。

## 正式模型

- ModelScope 仓库：`Bithyhy/sombench-qwen3-14b-a-v1`
- 固定标签：`sombench-a-20260805-v2`
- 提交：`41f6400b07a4f3003bcaf20bf0c9267f921ded88`
- 评测兼容性：已在 `Transformers 4.51.3` 与 `vLLM 0.8.5.post1` 环境完成加载、`vllm serve` 和推理验证。

正式可提交模型仍以 ModelScope 上述固定版本为准。本仓库用于实验复现和灾难恢复，不替代 ModelScope 模型仓库。

## 备份内容

Release 归档包含：

- `project`：训练、评测、合并和发布相关代码及配置；
- `datasets`：训练和公开测试数据快照；
- `evidence`、`logs`、`delivery_20260805`：验证证据、关键日志和交付材料；
- `outputs/eval-*` 及顶层评测结果；
- 三阶段训练的最终 adapters；
- Stage3 可续训检查点 `checkpoint-66`。

为控制备份体积，归档不包含：

- Qwen3-14B 基础模型；
- merged 与 verified 模型副本；
- 模型缓存、Python/Conda 环境、下载目录；
- smoke 测试生成的模型副本；
- 各阶段旧检查点，包括 Stage3 `checkpoint-50`。

这些排除项可由正式 ModelScope 模型、基础模型来源及仓库内脚本重新生成。

## Release 归档

完整归档拆分为三个 Release 资产：

```text
SoMBench_repro_backup_20260806.tar.part-00
SoMBench_repro_backup_20260806.tar.part-01
SoMBench_repro_backup_20260806.tar.part-02
```

下载三卷文件及 SHA-256 清单后，按 [RESTORE.md](RESTORE.md) 完成校验、合并和解包。不要把归档分卷或模型权重直接提交到 Git 历史。

仓库内的 [`manifests/FILES.sha256`](manifests/FILES.sha256) 可用于恢复后的逐文件校验；[`manifests/BACKUP_METADATA.json`](manifests/BACKUP_METADATA.json) 记录归档大小、哈希、正式模型固定版本和可续训检查点。

## 安全说明

备份制作前已检查常见凭据与已知临时凭据。本仓库及 Release 不应保存访问令牌、密码、私钥或服务端登录信息；如发现疑似凭据，应先撤销并清理后再更新备份。
