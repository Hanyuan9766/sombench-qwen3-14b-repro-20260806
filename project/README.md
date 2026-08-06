# SocialMind / SoMBench Qwen3-14B

这是面向 2026 SocialMind 模型提交赛道的完整、可复现实现。正式产物是
Qwen3-14B 的合并 BF16 Hugging Face/ModelScope 仓库，不依赖在线 RAG、Agent、
外部 API 或未合并 Adapter。

## 已固定的技术协议

- 基座：`Qwen/Qwen3-14B`（后训练/对话模型，不是 `-Base`）
- 训练：单卡 A800 80GB，NF4 QLoRA，BF16 计算，LoRA r=64/alpha=128
- 训练长度：4096，assistant-only loss，71 个细维度均衡采样
- 官方推理：`cot`, temperature=0.6, top_p=0.95, max_tokens=8192, n=1
- 官方兼容：Transformers 4.51.3，vLLM 0.8.5.post1
- 提交：完整 Safetensors 权重、tokenizer/chat template、`config.json`

## 工程结构

- `data_pipeline/`：真实 schema 校验、故事簇防泄漏切分、CoT 压缩、71 维均衡、公开标签冲突审计
- `training/`：QLoRA、断点续训、BF16 合并、官方参数离线评测、仓库结构校验
- `infra/`：服务器目录、隔离环境、模型下载、token 审计、vLLM 启动和冒烟验证
- `modelscope/`：干净发布目录、SHA-256 manifest、无凭据落盘上传、远端回下载验证
- `tests/`：不依赖 GPU 的数据与训练纯函数测试
- `analysis/leaderboard_20260805.md`：榜单权重、增益分解与两阶段训练决策

## 当前服务器目录

```text
/root/autodl-tmp/sombench/
├── project/                  # 本工程
├── datasets/raw/             # 两个官方原始 JSONL
├── datasets/processed/v1/    # 清洗、切分、均衡与审计产物
├── models/Qwen3-14B/         # 固定基座快照
├── outputs/                  # adapter、checkpoint、评测结果
├── merged_model/             # 正式完整 BF16 合并仓库
├── release_model/            # 最终 ModelScope 上传目录（硬链接，不重复占权重空间）
├── envs/hub/                 # 轻量 ModelScope 下载/上传环境
├── envs/train/               # 训练环境
├── envs/eval/                # 官方版本 vLLM 环境
└── logs/
```

## 执行顺序

所有命令在服务器执行。

```bash
cd /root/autodl-tmp/sombench/project
bash infra/bootstrap_server.sh
bash infra/setup_hub_env.sh
bash infra/setup_train_env.sh
bash infra/prepare_data.sh
bash infra/download_base_model.sh
bash infra/install_flash_attn.sh

# 精确 tokenizer 全量长度审计；必须 total_overlength=0。
/root/autodl-tmp/sombench/envs/train/bin/python infra/audit_tokens.py \
  --model /root/autodl-tmp/sombench/models/Qwen3-14B \
  --input /root/autodl-tmp/sombench/datasets/processed/v1/train_balanced.jsonl \
  --input /root/autodl-tmp/sombench/datasets/processed/v1/dev.jsonl \
  --max-length 4096 \
  --output /root/autodl-tmp/sombench/datasets/processed/v1/token_audit.json

# 两步 CUDA 冒烟训练和完整合并。
/root/autodl-tmp/sombench/envs/train/bin/python -m training.train \
  --config training/configs/qwen3_14b_qlora_smoke.yaml
/root/autodl-tmp/sombench/envs/train/bin/python -m training.merge \
  --config training/configs/merge_qwen3_14b_smoke.yaml

# 正式训练采用两阶段课程：先覆盖全部 3,371 条独立训练行，再以更低
# 学习率在 2,982 条 71 维均衡集上校准。
/root/autodl-tmp/sombench/envs/train/bin/python -m training.train \
  --config training/configs/qwen3_14b_qlora_stage1.yaml
/root/autodl-tmp/sombench/envs/train/bin/python -m training.train \
  --config training/configs/qwen3_14b_qlora.yaml
/root/autodl-tmp/sombench/envs/train/bin/python -m training.merge \
  --config training/configs/merge_qwen3_14b.yaml

# 官方兼容环境与本地服务。
bash infra/setup_eval_env.sh
bash infra/launch_vllm.sh /root/autodl-tmp/sombench/merged_model
```

另一个终端执行：

```bash
/root/autodl-tmp/sombench/envs/eval/bin/python infra/smoke_vllm.py
/root/autodl-tmp/sombench/envs/eval/bin/python -m training.evaluate \
  --config training/configs/eval_dev_vllm.yaml
/root/autodl-tmp/sombench/envs/eval/bin/python -m training.evaluate \
  --config training/configs/eval_vllm.yaml
```

以按故事簇隔离的 375 条 `dev.jsonl` 作为模型选择依据。公开集的 59/71 个故事与
参考训练集精确重叠，且其中 46 个故事进入训练子集，因此公开集只用于协议、解析和
服务兼容性诊断，不能作为独立泛化成绩。

## ModelScope 发布

先构建与校验干净仓库：

```bash
/root/autodl-tmp/sombench/envs/eval/bin/python modelscope/prepare_release.py \
  /root/autodl-tmp/sombench/merged_model \
  /root/autodl-tmp/sombench/release_model \
  --validator infra/validate_hf_repo.py \
  --source-revision-manifest /root/autodl-tmp/sombench/models/Qwen3-14B.source_revision.json \
  --base-revision master \
  --repo-id Bithyhy/sombench-qwen3-14b-a-v1 \
  --upload-revision master \
  --release-revision sombench-a-20260805-v1 \
  --training-summary /root/autodl-tmp/sombench/outputs/qwen3-14b-qlora-v1/train_results.json
```

上传时才临时注入凭据，不调用持久化登录：

```bash
export MODELSCOPE_MODEL_ID='Bithyhy/sombench-qwen3-14b-a-v1'
export MODELSCOPE_API_TOKEN='TEMPORARY_WRITE_TOKEN'
/root/autodl-tmp/sombench/envs/eval/bin/python modelscope/publish.py \
  /root/autodl-tmp/sombench/release_model \
  --tag sombench-a-20260805-v1
unset MODELSCOPE_API_TOKEN
```

然后用固定 tag 回下载到干净目录，再做结构校验与 vLLM 冒烟。赛事提交 revision 使用
固定 tag，不使用 `main` 或 `latest`。

## 已知待补内容

这些不阻塞 Q1-Q3 的首个完整模型：

1. 官方是否在 A/B 隐藏集纳入 Q4；当前上传训练集明确为 `without_Q4`，不能伪造为已覆盖。
2. 公开 213 条中 77 条需要人工裁定；主指标只报告严格安全的 136 条，冲突题不用于调参标签。
3. 主办方未公开的 GPU、完整 `vllm serve` 命令、stop/EOS 和整体超时限制。

`models/Qwen3-14B.source_revision.json` 会保存下载前由 ModelScope 返回的 revision
元数据、19 个文件 SHA-256 与整体内容指纹；下载后还会逐文件回验。ModelScope 当前
接口没有暴露该 `master` 的 Git commit，因此正式模型卡会同时如实记录请求 revision
和不可变内容指纹，而不会虚构 commit。
