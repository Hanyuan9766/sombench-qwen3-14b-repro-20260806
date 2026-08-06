# 恢复说明

## 1. 下载资产

从本仓库的私有 Release 下载以下文件到同一目录：

```text
SoMBench_repro_backup_20260806.tar.part-00
SoMBench_repro_backup_20260806.tar.part-01
SoMBench_repro_backup_20260806.tar.part-02
SoMBench_repro_backup_20260806.parts.sha256
SoMBench_repro_backup_20260806.tar.sha256
```

## 2. 校验三个分卷

```bash
sha256sum -c SoMBench_repro_backup_20260806.parts.sha256
```

所有条目必须显示 `OK`。校验失败时不要继续解包，应重新下载对应分卷。

## 3. 按顺序合并归档

```bash
cat \
  SoMBench_repro_backup_20260806.tar.part-00 \
  SoMBench_repro_backup_20260806.tar.part-01 \
  SoMBench_repro_backup_20260806.tar.part-02 \
  > SoMBench_repro_backup_20260806.tar
```

## 4. 校验完整归档

```bash
sha256sum -c SoMBench_repro_backup_20260806.tar.sha256
```

## 5. 解包

```bash
mkdir -p restored
tar -xf SoMBench_repro_backup_20260806.tar -C restored
```

解包后，备份根目录应包含项目代码、数据、评测证据、日志、交付材料、三阶段最终 adapters，以及 Stage3 `checkpoint-66`。

## 6. 模型与环境

正式模型从 ModelScope 固定标签 `sombench-a-20260805-v2` 获取，对应提交为 `41f6400b07a4f3003bcaf20bf0c9267f921ded88`。目标评测环境应保持 `Transformers 4.51.3` 和 `vLLM 0.8.5.post1`；该组合已通过加载、服务启动和推理验证。

恢复过程不需要、也不应把任何访问令牌写入仓库。临时访问凭据应仅通过环境变量或受控凭据存储传入，并在使用后撤销。
