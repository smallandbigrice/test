# main.py 更新规则

本文档用于记录本仓库后续修改 `main.py` 时必须遵守的流程。

## 固定要求

每次更新 `main.py` 后，必须完成以下两件事：

1. 推送到 GitHub 仓库。
2. 新增或更新一份 Markdown 修改说明，说明本次改动内容。

## 修改说明要求

每次 `main.py` 改动后，需要在 Markdown 中写清楚：

- 修改日期。
- 修改目标。
- 改动了哪些算法逻辑。
- 是否影响帧差、蒙版、ROI、YOLO 推理、轨迹判断、绿框确认逻辑。
- 默认参数是否变化。
- 是否做过本地视频推理验证。
- 已知问题或后续需要继续优化的点。

## 推荐文件命名

建议每次更新时新增一份独立说明，放在 `docs/main_updates/` 目录下：

```text
docs/main_updates/YYYYMMDD_简短说明.md
```

示例：

```text
docs/main_updates/20260614_frame_diff_runtime.md
```

## GitHub 推送要求

每次 `main.py` 修改完成后，建议提交内容至少包含：

- `main.py`
- 本次 Markdown 修改说明

提交信息建议写清楚版本目的，例如：

```text
update main frame-diff runtime
```

推送目标仓库：

```text
https://github.com/smallandbigrice/test.git
```

当前默认分支：

```text
main
```

## 执行原则

如果只是测试脚本、数据处理脚本、临时推理脚本发生变化，不强制触发本规则。

只要 `main.py` 被修改，就必须同步写说明并推送 GitHub。
