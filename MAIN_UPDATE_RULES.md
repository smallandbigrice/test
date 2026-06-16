# main.py 更新规则

本文档用于约束本仓库后续对 `main.py` 的修改流程，避免出现“代码已经本地改了，但没有同步到 GitHub，也没有留下版本说明”的情况。

## 固定要求

每次修改 `main.py` 后，必须同时完成以下两件事：

1. 推送到 GitHub 仓库。
2. 新增或更新一份 Markdown 修改说明，记录本次改动内容。

只要 `main.py` 被修改，这两项都不能省。

## 修改说明要求

每次 `main.py` 改动后，Markdown 说明至少要写清楚：

- 修改日期
- 修改目标
- 改动了哪些算法逻辑
- 是否影响帧差、蒙版、ROI、YOLO 推理、轨迹判断、绿框确认逻辑
- 默认参数是否变化
- 是否做过本地视频推理验证
- 已知问题与后续待优化方向

## 说明文件命名

每次更新建议新增独立说明文件，放在 `docs/main_updates/` 目录下：

```text
docs/main_updates/YYYYMMDD_简短说明.md
```

示例：

```text
docs/main_updates/20260616_main_process_default_staggered_dynamic_mask.md
```

## Git 分支与推送要求

每次修改 `main.py` 后：

1. 使用新的独立分支提交，不直接覆盖旧分支历史。
2. 提交内容至少包含：
   - `main.py`
   - 本次 Markdown 修改说明
3. 推送到 GitHub 仓库：

```text
https://github.com/smallandbigrice/test.git
```

## 本地 Git 调用要求

以后凡是执行本地 Git 操作，必须先确认 `git` 的真实可执行路径，不能默认认为系统里“装过 Git”就一定能直接调用。

执行前必须先做以下检查：

1. 先运行 `Get-Command git` 检查 `git` 是否在 `PATH` 中。
2. 如果不在 `PATH` 中，必须继续确认本机真实的 `git.exe` 路径。
3. 若只能查到 Chocolatey 安装记录，但找不到真实 `git.exe`，则不能视为“本地 Git 可用”。

如果 `git` 不在 `PATH` 中，则后续命令、修改说明中都要明确记录使用的是“绝对路径调用”，避免下次重复踩坑。

当前这台机器已确认可用的本地 Git 路径为：

```text
C:\Program Files\Git\cmd\git.exe
```

后续如果 `git` 命令解析异常，优先直接使用这个绝对路径。

## main.py 修改完成的判定

以后 `main.py` 的一次修改，只有在以下条件全部满足后，才算真正完成：

1. 本地代码已改完。
2. Markdown 修改说明已写完。
3. 已完成 Git 提交与 GitHub 推送，或者已明确说明本次为什么无法推送。

只做到前两步，不能算完成。

## 视频推理验证原则

进行本地或 PC 端视频推理测试时，默认不直接修改 `main.py`。

视频推理阶段只允许修改：

- 临时测试脚本
- 推理命令参数
- 独立实验脚本

只有在视频推理效果确认可用后，才允许把对应逻辑整理后写回 `main.py`。

一旦写回 `main.py`，就必须同步：

1. 新建独立 Git 分支
2. 记录 Markdown 修改说明
3. 提交并推送到 GitHub，确保历史可回溯

## 视频推理环境

本机运行 `single_video_main_pt.py` 或其他 PT 视频推理脚本时，默认使用：

- Python: `D:\conda\python.exe`

运行前需要补齐 `PATH`：

- `E:\detect uav\_runtime_dlls`
- `D:\conda`
- `D:\conda\Library\bin`
- `D:\conda\Scripts`

推荐启动方式：

```powershell
$env:PYTHONNOUSERSITE='0'
$env:PATH='E:\detect uav\_runtime_dlls;D:\conda;D:\conda\Library\bin;D:\conda\Scripts;' + $env:PATH
python single_video_main_pt.py ...
```

说明：

- 当前默认 `python` 实际指向 `D:\conda\python.exe`
- `torch` 位于用户目录 `C:\Users\31379\AppData\Roaming\Python\Python311\site-packages`
- 如果不补齐上述 DLL 路径，`torch` 可能因 `shm.dll` 或其依赖缺失而无法启动
- 后续凡是做 PC 端 PT 模型视频推理，优先沿用这套环境配置
