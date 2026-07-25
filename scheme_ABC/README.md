# UAV 检测方案 ABC 本地归档

生成日期：2026-07-25

本目录用于固定当前本地电脑上的三套板端检测方案，把代码、模型、配置、数据集、原始视频、板端回放视频和切换脚本放在同一个归档根目录下。后续查找方案 A/B/C 时，优先从本目录进入，不再回到旧的零散路径里翻文件。

## 目录结构

- `code/`：当前可部署主代码快照。
- `configs/`：三套方案的 systemd 参数模板。
- `models/`：三套方案和候选模型。
- `datasets/`：训练数据集与来源数据。
- `videos/`：原始验证视频、板端录制视频和手工轨迹。
- `results/`：关键板端验证结果。
- `scripts/`：方案切换脚本备份。
- `asset_hashes.csv`：代码、配置、模型的 MD5 清单。
- `migration_20260721.csv`：第一批核心数据/视频迁移记录。
- `migration_extra_20260721.csv`：第二批补充数据/视频迁移记录。

## 方案 A：复杂场景500米

适用板子：板1、板2。

使用场景：低层复杂背景，画面存在建筑、树木、楼体边缘、增益噪声等干扰。

当前口径：复杂区域使用场景抑制，有效天空区域使用白天高层500米链路。检测输入保持 2K，视频回传保持低码率。

核心文件：

- 代码：`E:\detect uav\scheme_ABC\code\main.py`
- 切换脚本：`E:\detect uav\scheme_ABC\scripts\switch_scheme_a_complex500.ps1`
- 参数模板：`E:\detect uav\scheme_ABC\configs\scheme_A_complex500.conf`
- 主要模型：`E:\detect uav\scheme_ABC\models\yolov5s_day_20260626.rknn`
- 模型 MD5：`30261610759aec78c94cdf7d6f6510ce`

关联数据：

- `E:\detect uav\scheme_ABC\datasets\A_complex_scene_500m\run_014`
- `E:\detect uav\scheme_ABC\datasets\A_complex_scene_500m\run_014.zip`
- `E:\detect uav\scheme_ABC\datasets\A_complex_scene_500m\run_014_roi640`
- `E:\detect uav\scheme_ABC\datasets\A_complex_scene_500m\run_015.zip`
- `E:\detect uav\scheme_ABC\datasets\A_complex_scene_500m\run_017`
- `E:\detect uav\scheme_ABC\datasets\A_complex_scene_500m\run_017_roi640`

关联视频：

- `E:\detect uav\scheme_ABC\videos\original\video5_complex_scene.mp4`
- `E:\detect uav\scheme_ABC\videos\board_records\board9_h265_2k_20240825_214140`
- `E:\detect uav\scheme_ABC\videos\board_records\board9_run014_live2k_20240826_013700`

## 方案 B：白天高层500米

适用板子：板3到板9。

使用场景：白天高层、天空占比高、复杂背景少，目标距离远并存在悬停或慢速运动。

核心文件：

- 代码：`E:\detect uav\scheme_ABC\code\main.py`
- 切换脚本：`E:\detect uav\scheme_ABC\scripts\switch_high_layer_profiles.ps1 -Mode day`
- 参数模板：`E:\detect uav\scheme_ABC\configs\scheme_B_day_high500.conf`
- 模型：`E:\detect uav\scheme_ABC\models\yolov5s_day_20260626.rknn`
- 模型 MD5：`30261610759aec78c94cdf7d6f6510ce`

关联数据：

- `E:\detect uav\scheme_ABC\datasets\B_day_high500\datasets1_fusion_gray8_8_2`
- `E:\detect uav\scheme_ABC\datasets\B_day_high500\datasets1_fusion_gray8_8_2.zip`
- `E:\detect uav\scheme_ABC\datasets\B_day_high500\datasets_gray_base26_plus_0701_8_2_20260704`
- `E:\detect uav\scheme_ABC\datasets\B_day_high500\all`
- `E:\detect uav\scheme_ABC\datasets\B_day_high500\source_components`

关联视频：

- `E:\detect uav\scheme_ABC\videos\original\video4_day_high.mp4`
- `E:\detect uav\scheme_ABC\videos\original\video10_day_high.mkv`

## 方案 C：夜间无光全场景检测

适用板子：板1到板9，夜间无光或弱光模式。

使用场景：夜间高层、目标弱小、环境中存在星点/固定亮点干扰，要求视频回传流畅。

当前口径：夜间使用 H265 彩色摄像头输入，板端硬解后只取 NV12 的 Y 平面进入方案 C；采集和帧差均为 640x480；只有帧差产生运动点时送入 YOLO；围绕运动点裁剪 160 ROI 后放大到模型输入尺寸；关闭全帧兜底和悬停保持；开启静止亮点屏蔽；启动后 1 分钟内检测到的小目标登记为星点区域并按检测框大小屏蔽。

验证说明：2026-07-25 在板8上完成单路 H265 摄像头 Y 平面验证。正式封档仍保持原方案 C 的 5 摄像头结构，`UAV_H265_DEVICES=auto` 会按板端 H265 设备自动分配到各路摄像头；单路验证只作为输入链路确认，不改变后续多摄部署口径。

核心文件：

- 代码：`E:\detect uav\scheme_ABC\code\main.py`
- 切换脚本：`E:\detect uav\scheme_ABC\scripts\switch_high_layer_profiles.ps1 -Mode night`
- 参数模板：`E:\detect uav\scheme_ABC\configs\scheme_C_night_no_light_all_scene.conf`
- 模型：`E:\detect uav\scheme_ABC\models\yolov5s_night_latest.rknn`
- 模型 MD5：`003214966000be2605e8a0148790b931`

关联数据：

- `E:\detect uav\scheme_ABC\datasets\C_night400\8night`
- `E:\detect uav\scheme_ABC\datasets\C_night400\8night_roi640`
- `E:\detect uav\scheme_ABC\datasets\C_night400\night8_roi640_gray_for_cloud.zip`
- `E:\detect uav\scheme_ABC\datasets\C_night400\manual_video9`
- `E:\detect uav\scheme_ABC\datasets\C_night400\datasets1_fusion_gray8_9manual_8_2_fixed`
- `E:\detect uav\scheme_ABC\datasets\C_night400\datasets_gray_base26_plus_0701_night8_night9_8_2_20260704`

关联视频：

- `E:\detect uav\scheme_ABC\videos\original\video8_night.mp4`
- `E:\detect uav\scheme_ABC\videos\original\video9_night.mkv`

## 补充来源数据

旧目录中剩余的基础数据、抽帧数据、融合数据和中间合并缓存已经集中到：

- `E:\detect uav\scheme_ABC\datasets\source_components`

这里不是单独的一套运行方案，而是训练追溯时使用的数据来源池。

## 桌面入口

桌面目录：

`C:\Users\31379\Desktop\UAV_方案ABC切换`

其中包含：

- `一键切换_方案A_复杂场景500米_板1板2.bat`
- `一键切换_方案B_白天高层500米_板3到板9.bat`
- `一键切换_方案C_夜间无光全场景检测_板1到板9.bat`
- `打开接收端_640x480.bat`
- `README_方案ABC切换.md`
