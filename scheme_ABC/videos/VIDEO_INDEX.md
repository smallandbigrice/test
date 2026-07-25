# 视频索引

生成日期：2026-07-21

本索引记录已经实际迁移到 `E:\detect uav\scheme_ABC\videos` 下的视频、板端录制结果和手工轨迹。

## 原始验证视频

| 视频 | 新路径 | 主要用途 |
|---|---|---|
| 视频3 | `E:\detect uav\scheme_ABC\videos\original\video3_source.mp4` | 早期原始验证视频 |
| 视频4 | `E:\detect uav\scheme_ABC\videos\original\video4_day_high.mp4` | 方案 B 白天高层验证 |
| 视频5 | `E:\detect uav\scheme_ABC\videos\original\video5_complex_scene.mp4` | 方案 A 复杂背景验证 |
| 视频6 | `E:\detect uav\scheme_ABC\videos\original\video6_source.mp4` | 早期原始验证视频 |
| 视频7 | `E:\detect uav\scheme_ABC\videos\original\video7_source.mp4` | 早期原始验证视频 |
| 视频8 | `E:\detect uav\scheme_ABC\videos\original\video8_night.mp4` | 方案 C 夜间数据来源 |
| 视频9 | `E:\detect uav\scheme_ABC\videos\original\video9_night.mkv` | 方案 C 夜间验证 |
| 视频10 | `E:\detect uav\scheme_ABC\videos\original\video10_day_high.mkv` | 方案 B 白天高层验证 |

## 板端原始录制视频

| 视频 | 新路径 | 主要用途 |
|---|---|---|
| board9 300米原始录制 | `E:\detect uav\scheme_ABC\videos\board_records\board9_run014_live2k_20240826_013700\cam0_gray_y_to_h265_2k_20240826_013710.mp4` | 方案 A 300米人工轨迹对比 |
| board9 H265 2K原始录制 | `E:\detect uav\scheme_ABC\videos\board_records\board9_h265_2k_20240825_214140\cam0_gray_y_to_h265_2k_20240825_214146.mp4` | H265 2K 输入链路验证 |

## 手工轨迹

| 标注 | 新路径 | 对应视频 |
|---|---|---|
| 300米从55秒开始 | `E:\detect uav\scheme_ABC\videos\board_records\board9_run014_live2k_20240826_013700\manual_uav_track_300m_from55s.csv` | board9 300米回放 |
| 300米1Hz探针 | `E:\detect uav\scheme_ABC\videos\board_records\board9_run014_live2k_20240826_013700\manual_uav_track_300m_1hz_probe.csv` | board9 300米回放 |
| 93秒后轨迹点 | `E:\detect uav\scheme_ABC\videos\board_records\board9_h265_2k_20240825_214140\manual_uav_track_points_from_93s.csv` | board9 H265 2K实测 |

## 保留验证结果

| 结果 | 路径 |
|---|---|
| 300米旧模型检测 CSV | `E:\detect uav\scheme_ABC\results\board9_300m_old_vs_new\old_run014_run015_detections.csv` |
| 300米新模型检测 CSV | `E:\detect uav\scheme_ABC\results\board9_300m_old_vs_new\new_run017_epoch71_detections.csv` |
| 300米回放检测 CSV | `E:\detect uav\scheme_ABC\results\board9_300m_old_vs_new\board9_300m_replay_detections.csv` |
| 300米模型对比说明 | `E:\detect uav\scheme_ABC\results\board9_300m_old_vs_new\manual_compare_summary.txt` |

## 迁移记录

- 第一批迁移日志：`E:\detect uav\scheme_ABC\migration_20260721.csv`
- 第二批迁移日志：`E:\detect uav\scheme_ABC\migration_extra_20260721.csv`

## 视频清理记录

2026-07-21 已清理代码推理产生的回放、叠加、拼接、对比和测试输出视频，只保留原始输入视频及原始录制视频。

- 删除清单：`E:\detect uav\scheme_ABC\cleanup_logs\deleted_generated_videos_20260721.csv`
- 保留清单：`E:\detect uav\scheme_ABC\cleanup_logs\kept_original_videos_20260721.csv`
