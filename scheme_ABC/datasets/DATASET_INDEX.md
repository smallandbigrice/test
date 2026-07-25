# 数据集索引

生成日期：2026-07-21

本索引记录已经实际迁移到 `E:\detect uav\scheme_ABC\datasets` 下的数据集。旧路径中的核心数据集已经移动，不再只做引用索引。

## 方案 A：复杂场景500米

用途：复杂背景下的远距无人机 ROI 训练、板端回放和模型对比。

| 名称 | 新路径 | 用途 |
|---|---|---|
| run_014 | `E:\detect uav\scheme_ABC\datasets\A_complex_scene_500m\run_014` | 复杂背景实测数据 |
| run_014 包 | `E:\detect uav\scheme_ABC\datasets\A_complex_scene_500m\run_014.zip` | 云端训练归档包 |
| run_014_roi640 | `E:\detect uav\scheme_ABC\datasets\A_complex_scene_500m\run_014_roi640` | ROI640 裁剪训练集 |
| run_014_roi640 包 | `E:\detect uav\scheme_ABC\datasets\A_complex_scene_500m\run_014_roi640_20260719.tar.gz` | 云端训练包 |
| run_015 包 | `E:\detect uav\scheme_ABC\datasets\A_complex_scene_500m\run_015.zip` | 复杂背景实测增量 |
| run_017 | `E:\detect uav\scheme_ABC\datasets\A_complex_scene_500m\run_017` | 7月20日训练增量 |
| run_017 包 | `E:\detect uav\scheme_ABC\datasets\A_complex_scene_500m\run_017.zip` | 训练归档包 |
| run_017_roi640 | `E:\detect uav\scheme_ABC\datasets\A_complex_scene_500m\run_017_roi640` | ROI640 裁剪训练集 |
| run_017_roi640 包 | `E:\detect uav\scheme_ABC\datasets\A_complex_scene_500m\run_017_roi640_20260720.tar.gz` | 云端训练包 |

## 方案 B：白天高层500米

用途：白天高层远距离天空小目标检测。

| 名称 | 新路径 | 用途 |
|---|---|---|
| datasets1_fusion_gray8_8_2 | `E:\detect uav\scheme_ABC\datasets\B_day_high500\datasets1_fusion_gray8_8_2` | 白天/灰度基础合集 |
| datasets1_fusion_gray8_8_2 包 | `E:\detect uav\scheme_ABC\datasets\B_day_high500\datasets1_fusion_gray8_8_2.zip` | 基础合集压缩包 |
| datasets_gray_base26_plus_0701 | `E:\detect uav\scheme_ABC\datasets\B_day_high500\datasets_gray_base26_plus_0701_8_2_20260704` | 白天基准扩展 |
| datasets_gray_base26_plus_0701 包 | `E:\detect uav\scheme_ABC\datasets\B_day_high500\datasets_gray_base26_plus_0701_8_2_20260704.zip` | 白天扩展压缩包 |
| all | `E:\detect uav\scheme_ABC\datasets\B_day_high500\all` | 本地完整融合归档 |
| all 包 | `E:\detect uav\scheme_ABC\datasets\B_day_high500\all.zip` | 本地完整归档包 |
| 来源组件 | `E:\detect uav\scheme_ABC\datasets\B_day_high500\source_components` | 原 `E:\detect uav\数据集` 下的组成数据 |

## 方案 C：夜间无光全场景检测

用途：夜间高层弱小目标检测，低分辨率输入加 160 ROI 放大。

| 名称 | 新路径 | 用途 |
|---|---|---|
| 8night | `E:\detect uav\scheme_ABC\datasets\C_night400\8night` | 夜间原始整理数据 |
| 8night_roi640 | `E:\detect uav\scheme_ABC\datasets\C_night400\8night_roi640` | 夜间 ROI640 |
| night8_roi640_gray_for_cloud | `E:\detect uav\scheme_ABC\datasets\C_night400\night8_roi640_gray_for_cloud.zip` | 夜间灰度 ROI640 |
| video9 manual plus 0701 | `E:\detect uav\scheme_ABC\datasets\C_night400\datasets1_fusion_gray8_8_2_before_video9_manual_20260626_165034_plus_datasets_20260701` | video9 增量融合 |
| video9 manual plus 0701 包 | `E:\detect uav\scheme_ABC\datasets\C_night400\datasets1_fusion_gray8_8_2_before_video9_manual_20260626_165034_plus_datasets_20260701.zip` | video9 增量融合压缩包 |
| gray base plus night8/night9 | `E:\detect uav\scheme_ABC\datasets\C_night400\datasets_gray_base26_plus_0701_night8_night9_8_2_20260704` | 夜间高层训练合集 |
| gray base plus night8/night9 包 | `E:\detect uav\scheme_ABC\datasets\C_night400\datasets_gray_base26_plus_0701_night8_night9_8_2_20260704.zip` | 夜间高层训练合集压缩包 |
| video9 手工标注来源 | `E:\detect uav\scheme_ABC\datasets\C_night400\manual_video9` | video9 全帧和 ROI 标注数据 |
| video9 fixed 合集 | `E:\detect uav\scheme_ABC\datasets\C_night400\datasets1_fusion_gray8_9manual_8_2_fixed` | video9 手工标注修正合集 |
| video9 fixed 合集包 | `E:\detect uav\scheme_ABC\datasets\C_night400\datasets1_fusion_gray8_9manual_8_2_fixed.zip` | video9 手工标注修正压缩包 |
| 7月1日标签包 | `E:\detect uav\scheme_ABC\datasets\C_night400\july0701_labels_for_cloud.zip` | 夜间训练标签来源 |

## 补充来源数据

下列旧数据源已经从 `record_2k_pure` 集中迁移到：

`E:\detect uav\scheme_ABC\datasets\source_components`

其中包含 `datasets0607_2`、`datasets0607_2_fusion_clean`、`small_fusion_roi640`、`bigsmall`、`wx_2k_fusion_roi_unique`、抽帧数据、旧压缩包和合并缓存。该目录用于训练追溯，不作为单独部署方案。

## 迁移记录

- 第一批迁移日志：`E:\detect uav\scheme_ABC\migration_20260721.csv`
- 第二批迁移日志：`E:\detect uav\scheme_ABC\migration_extra_20260721.csv`
