# 2026-07-21 数据集和视频迁移说明

本次整理已经把方案 A/B/C 相关的数据集和视频实际移动到 `E:\detect uav\scheme_ABC` 下，不再只是写索引。

## 迁移结果

- 第一批核心迁移：31 项，日志为 `E:\detect uav\scheme_ABC\migration_20260721.csv`。
- 第二批补充迁移：32 项，日志为 `E:\detect uav\scheme_ABC\migration_extra_20260721.csv`。
- 两批失败记录均为 0 项。

## 数据集入口

- 方案 A：`E:\detect uav\scheme_ABC\datasets\A_complex_scene_500m`
- 方案 B：`E:\detect uav\scheme_ABC\datasets\B_day_high500`
- 方案 C：`E:\detect uav\scheme_ABC\datasets\C_night400`
- 补充来源：`E:\detect uav\scheme_ABC\datasets\source_components`

## 视频入口

- 原始验证视频：`E:\detect uav\scheme_ABC\videos\original`
- 板端实测视频：`E:\detect uav\scheme_ABC\videos\board_records`

## 旧目录说明

旧的 `record_2k_pure`、`board_records`、`数据集` 目录中仍可能保留历史测试输出、截图、CSV 和辅助脚本。方案 A/B/C 的核心数据集与原始视频以后以 `scheme_ABC` 为准。
