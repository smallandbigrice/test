# 2026-07-21 测试视频清理说明

本次清理目标：只保留原始输入视频、板端原始录制视频和数据集源视频，删除代码推理产生的测试视频。

## 已删除

- 删除数量：531 个视频文件
- 释放空间：约 12.37GB
- 删除内容：推理回放、检测叠加视频、拼接视频、对比视频、调参测试视频、屏幕录制测试视频
- 失败数量：0

删除日志：

`E:\detect uav\scheme_ABC\cleanup_logs\deleted_generated_videos_20260721.csv`

## 已保留

- 保留数量：195 个视频文件
- 保留空间：约 4.86GB
- 保留内容：原始输入视频、板端原始录制视频、数据集源视频

保留日志：

`E:\detect uav\scheme_ABC\cleanup_logs\kept_original_videos_20260721.csv`

## 原始视频入口

- 方案原始验证视频：`E:\detect uav\scheme_ABC\videos\original`
- 板端原始录制视频：`E:\detect uav\scheme_ABC\videos\board_records`
- 数据集源视频：`E:\detect uav\scheme_ABC\datasets\A_complex_scene_500m`
