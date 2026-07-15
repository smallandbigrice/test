# 板端正式检测方案部署目录

当前正式方案分为三类：

| 编号 | 名称 | 适用板位 |
| --- | --- | --- |
| 方案A | 复杂场景500米方案A | 板1、板2白天 |
| 方案B | 白天500米高层方案B | 板3到板9白天 |
| 方案C | 夜间400米方案C | 板1到板9夜间 |

## 方案A归档

`scheme_A_complex500` 保存板1、板2当前复杂场景500米方案A的可部署文件：

```text
scheme_A_complex500/main.py
scheme_A_complex500/yololib.py
scheme_A_complex500/comms.py
scheme_A_complex500/zzzzz-uav-scene-mode.conf
```

部署板1、板2：

```bat
scripts\switch_board1_2_complex500_A.bat
```

## 方案B切换

部署板3到板9白天500米高层方案B：

```bat
scripts\switch_board3_9_day500.bat
```

## 方案C切换

部署板1、板2夜间400米方案C：

```bat
scripts\switch_board1_2_night400_C.bat
```

部署板3到板9夜间400米方案C：

```bat
scripts\switch_board3_9_night400.bat
```
