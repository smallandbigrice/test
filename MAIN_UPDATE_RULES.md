# main.py 更新规则

本文件约束 `main.py`、板端测试代码和相关文档的修改流程。

## 固定要求

1. 修改 `main.py` 时，同时在 `docs/main_updates/` 新增中文说明。
2. 说明至少包含修改目标、算法变化、参数变化、验证方法、验证结果和回退方法。
3. 视频推理阶段优先修改 `videomain.py`；只有板端视频验证达到预期后，才同步到正式 `main.py`。
4. 用户明确要求“暂不上传 GitHub”时，只保留本地修改，不提交、不推送。
5. 用户允许上传后，使用新的 `codex/` 分支保存本次迭代，确保旧版本可回溯。
6. 不覆盖或删除用户已有修改，不使用 `git reset --hard` 或 `git checkout --` 回退工作区。

## 板端验证规则

1. 当前以 RK3588 板端结果为准，不以 PC 端推理结果替代板端结论。
2. 运行板端视频测试前检查并停止自启动服务，避免两套 RKNN 同时占用 NPU：

```bash
sudo systemctl stop python_autostar.service
```

3. 测试后确认无残留 `main.py` 或 `videomain.py` 进程。
4. 先运行 5 秒冒烟测试，再运行 20 秒对比测试。
5. A/B 测试只改变一个变量，并记录 ROI 数、RKNN 推理数、延迟、原始命中和绿框命中。
6. RKNN 静态输入模型出现 dynamic range 查询警告时可忽略，但其他错误必须记录。

## 目录约束

- 数据集处理脚本放入 `tools/datasets/`。
- 推理、统计和可视化脚本放入 `tools/diagnostics/`。
- 板端部署脚本放入 `tools/deployment/`。
- 新生成的视频、CSV 和图片放入 `artifacts/` 的实验子目录。
- 历史备份放入 `archive/code_backups/`。
- 板端视频回归入口放入 `apps/video_inference/`，不再放在根目录。
- 根目录不再新增临时脚本、备份文件或推理产物。
- 完整目录规范见 `docs/PROJECT_STRUCTURE.md`。

## 本机工具环境

PC 辅助脚本如确需运行，使用：

```powershell
& 'C:\Users\31379\.conda\envs\uav-main-gpu2\python.exe' <script.py>
```

Git 使用系统已安装版本。执行提交或推送前必须先检查：

```powershell
git status --short
git diff -- main.py
```
