# Clean State Checklist

## Session Closeout

- [ ] 已运行当前任务要求的验证命令。
- [ ] `feature_list.json` 已更新状态、验证和证据。
- [ ] `claude-progress.md` 已更新当前 active task 与最近完成项。
- [ ] 未执行的验证项已明确写出原因，而不是默默跳过。
- [ ] 新增脚本与入口文件的职责边界清晰，没有把 `init.*`、`dev_up.*`、`qa_full.ps1` 混成一个入口。
- [ ] 工作树处于可恢复状态，没有留下未说明的破坏性改动。
- [ ] 如需继续下一轮实现，下一位 agent 只读根目录入口文件即可知道从哪里开始。
