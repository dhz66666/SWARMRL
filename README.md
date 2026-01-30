git remote -v

修改远程仓库地址： 直接把 origin 指向你的新地址：
Bash

git remote set-url origin <你的新仓库地址>


git push -u origin main --force

第一步：检查当前状态
Bash

git status

确保显示的是 Your branch is ahead of 'origin/main' by X commits。如果文件已经在“暂存区”（Staged），请执行下一步。
第二步：在命令行提交（跳过 VS Code 界面卡顿）
Bash

git commit -m "update training scripts"

第三步：使用 SSH 或加大缓冲区推送

由于你之前遇到了 HTTP 408 错误，尝试再次强制增加缓冲区并推送：
Bash

# 增加缓冲区到 500MB
git config http.postBuffer 524288000

# 执行推送
git push origin main