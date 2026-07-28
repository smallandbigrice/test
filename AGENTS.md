# Codex Local Rules

## Encoding

- Treat all Markdown, PowerShell, Python, JSON, CSV, and generated text files as UTF-8.
- In PowerShell, read and write Chinese text with explicit encoding when possible: `Get-Content -Encoding UTF8`, `Set-Content -Encoding UTF8`, `Out-File -Encoding UTF8`.
- In Python, always pass `encoding="utf-8"` for text files.
- Do not repair mojibake by guessing inside the final document. Re-open the source with the correct encoding, then regenerate the output.
- Keep Git configured with `core.quotepath=false` so Chinese file paths are readable in status and diffs.

## Markdown Change Control

- Do not create or modify Markdown files unless the user explicitly allows Markdown changes for the current task.
- When Markdown changes are allowed, keep them narrowly scoped to the requested record or handoff.

## Remote And Long-Running Commands

- Do not use unbounded board/network commands from Codex.
- For board checks, prefer `ping.exe -n 1 -w 500 <ip>` instead of `Test-Connection` loops.
- For SSH/SCP, add short connection guards where practical: `-o ConnectTimeout=5 -o ServerAliveInterval=5 -o ServerAliveCountMax=1`.
- Keep `timeout_ms` tight for probing commands, normally 10-30 seconds. Use longer timeouts only for known long jobs such as training or video conversion.
- Avoid shell syntax from another shell. In PowerShell, do not use Bash heredocs like `python - <<'PY'`; use PowerShell here-strings piped to Python instead.
- When a command may hang, probe one board first, then fan out after the path is confirmed.
