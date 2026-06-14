param(
    [string]$Source = ".\main.py",
    [string]$User = "Tronlong",
    [string]$BasePrefix = "12.168.0",
    [int[]]$Boards = @(1, 2, 3, 4, 5, 6, 7, 8, 9),
    [string]$Dest = "/home/Tronlong/rknn_model_zoo/examples/yolov5/python/main.py",
    [string]$RestartCommand = "",
    [switch]$NoBackup,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Source)) {
    throw "Source file not found: $Source"
}

$scp = Get-Command scp -ErrorAction SilentlyContinue
$ssh = Get-Command ssh -ErrorAction SilentlyContinue
if (-not $scp -or -not $ssh) {
    throw "OpenSSH scp/ssh not found. Install Windows OpenSSH Client or run this script from a machine with ssh/scp."
}

$sourceFull = (Resolve-Path -LiteralPath $Source).Path
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

function Invoke-LoggedCommand {
    param([string]$FilePath, [string[]]$ArgsForCommand)
    Write-Host ("$FilePath " + ($ArgsForCommand -join " "))
    if ($DryRun) {
        return
    }
    & $FilePath @ArgsForCommand
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath"
    }
}

function Invoke-LoggedSshScript {
    param([string]$Remote, [string]$ScriptText)
    Write-Host "$($ssh.Source) $Remote sh -s < remote update script"
    if ($DryRun) {
        Write-Host $ScriptText
        return
    }
    $ScriptText | & $ssh.Source $Remote "sh -s"
    if ($LASTEXITCODE -ne 0) {
        throw "Remote update script failed with exit code ${LASTEXITCODE}: $Remote"
    }
}

$remoteScriptTemplate = @'
set -eu
dest='__DEST__'
tmp='__TMP__'
restart_cmd='__RESTART__'
backup_old='__BACKUP__'

get_python_assignment() {
    key="$1"
    file="$2"
    awk -v key="$key" '
        BEGIN { pattern = "^[[:space:]]*" key "[[:space:]]*=" }
        $0 ~ pattern {
            sub(/^[^=]*=[[:space:]]*/, "", $0)
            sub(/[[:space:]]*#.*/, "", $0)
            print
            exit
        }
    ' "$file"
}

set_python_assignment() {
    key="$1"
    value="$2"
    file="$3"
    tmp_assign="$file.assign.$$"
    awk -v key="$key" -v value="$value" '
        BEGIN { pattern = "^[[:space:]]*" key "[[:space:]]*=" }
        $0 ~ pattern {
            match($0, /^[[:space:]]*/)
            indent = substr($0, 1, RLENGTH)
            print indent key " = " value
            next
        }
        { print }
    ' "$file" > "$tmp_assign"
    mv "$tmp_assign" "$file"
}

dest_dir=$(dirname -- "$dest")
if [ ! -d "$dest_dir" ]; then
    echo "destination directory does not exist: $dest_dir" >&2
    exit 1
fi

if [ -f "$dest" ]; then
    old_board_id=$(get_python_assignment BOARD_ID "$dest" || true)
    old_row_idx=$(get_python_assignment BOARD_ROW_IDX "$dest" || true)
    if [ -n "$old_board_id" ]; then
        set_python_assignment BOARD_ID "$old_board_id" "$tmp"
        echo "preserved BOARD_ID = $old_board_id"
    fi
    if [ -n "$old_row_idx" ]; then
        set_python_assignment BOARD_ROW_IDX "$old_row_idx" "$tmp"
        echo "preserved BOARD_ROW_IDX = $old_row_idx"
    fi
    if cmp -s "$tmp" "$dest"; then
        rm -f "$tmp"
        echo "main.py already up to date"
        exit 0
    fi
    if [ "$backup_old" = "1" ]; then
        backup="$dest.bak.$(date '+%Y%m%d_%H%M%S')"
        cp "$dest" "$backup"
        echo "backup created: $backup"
    fi
    mode=$(stat -c '%a' "$dest" 2>/dev/null || printf '644')
    chmod "$mode" "$tmp" 2>/dev/null || chmod 644 "$tmp"
else
    chmod 644 "$tmp"
fi

mv "$tmp" "$dest"
sync
echo "updated $dest"

if [ -n "$restart_cmd" ]; then
    echo "running restart command: $restart_cmd"
    sh -c "$restart_cmd"
fi
'@

foreach ($board in $Boards) {
    $hostIp = "$BasePrefix.$board"
    $remote = "$User@$hostIp"
    $remoteTmp = "/tmp/main.py.codex_update.$stamp.$PID"
    Write-Host ""
    Write-Host "=== Updating $remote ==="

    Invoke-LoggedCommand -FilePath $scp.Source -ArgsForCommand @($sourceFull, "${remote}:$remoteTmp")

    $backupFlag = if ($NoBackup) { "0" } else { "1" }
    $remoteScript = $remoteScriptTemplate.
        Replace("__DEST__", $Dest.Replace("'", "'\''")).
        Replace("__TMP__", $remoteTmp.Replace("'", "'\''")).
        Replace("__RESTART__", $RestartCommand.Replace("'", "'\''")).
        Replace("__BACKUP__", $backupFlag)

    Invoke-LoggedSshScript -Remote $remote -ScriptText $remoteScript
}

Write-Host ""
Write-Host "Done."
