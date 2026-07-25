param(
    [Parameter(Mandatory = $true)]
    [string]$BoardIp,

    [Parameter(Mandatory = $true)]
    [ValidateSet("day", "night")]
    [string]$Mode,

    [string]$User = "root",
    [string]$Service = "python_autostar.service",
    [string]$RemoteModelDir = "/home/Tronlong/rknn_model_zoo/examples/yolov5/model",
    [string]$DayModelLocal = "",
    [string]$NightModelLocal = "",
    [int]$ConnectTimeoutSec = 5,
    [int]$ServerAliveIntervalSec = 5,
    [int]$ServerAliveCountMax = 1,
    [switch]$DeployModel,
    [switch]$SkipModelDeploy,
    [switch]$NoRestart
)

$ErrorActionPreference = "Stop"
$SshOptions = @(
    "-o", "ConnectTimeout=$ConnectTimeoutSec",
    "-o", "ServerAliveInterval=$ServerAliveIntervalSec",
    "-o", "ServerAliveCountMax=$ServerAliveCountMax"
)
$ScpOptions = $SshOptions

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
if (-not $DayModelLocal) {
    $DayModelLocal = Join-Path $RepoRoot "model\yolov5s_day_20260626.rknn"
}
if (-not $NightModelLocal) {
    $NightModelLocal = Join-Path $RepoRoot "model\yolov5s_night_latest.rknn"
}

$dayModelRemote = "$RemoteModelDir/yolov5s_day_20260626.rknn"
$nightModelRemote = "$RemoteModelDir/yolov5s_night_latest.rknn"
$selectedModelLocal = $DayModelLocal
$selectedModelRemote = $dayModelRemote

if ($Mode -eq "day") {
    $selectedModelLocal = $DayModelLocal
    $selectedModelRemote = $dayModelRemote
    $dropIn = @"
[Service]
Environment="UAV_ALGORITHM_NOTE=scheme-b-day-high500"
Environment="UAV_SCENE_MODE=day"
Environment="UAV_LAYER_MODE=high"
Environment="UAV_VIDEO_CAM_COUNT=5"
Environment="UAV_RKNN_MODEL=$selectedModelRemote"
Environment="UAV_H265_YPLANE_CAPTURE=0"
Environment="UAV_H265_DEVICES="
Environment="UAV_CAPTURE_W=2560"
Environment="UAV_CAPTURE_H=1440"
Environment="UAV_DIFF_W=1920"
Environment="UAV_DIFF_H=1080"
Environment="UAV_PROCESS_EVERY_N_FRAMES=3"
Environment="UAV_DIRECT_FULL_FRAME_INFERENCE=0"
Environment="UAV_MOTION_ZOOM_CROP_SIZE=0"
Environment="UAV_MOTION_ZOOM_MAX_ROIS=0"
Environment="UAV_MOTION_ZOOM_ONLY=0"
Environment="UAV_TRACK_ZOOM_CROP_SIZE=0"
Environment="UAV_TRACK_SEARCH_ROIS=0"
Environment="UAV_HOVER_HOLD=1"
Environment="UAV_FULLFRAME_SEARCH_FALLBACK=0"
Environment="UAV_FULLFRAME_FALLBACK_ONLY=0"
Environment="UAV_DRAW_INTERMEDIATE_BOXES=0"
Environment="UAV_NIGHT_STARTUP_STAR_SUPPRESS=0"
"@
} else {
    $selectedModelLocal = $NightModelLocal
    $selectedModelRemote = $nightModelRemote
    $dropIn = @"
[Service]
Environment="UAV_ALGORITHM_NOTE=scheme-c-night-no-light-all-scene-h265-yplane"
Environment="UAV_SCENE_MODE=night"
Environment="UAV_LAYER_MODE=high"
Environment="UAV_VIDEO_CAM_COUNT=5"
Environment="UAV_RKNN_MODEL=$selectedModelRemote"
Environment="UAV_H265_YPLANE_CAPTURE=1"
Environment="UAV_H265_DEVICES=auto"
Environment="UAV_H265_FPS=30"
Environment="UAV_H265_GRAY_PREVIEW=1"
Environment="UAV_CAPTURE_W=640"
Environment="UAV_CAPTURE_H=480"
Environment="UAV_DIFF_W=640"
Environment="UAV_DIFF_H=480"
Environment="UAV_PROCESS_EVERY_N_FRAMES=2"
Environment="UAV_DIRECT_FULL_FRAME_INFERENCE=0"
Environment="UAV_MOTION_ZOOM_CROP_SIZE=160"
Environment="UAV_MOTION_ZOOM_MAX_ROIS=1"
Environment="UAV_MOTION_ZOOM_ONLY=1"
Environment="UAV_TRACK_ZOOM_CROP_SIZE=0"
Environment="UAV_TRACK_SEARCH_ROIS=0"
Environment="UAV_HOVER_HOLD=0"
Environment="UAV_FULLFRAME_SEARCH_FALLBACK=0"
Environment="UAV_FULLFRAME_FALLBACK_ONLY=0"
Environment="UAV_DRAW_INTERMEDIATE_BOXES=0"
Environment="UAV_NIGHT_STATIC_POINT_SUPPRESS=1"
Environment="UAV_NIGHT_STARTUP_STAR_SUPPRESS=1"
Environment="UAV_NIGHT_STARTUP_STAR_SECONDS=60"
Environment="UAV_NIGHT_STARTUP_STAR_PAD=6"
"@
}

$bytes = [System.Text.Encoding]::UTF8.GetBytes($dropIn)
$encoded = [Convert]::ToBase64String($bytes)
$remote = "$User@$BoardIp"
$dropInPath = "/etc/systemd/system/$Service.d/zzzzz-uav-scene-mode.conf"

if ($DeployModel -and -not $SkipModelDeploy) {
    if (Test-Path -LiteralPath $selectedModelLocal) {
        $modelName = Split-Path -Leaf $selectedModelRemote
        $tmpRemote = "/tmp/$modelName"
        Write-Host "Uploading $Mode model: $selectedModelLocal -> ${remote}:$selectedModelRemote"
        scp @ScpOptions $selectedModelLocal "${remote}:$tmpRemote"
        ssh @SshOptions $remote "mkdir -p '$RemoteModelDir' && mv '$tmpRemote' '$selectedModelRemote' && chmod 644 '$selectedModelRemote'"
    } else {
        Write-Warning "Local $Mode model not found: $selectedModelLocal. The drop-in will still point to $selectedModelRemote."
    }
}

ssh @SshOptions $remote "mkdir -p /etc/systemd/system/$Service.d && echo $encoded | base64 -d > $dropInPath && systemctl daemon-reload"

if (-not $NoRestart) {
    ssh @SshOptions $remote "systemctl restart $Service && sleep 2 && systemctl --no-pager --lines=20 status $Service"
}

Write-Host "Scene mode '$Mode' applied on $BoardIp via $dropInPath"
Write-Host "RKNN model: $selectedModelRemote"
