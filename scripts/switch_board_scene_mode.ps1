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
    [switch]$SkipModelDeploy,
    [switch]$NoRestart
)

$ErrorActionPreference = "Stop"

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
Environment="UAV_SCENE_MODE=day"
Environment="UAV_RKNN_MODEL=$selectedModelRemote"
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
"@
} else {
    $selectedModelLocal = $NightModelLocal
    $selectedModelRemote = $nightModelRemote
    $dropIn = @"
[Service]
Environment="UAV_SCENE_MODE=night"
Environment="UAV_RKNN_MODEL=$selectedModelRemote"
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
"@
}

$bytes = [System.Text.Encoding]::UTF8.GetBytes($dropIn)
$encoded = [Convert]::ToBase64String($bytes)
$remote = "$User@$BoardIp"
$dropInPath = "/etc/systemd/system/$Service.d/35-uav-scene-mode.conf"

if (-not $SkipModelDeploy) {
    if (Test-Path -LiteralPath $selectedModelLocal) {
        $modelName = Split-Path -Leaf $selectedModelRemote
        $tmpRemote = "/tmp/$modelName"
        Write-Host "Uploading $Mode model: $selectedModelLocal -> ${remote}:$selectedModelRemote"
        scp $selectedModelLocal "${remote}:$tmpRemote"
        ssh $remote "mkdir -p '$RemoteModelDir' && mv '$tmpRemote' '$selectedModelRemote' && chmod 644 '$selectedModelRemote'"
    } else {
        Write-Warning "Local $Mode model not found: $selectedModelLocal. The drop-in will still point to $selectedModelRemote."
    }
}

ssh $remote "mkdir -p /etc/systemd/system/$Service.d && echo $encoded | base64 -d > $dropInPath && systemctl daemon-reload"

if (-not $NoRestart) {
    ssh $remote "systemctl restart $Service && sleep 2 && systemctl --no-pager --lines=20 status $Service"
}

Write-Host "Scene mode '$Mode' applied on $BoardIp via $dropInPath"
Write-Host "RKNN model: $selectedModelRemote"
