param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("day", "night")]
    [string]$Mode,

    [string[]]$BoardIps = @(
        "192.168.0.3",
        "192.168.0.4",
        "192.168.0.5",
        "192.168.0.6",
        "192.168.0.7",
        "192.168.0.8"
    ),

    [string]$User = "root",
    [string]$Service = "python_autostar.service",
    [string]$RemoteModelDir = "/home/Tronlong/rknn_model_zoo/examples/yolov5/model",
    [string]$VideoTargetIp = "192.168.0.200",
    [int]$VideoPort = 9999,
    [int]$VideoWidth = 640,
    [int]$VideoHeight = 480,
    [int]$VideoSendEvery = 2,
    [int]$VideoJpegQuality = 25,
    [switch]$DeployModels,
    [switch]$NoRestart,
    [switch]$NoClean
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$DayModelLocal = Join-Path $RepoRoot "model\yolov5s_day_20260626.rknn"
$NightModelLocal = Join-Path $RepoRoot "model\yolov5s_night_latest.rknn"
$DayModelRemote = "$RemoteModelDir/yolov5s_day_20260626.rknn"
$NightModelRemote = "$RemoteModelDir/yolov5s_night_latest.rknn"

if ($Mode -eq "day") {
    $SelectedModelLocal = $DayModelLocal
    $SelectedModelRemote = $DayModelRemote
    $CaptureW = 2560
    $CaptureH = 1440
    $DiffW = 1920
    $DiffH = 1080
    $ProcessEvery = 3
    $MotionZoom = 0
    $MotionZoomMax = 0
    $MotionZoomOnly = 0
    $HoverHold = 1
    $YoloConf = "0.40"
    $NightStaticPointSuppress = 0
} else {
    $SelectedModelLocal = $NightModelLocal
    $SelectedModelRemote = $NightModelRemote
    $CaptureW = 640
    $CaptureH = 480
    $DiffW = 640
    $DiffH = 480
    $ProcessEvery = 2
    $MotionZoom = 160
    $MotionZoomMax = 1
    $MotionZoomOnly = 1
    $HoverHold = 0
    $YoloConf = "0.30"
    $NightStaticPointSuppress = 1
}

$DropIn = @"
[Service]
Environment="UAV_SCENE_MODE=$Mode"
Environment="UAV_LAYER_MODE=high"
Environment="UAV_RKNN_MODEL=$SelectedModelRemote"
Environment="UAV_YOLO_CONF=$YoloConf"
Environment="UAV_CAPTURE_W=$CaptureW"
Environment="UAV_CAPTURE_H=$CaptureH"
Environment="UAV_DIFF_W=$DiffW"
Environment="UAV_DIFF_H=$DiffH"
Environment="UAV_PROCESS_EVERY_N_FRAMES=$ProcessEvery"
Environment="UAV_DIRECT_FULL_FRAME_INFERENCE=0"
Environment="UAV_MOTION_ZOOM_CROP_SIZE=$MotionZoom"
Environment="UAV_MOTION_ZOOM_MAX_ROIS=$MotionZoomMax"
Environment="UAV_MOTION_ZOOM_ONLY=$MotionZoomOnly"
Environment="UAV_TRACK_ZOOM_CROP_SIZE=0"
Environment="UAV_TRACK_SEARCH_ROIS=0"
Environment="UAV_HOVER_HOLD=$HoverHold"
Environment="UAV_FULLFRAME_SEARCH_FALLBACK=0"
Environment="UAV_FULLFRAME_FALLBACK_ONLY=0"
Environment="UAV_EDGE_REGION_LAYER=0"
Environment="UAV_AUTO_CAM_LAYER=0"
Environment="UAV_SHOW_WINDOWS=0"
Environment="UAV_DRAW_INTERMEDIATE_BOXES=0"
Environment="UAV_NIGHT_STATIC_POINT_SUPPRESS=$NightStaticPointSuppress"
Environment="UAV_VIDEO_TARGET_IP=$VideoTargetIp"
Environment="UAV_VIDEO_BASE_PORT=$VideoPort"
Environment="UAV_VIDEO_STREAM_CAMS="
Environment="UAV_VIDEO_STREAM_W=$VideoWidth"
Environment="UAV_VIDEO_STREAM_H=$VideoHeight"
Environment="UAV_VIDEO_SEND_EVERY_N_FRAMES=$VideoSendEvery"
Environment="UAV_VIDEO_JPEG_QUALITY=$VideoJpegQuality"
"@

$EncodedDropIn = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($DropIn))
$DropInPath = "/etc/systemd/system/$Service.d/zzzzz-uav-scene-mode.conf"
$CleanupCommand = @"
dir="/etc/systemd/system/$Service.d"
mkdir -p "`$dir"
find "`$dir" -maxdepth 1 -type f \( -name '10-uav-ui.conf' -o -name '20-uav-video-receiver.conf' -o -name '30-*.conf' -o -name '35-uav-scene-mode.conf' -o -name '99-*.conf' -o -name 'zz*.conf' \) ! -name 'zzzzz-uav-scene-mode.conf' -delete
"@

foreach ($BoardIp in $BoardIps) {
    $Remote = "$User@$BoardIp"
    Write-Host "=== $BoardIp -> $Mode high-layer profile ==="

    if ($DeployModels) {
        if (-not (Test-Path -LiteralPath $SelectedModelLocal)) {
            throw "Local model not found: $SelectedModelLocal"
        }
        $ModelName = Split-Path -Leaf $SelectedModelRemote
        $TmpRemote = "/tmp/$ModelName"
        Write-Host "Uploading model: $SelectedModelLocal"
        scp $SelectedModelLocal "${Remote}:$TmpRemote"
        ssh $Remote "mkdir -p '$RemoteModelDir' && mv '$TmpRemote' '$SelectedModelRemote' && chmod 644 '$SelectedModelRemote'"
    }

    if (-not $NoClean) {
        $CleanupEncoded = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($CleanupCommand))
        ssh $Remote "echo $CleanupEncoded | base64 -d | sh"
    }

    ssh $Remote "mkdir -p /etc/systemd/system/$Service.d && echo $EncodedDropIn | base64 -d > $DropInPath && systemctl daemon-reload"

    if (-not $NoRestart) {
        ssh $Remote "systemctl restart $Service && sleep 2"
    }

    $Status = ssh $Remote "systemctl is-active $Service; systemctl show $Service -p MainPID"
    Write-Host ($Status -join "`n")
    $PidLine = ($Status | Where-Object { $_ -like "MainPID=*" } | Select-Object -First 1)
    $MainPid = ($PidLine -split "=")[1].Trim()
    if ($MainPid -and $MainPid -ne "0") {
        ssh $Remote "tr '\0' '\n' < /proc/$MainPid/environ | grep -E 'UAV_(BOARD_ID|SCENE_MODE|LAYER_MODE|RKNN_MODEL|CAPTURE|DIFF|PROCESS|MOTION_ZOOM|HOVER|VIDEO_STREAM|VIDEO_SEND|VIDEO_JPEG|SHOW_WINDOWS|DRAW|YOLO_CONF)' | sort"
    }
}
