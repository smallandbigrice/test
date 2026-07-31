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
        "192.168.0.8",
        "192.168.0.9"
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
    [int]$ConnectTimeoutSec = 5,
    [int]$ServerAliveIntervalSec = 5,
    [int]$ServerAliveCountMax = 1,
    [switch]$DeployModels,
    [switch]$NoRestart,
    [switch]$NoClean
)

$ErrorActionPreference = "Stop"
$SshOptions = @(
    "-o", "ConnectTimeout=$ConnectTimeoutSec",
    "-o", "ServerAliveInterval=$ServerAliveIntervalSec",
    "-o", "ServerAliveCountMax=$ServerAliveCountMax"
)
$ScpOptions = $SshOptions

$BoardIps = @(
    $BoardIps |
        ForEach-Object { $_ -split "," } |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ }
)

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$DayModelLocal = Join-Path $RepoRoot "models\yolov5s_day_20260626.rknn"
$NightModelLocal = Join-Path $RepoRoot "models\yolov5s_night_latest.rknn"
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
    $H265Capture = 1
    $H265ColorCapture = 1
    $H265Devices = "by-id"
    $H265GrayPreview = 0
    $StartupStarSuppress = 0
    $AlgorithmNote = "scheme-b-day-high500-h265-color-hover-relaxed"
    $TrackSearchRois = 1
    $DrawIntermediateBoxes = 1
    $MaxDrawBoxes = 500
    $ExtraConfirmDropIn = @"
Environment="UAV_HIGH_TRACK_SEARCH_MAX_ROIS=4"
Environment="UAV_TRACK_RECHECK_MIN_GAP_FRAMES=3"
Environment="UAV_HOVER_ENTER_YOLO_HITS=2"
Environment="UAV_HOVER_ENTER_RECENT_HITS=1"
Environment="UAV_HOVER_RECHECK_INTERVAL_FRAMES=9"
Environment="UAV_HOVER_MAX_RECHECK_MISSES=16"
Environment="UAV_HOVER_MAX_YOLO_GAP_FRAMES=240"
Environment="UAV_HOVER_ROI_SIZE=640"
Environment="UAV_HOVER_MIN_TRACK_SCORE=0.28"
Environment="UAV_HOVER_MAX_SPEED_PX_PER_FRAME=6.0"
Environment="UAV_HOVER_MAX_RECENT_NET_MOTION_PX=80"
Environment="UAV_STATIC_CONFIRM=1"
Environment="UAV_STATIC_CONFIRM_YOLO_HITS=2"
Environment="UAV_STATIC_CONFIRM_RECENT_HITS=2"
Environment="UAV_STATIC_CONFIRM_MAX_MISSES=6"
Environment="UAV_STATIC_CONFIRM_KEEP_MISSES=48"
Environment="UAV_STATIC_CONFIRM_MAX_AGE_FRAMES=240"
Environment="UAV_STATIC_CONFIRM_MAX_YOLO_GAP_FRAMES=240"
Environment="UAV_STATIC_CONFIRM_MIN_SCORE=0.28"
Environment="UAV_STATIC_CONFIRM_MIN_MEAN_CONF=0.28"
Environment="UAV_STATIC_CONFIRM_MAX_SPEED_PX_PER_FRAME=6.0"
Environment="UAV_STATIC_CONFIRM_MAX_RECENT_NET_MOTION_PX=80"
Environment="UAV_STATIC_CONFIRM_MAX_CENTER_JITTER=110"
Environment="UAV_STATIC_CONFIRM_MAX_BOX_JITTER_RATIO=1.5"
Environment="UAV_TRACKER_MIN_HITS=2"
Environment="UAV_TRACKER_MIN_RECENT_HITS=1"
Environment="UAV_TRACKER_CONFIRM_SCORE=0.35"
Environment="UAV_TRACKER_MIN_TRAJ_SCORE=0.10"
Environment="UAV_YOLO_DIRECT_CONFIRM_HITS=2"
Environment="UAV_YOLO_DIRECT_CONFIRM_RECENT_HITS=1"
Environment="UAV_YOLO_DIRECT_CONFIRM_SCORE=0.30"
Environment="UAV_YOLO_DIRECT_CONFIRM_MAX_MISSES=2"
Environment="UAV_TRACK_CONFIRM_MIN_NET_MOTION_PX_HIGH=0.0"
Environment="UAV_TRACK_CONFIRM_MIN_STRAIGHTNESS_HIGH=0.0"
Environment="UAV_BG_ANCHORED_TRACK_FILTER=0"
"@
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
    $HoverHold = 1
    $YoloConf = "0.30"
    $NightStaticPointSuppress = 1
    $H265Capture = 1
    $H265ColorCapture = 0
    $H265Devices = "by-id"
    $H265GrayPreview = 1
    $StartupStarSuppress = 1
    $AlgorithmNote = "scheme-c-night-no-light-all-scene-h265-yplane"
    $TrackSearchRois = 0
    $DrawIntermediateBoxes = 0
    $MaxDrawBoxes = 80
    $ExtraConfirmDropIn = @"
Environment="UAV_HIGH_TRACK_SEARCH_MAX_ROIS=0"
Environment="UAV_TRACK_RECHECK_MIN_GAP_FRAMES=3"
Environment="UAV_HOVER_ENTER_YOLO_HITS=2"
Environment="UAV_HOVER_ENTER_RECENT_HITS=1"
Environment="UAV_HOVER_RECHECK_INTERVAL_FRAMES=9"
Environment="UAV_HOVER_MAX_RECHECK_MISSES=16"
Environment="UAV_HOVER_MAX_YOLO_GAP_FRAMES=240"
Environment="UAV_HOVER_ROI_SIZE=640"
Environment="UAV_HOVER_MIN_TRACK_SCORE=0.28"
Environment="UAV_HOVER_MAX_SPEED_PX_PER_FRAME=6.0"
Environment="UAV_HOVER_MAX_RECENT_NET_MOTION_PX=80"
Environment="UAV_STATIC_CONFIRM=1"
Environment="UAV_STATIC_CONFIRM_YOLO_HITS=2"
Environment="UAV_STATIC_CONFIRM_RECENT_HITS=2"
Environment="UAV_STATIC_CONFIRM_MAX_MISSES=6"
Environment="UAV_STATIC_CONFIRM_KEEP_MISSES=48"
Environment="UAV_STATIC_CONFIRM_MAX_AGE_FRAMES=240"
Environment="UAV_STATIC_CONFIRM_MAX_YOLO_GAP_FRAMES=240"
Environment="UAV_STATIC_CONFIRM_MIN_SCORE=0.28"
Environment="UAV_STATIC_CONFIRM_MIN_MEAN_CONF=0.28"
Environment="UAV_STATIC_CONFIRM_MAX_SPEED_PX_PER_FRAME=6.0"
Environment="UAV_STATIC_CONFIRM_MAX_RECENT_NET_MOTION_PX=80"
Environment="UAV_STATIC_CONFIRM_MAX_CENTER_JITTER=110"
Environment="UAV_STATIC_CONFIRM_MAX_BOX_JITTER_RATIO=1.5"
Environment="UAV_TRACKER_MIN_HITS=2"
Environment="UAV_TRACKER_MIN_RECENT_HITS=1"
Environment="UAV_TRACKER_CONFIRM_SCORE=0.35"
Environment="UAV_TRACKER_MIN_TRAJ_SCORE=0.10"
Environment="UAV_YOLO_DIRECT_CONFIRM_HITS=2"
Environment="UAV_YOLO_DIRECT_CONFIRM_RECENT_HITS=1"
Environment="UAV_YOLO_DIRECT_CONFIRM_SCORE=0.30"
Environment="UAV_YOLO_DIRECT_CONFIRM_MAX_MISSES=2"
Environment="UAV_TRACK_CONFIRM_MIN_NET_MOTION_PX_HIGH=0.0"
Environment="UAV_TRACK_CONFIRM_MIN_STRAIGHTNESS_HIGH=0.0"
Environment="UAV_BG_ANCHORED_TRACK_FILTER=1"
"@
}

$DropIn = @"
[Service]
Environment="UAV_ALGORITHM_NOTE=$AlgorithmNote"
Environment="UAV_VIDEO_TEST=0"
Environment="UAV_SCENE_MODE=$Mode"
Environment="UAV_LAYER_MODE=high"
Environment="UAV_VIDEO_CAM_COUNT=5"
Environment="UAV_BOARD_ROW_IDX=1"
Environment="UAV_CAM_LAYER_MODES=high,high,high,high,high"
Environment="UAV_RKNN_MODEL=$SelectedModelRemote"
Environment="UAV_YOLO_CONF=$YoloConf"
Environment="UAV_H265_YPLANE_CAPTURE=$H265Capture"
Environment="UAV_H265_COLOR_CAPTURE=$H265ColorCapture"
Environment="UAV_H265_DEVICE=/dev/video1"
Environment="UAV_H265_DEVICES=$H265Devices"
Environment="UAV_H265_FPS=30"
Environment="UAV_CAMERA_WARMUP_FRAMES=60"
Environment="UAV_H265_GRAY_PREVIEW=$H265GrayPreview"
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
Environment="UAV_TRACK_SEARCH_ROIS=$TrackSearchRois"
Environment="UAV_HOVER_HOLD=$HoverHold"
Environment="UAV_FULLFRAME_SEARCH_FALLBACK=0"
Environment="UAV_FULLFRAME_FALLBACK_ONLY=0"
Environment="UAV_EDGE_REGION_LAYER=0"
Environment="UAV_AUTO_CAM_LAYER=0"
Environment="UAV_SHOW_WINDOWS=0"
Environment="UAV_DRAW_INTERMEDIATE_BOXES=$DrawIntermediateBoxes"
Environment="UAV_MAX_DRAW_BOXES=$MaxDrawBoxes"
Environment="UAV_NIGHT_STATIC_POINT_SUPPRESS=$NightStaticPointSuppress"
Environment="UAV_NIGHT_STARTUP_STAR_SUPPRESS=$StartupStarSuppress"
Environment="UAV_NIGHT_STARTUP_STAR_SECONDS=60"
Environment="UAV_NIGHT_STARTUP_STAR_PAD=6"
Environment="UAV_VIDEO_TARGET_IP=$VideoTargetIp"
Environment="UAV_VIDEO_BASE_PORT=$VideoPort"
Environment="UAV_VIDEO_STREAM_CAMS="
Environment="UAV_VIDEO_STREAM_W=$VideoWidth"
Environment="UAV_VIDEO_STREAM_H=$VideoHeight"
Environment="UAV_VIDEO_SEND_EVERY_N_FRAMES=$VideoSendEvery"
Environment="UAV_VIDEO_JPEG_QUALITY=$VideoJpegQuality"
Environment="UAV_VIDEO_SNDBUF_BYTES=262144"
Environment="UAV_RKNN_WORKERS=3"
Environment="UAV_NPU_WORKERS=3"
Environment="UAV_QUEUE_PER_CAM=6"
Environment="UAV_TASK_MAX_AGE_SEC=0.40"
Environment="UAV_NPU_TASK_MAX_AGE_SEC=0.40"
Environment="UAV_RESULT_MAX_AGE_FRAMES=12"
Environment="UAV_REPLACE_PENDING_TASKS=1"
Environment="UAV_REPLACE_PENDING=1"
$ExtraConfirmDropIn
"@

$EncodedDropIn = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($DropIn))
$DropInPath = "/etc/systemd/system/$Service.d/zzzzz-uav-scene-mode.conf"
$ReloadCommand = @"
service="$Service"
systemctl daemon-reload
need_reload="`$(systemctl show "`$service" -p NeedDaemonReload --value 2>/dev/null || true)"
if [ "`$need_reload" = "yes" ]; then
    touch "/etc/systemd/system/`$service" 2>/dev/null || true
    touch /etc/systemd/system/"`$service".d/*.conf 2>/dev/null || true
    systemctl daemon-reload
fi
"@
$ReloadCommand = $ReloadCommand -replace "`r`n", "`n"
$ReloadEncoded = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($ReloadCommand))
$CleanupCommand = @"
dir="/etc/systemd/system/$Service.d"
mkdir -p "`$dir"
find "`$dir" -maxdepth 1 -type f \( -name '10-uav-ui.conf' -o -name '20-uav-video-receiver.conf' -o -name '30-*.conf' -o -name '35-uav-scene-mode.conf' -o -name '99-*.conf' -o -name 'zz*.conf' -o -name '*.conf.bak*' \) ! -name 'zzzzz-uav-scene-mode.conf' -delete
"@
$CleanupCommand = $CleanupCommand -replace "`r`n", "`n"

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
        scp @ScpOptions $SelectedModelLocal "${Remote}:$TmpRemote"
        ssh @SshOptions $Remote "mkdir -p '$RemoteModelDir' && mv '$TmpRemote' '$SelectedModelRemote' && chmod 644 '$SelectedModelRemote'"
    }

    if (-not $NoClean) {
        $CleanupEncoded = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($CleanupCommand))
        ssh @SshOptions $Remote "echo $CleanupEncoded | base64 -d | sh"
    }

    ssh @SshOptions $Remote "mkdir -p /etc/systemd/system/$Service.d && echo $EncodedDropIn | base64 -d > $DropInPath && echo $ReloadEncoded | base64 -d | sh"

    if (-not $NoRestart) {
        ssh @SshOptions $Remote "systemctl restart $Service && sleep 2"
    }

    $Status = ssh @SshOptions $Remote "echo $ReloadEncoded | base64 -d | sh; systemctl is-active $Service; systemctl show $Service -p MainPID -p NeedDaemonReload"
    Write-Host ($Status -join "`n")
    $PidLine = ($Status | Where-Object { $_ -like "MainPID=*" } | Select-Object -First 1)
    $MainPid = ($PidLine -split "=")[1].Trim()
    if ($MainPid -and $MainPid -ne "0") {
        ssh @SshOptions $Remote "tr '\0' '\n' < /proc/$MainPid/environ | grep -E 'UAV_(BOARD_ID|ALGORITHM_NOTE|SCENE_MODE|LAYER_MODE|RKNN_MODEL|H265|CAPTURE|DIFF|PROCESS|MOTION_ZOOM|HOVER|NIGHT_STATIC|NIGHT_STARTUP_STAR|VIDEO_CAM_COUNT|VIDEO_STREAM|VIDEO_SEND|VIDEO_JPEG|SHOW_WINDOWS|DRAW|YOLO_CONF)' | sort"
    }
}
