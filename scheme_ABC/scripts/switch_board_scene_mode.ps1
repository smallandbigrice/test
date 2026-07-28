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
    $DayModelLocal = Join-Path $RepoRoot "models\yolov5s_day_20260626.rknn"
}
if (-not $NightModelLocal) {
    $NightModelLocal = Join-Path $RepoRoot "models\yolov5s_night_latest.rknn"
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
Environment="UAV_ALGORITHM_NOTE=scheme-b-day-high500-h265-color-hover-relaxed"
Environment="UAV_VIDEO_TEST=0"
Environment="UAV_SCENE_MODE=day"
Environment="UAV_LAYER_MODE=high"
Environment="UAV_VIDEO_CAM_COUNT=5"
Environment="UAV_BOARD_ROW_IDX=1"
Environment="UAV_CAM_LAYER_MODES=high,high,high,high,high"
Environment="UAV_RKNN_MODEL=$selectedModelRemote"
Environment="UAV_YOLO_CONF=0.40"
Environment="UAV_H265_YPLANE_CAPTURE=1"
Environment="UAV_H265_COLOR_CAPTURE=1"
Environment="UAV_H265_DEVICE=/dev/video1"
Environment="UAV_H265_DEVICES=by-id"
Environment="UAV_H265_FPS=30"
Environment="UAV_CAMERA_WARMUP_FRAMES=60"
Environment="UAV_H265_GRAY_PREVIEW=0"
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
Environment="UAV_TRACK_SEARCH_ROIS=1"
Environment="UAV_HOVER_HOLD=1"
Environment="UAV_FULLFRAME_SEARCH_FALLBACK=0"
Environment="UAV_FULLFRAME_FALLBACK_ONLY=0"
Environment="UAV_EDGE_REGION_LAYER=0"
Environment="UAV_AUTO_CAM_LAYER=0"
Environment="UAV_SHOW_WINDOWS=0"
Environment="UAV_DRAW_INTERMEDIATE_BOXES=1"
Environment="UAV_MAX_DRAW_BOXES=500"
Environment="UAV_NIGHT_STATIC_POINT_SUPPRESS=0"
Environment="UAV_NIGHT_STARTUP_STAR_SUPPRESS=0"
Environment="UAV_VIDEO_TARGET_IP=192.168.0.200"
Environment="UAV_VIDEO_BASE_PORT=9999"
Environment="UAV_VIDEO_STREAM_CAMS="
Environment="UAV_VIDEO_STREAM_W=640"
Environment="UAV_VIDEO_STREAM_H=480"
Environment="UAV_VIDEO_SEND_EVERY_N_FRAMES=2"
Environment="UAV_VIDEO_JPEG_QUALITY=25"
Environment="UAV_VIDEO_SNDBUF_BYTES=262144"
Environment="UAV_RKNN_WORKERS=3"
Environment="UAV_NPU_WORKERS=3"
Environment="UAV_QUEUE_PER_CAM=6"
Environment="UAV_TASK_MAX_AGE_SEC=0.40"
Environment="UAV_NPU_TASK_MAX_AGE_SEC=0.40"
Environment="UAV_RESULT_MAX_AGE_FRAMES=12"
Environment="UAV_REPLACE_PENDING_TASKS=1"
Environment="UAV_REPLACE_PENDING=1"
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
    $selectedModelLocal = $NightModelLocal
    $selectedModelRemote = $nightModelRemote
    $dropIn = @"
[Service]
Environment="UAV_ALGORITHM_NOTE=scheme-c-night-no-light-all-scene-h265-yplane"
Environment="UAV_VIDEO_TEST=0"
Environment="UAV_SCENE_MODE=night"
Environment="UAV_LAYER_MODE=high"
Environment="UAV_VIDEO_CAM_COUNT=5"
Environment="UAV_BOARD_ROW_IDX=1"
Environment="UAV_CAM_LAYER_MODES=high,high,high,high,high"
Environment="UAV_RKNN_MODEL=$selectedModelRemote"
Environment="UAV_YOLO_CONF=0.30"
Environment="UAV_H265_YPLANE_CAPTURE=1"
Environment="UAV_H265_COLOR_CAPTURE=0"
Environment="UAV_H265_DEVICE=/dev/video1"
Environment="UAV_H265_DEVICES=by-id"
Environment="UAV_H265_FPS=30"
Environment="UAV_CAMERA_WARMUP_FRAMES=60"
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
Environment="UAV_HOVER_HOLD=1"
Environment="UAV_FULLFRAME_SEARCH_FALLBACK=0"
Environment="UAV_FULLFRAME_FALLBACK_ONLY=0"
Environment="UAV_EDGE_REGION_LAYER=0"
Environment="UAV_AUTO_CAM_LAYER=0"
Environment="UAV_SHOW_WINDOWS=0"
Environment="UAV_DRAW_INTERMEDIATE_BOXES=0"
Environment="UAV_MAX_DRAW_BOXES=80"
Environment="UAV_NIGHT_STATIC_POINT_SUPPRESS=1"
Environment="UAV_NIGHT_STARTUP_STAR_SUPPRESS=1"
Environment="UAV_NIGHT_STARTUP_STAR_SECONDS=60"
Environment="UAV_NIGHT_STARTUP_STAR_PAD=6"
Environment="UAV_VIDEO_TARGET_IP=192.168.0.200"
Environment="UAV_VIDEO_BASE_PORT=9999"
Environment="UAV_VIDEO_STREAM_CAMS="
Environment="UAV_VIDEO_STREAM_W=640"
Environment="UAV_VIDEO_STREAM_H=480"
Environment="UAV_VIDEO_SEND_EVERY_N_FRAMES=2"
Environment="UAV_VIDEO_JPEG_QUALITY=25"
Environment="UAV_VIDEO_SNDBUF_BYTES=262144"
Environment="UAV_RKNN_WORKERS=3"
Environment="UAV_NPU_WORKERS=3"
Environment="UAV_QUEUE_PER_CAM=6"
Environment="UAV_TASK_MAX_AGE_SEC=0.40"
Environment="UAV_NPU_TASK_MAX_AGE_SEC=0.40"
Environment="UAV_RESULT_MAX_AGE_FRAMES=12"
Environment="UAV_REPLACE_PENDING_TASKS=1"
Environment="UAV_REPLACE_PENDING=1"
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
