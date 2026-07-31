param(
    [string[]]$BoardIps = @("192.168.0.8"),
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
    [switch]$DeployModel,
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
$ComplexModelLocal = Join-Path $RepoRoot "models\yolov5s_dataset0727_roi640_20260728_fp16.rknn"
$ComplexModelRemote = "$RemoteModelDir/yolov5s_dataset0727_roi640_20260728_fp16.rknn"

$DropIn = @"
[Service]
Environment="UAV_SCHEME_NAME=A_complex_scene_500m"
Environment="UAV_ALGORITHM_NOTE=scheme-a-complex-field-relaxed-red-green-20260731"
Environment="UAV_VIDEO_CAM_COUNT=1"
Environment="UAV_SCENE_MODE=day"
Environment="UAV_LAYER_MODE=low"
Environment="UAV_AUTO_CAM_LAYER=0"
Environment="UAV_BOARD_ROW_IDX=0"
Environment="UAV_CAM_LAYER_MODES=low"
Environment="UAV_EDGE_REGION_LAYER=0"
Environment="UAV_EDGE_REGION_SUPPRESS_LOW=0"
Environment="UAV_EDGE_REGION_FILL_BELOW=0"
Environment="UAV_EDGE_REGION_DRAW_MASK=0"
Environment="UAV_H265_YPLANE_CAPTURE=1"
Environment="UAV_H265_COLOR_CAPTURE=1"
Environment="UAV_H265_DEVICE=/dev/video1"
Environment="UAV_H265_DEVICES=/dev/video1"
Environment="UAV_H265_CAM_INDEX=0"
Environment="UAV_H265_FPS=30"
Environment="UAV_H265_GRAY_PREVIEW=0"
Environment="UAV_CAMERA_WARMUP_FRAMES=90"
Environment="UAV_CAPTURE_W=2560"
Environment="UAV_CAPTURE_H=1440"
Environment="UAV_DIFF_W=1920"
Environment="UAV_DIFF_H=1080"
Environment="UAV_PROCESS_EVERY_N_FRAMES=3"
Environment="UAV_STATIC_BG_MASK=0"
Environment="UAV_STATIC_BG_ONLY_ROIS=0"
Environment="UAV_RKNN_MODEL=$ComplexModelRemote"
Environment="UAV_YOLO_CONF=0.40"
Environment="UAV_LOW_DIFF_THRESH=6"
Environment="UAV_LOW_MIN_LOCAL_DIFF_MEAN=4"
Environment="UAV_LOW_MIN_DIFF_AREA=3"
Environment="UAV_LOW_MOTION_ERODE_ITER=0"
Environment="UAV_LOW_MOTION_DILATE_ITER=1"
Environment="UAV_LOW_MOTION_CLOSE_ITER=0"
Environment="UAV_LOW_LCM_FILTER=1"
Environment="UAV_LOW_MEDIAN_BG_GATE=0"
Environment="UAV_LOW_BG_DENSE_SUPPRESS=0"
Environment="UAV_LOW_STATIC_BG_SEED_ROIS=0"
Environment="UAV_LOW_COMBINED_DIFF_BG_ROIS=0"
Environment="UAV_LOW_WEAK_TARGET_SEED_ROIS=0"
Environment="UAV_LOW_MAX_DET_BOX_W=220"
Environment="UAV_LOW_MAX_DET_BOX_H=180"
Environment="UAV_LOW_MAX_DET_BOX_AREA=32000"
Environment="UAV_GRU_GRAY_SEED_ROIS=0"
Environment="UAV_TIGHT_MOTION_ROI=0"
Environment="UAV_MOTION_ROI_STABILITY_GATE=1"
Environment="UAV_MOTION_ROI_STABILITY_HITS=2"
Environment="UAV_MOTION_ROI_STABILITY_MAX_AGE_FRAMES=12"
Environment="UAV_MOTION_ROI_STABILITY_CENTER_DIST=180"
Environment="UAV_MOTION_ROI_STABILITY_KEEP_TOP=1"
Environment="UAV_ROI_GRID_MAX_PER_CELL=1"
Environment="UAV_MAX_ROIS_PER_FRAME=2"
Environment="UAV_MAX_ROIS_PER_UPDATE=2"
Environment="UAV_ROI_OVERLAP_SUPPRESS=1"
Environment="UAV_ROI_OVERLAP_IOU_THRESH=0.18"
Environment="UAV_ROI_OVERLAP_COVER_THRESH=0.45"
Environment="UAV_DIRECT_FULL_FRAME_INFERENCE=0"
Environment="UAV_MOTION_ZOOM_CROP_SIZE=0"
Environment="UAV_MOTION_ZOOM_MAX_ROIS=0"
Environment="UAV_MOTION_ZOOM_ONLY=0"
Environment="UAV_TRACK_ZOOM_CROP_SIZE=0"
Environment="UAV_TRACK_SEARCH_ROIS=0"
Environment="UAV_TRACK_REQUIRE_CURRENT_MOTION=0"
Environment="UAV_TRACK_SEARCH_CONFIRMED_ONLY=1"
Environment="UAV_TRACK_SEARCH_MAX_AGE_FRAMES=24"
Environment="UAV_TRACK_SEARCH_PREDICT_MAX_MISSES=8"
Environment="UAV_TRACK_LOW_CONF_RECHECK=1"
Environment="UAV_TRACK_RECHECK_CONF=0.30"
Environment="UAV_TRACK_RECHECK_MAX_CENTER_DIST=120"
Environment="UAV_TRACKER_MIN_HITS=3"
Environment="UAV_TRACKER_MIN_RECENT_HITS=2"
Environment="UAV_TRACKER_CONFIRM_SCORE=0.40"
Environment="UAV_TRACKER_MIN_TRAJ_SCORE=0.35"
Environment="UAV_TRACK_CONFIRM_MIN_NET_MOTION_PX_LOW=4"
Environment="UAV_TRACK_CONFIRM_MIN_STRAIGHTNESS_LOW=0.18"
Environment="UAV_HOVER_HOLD=1"
Environment="UAV_HOVER_ENTER_YOLO_HITS=3"
Environment="UAV_HOVER_ENTER_RECENT_HITS=2"
Environment="UAV_HOVER_RECHECK_INTERVAL_FRAMES=12"
Environment="UAV_HOVER_MAX_RECHECK_MISSES=12"
Environment="UAV_HOVER_MAX_YOLO_GAP_FRAMES=90"
Environment="UAV_HOVER_MIN_TRACK_SCORE=0.30"
Environment="UAV_FULLFRAME_SEARCH_FALLBACK=0"
Environment="UAV_FULLFRAME_FALLBACK_ONLY=0"
Environment="UAV_NIGHT_STATIC_POINT_SUPPRESS=0"
Environment="UAV_DEBUG_ROI_STATS=0"
Environment="UAV_VIDEO_TARGET_IP=$VideoTargetIp"
Environment="UAV_VIDEO_BASE_PORT=$VideoPort"
Environment="UAV_VIDEO_STREAM_CAMS=0"
Environment="UAV_VIDEO_STREAM_W=$VideoWidth"
Environment="UAV_VIDEO_STREAM_H=$VideoHeight"
Environment="UAV_VIDEO_SEND_EVERY_N_FRAMES=$VideoSendEvery"
Environment="UAV_VIDEO_JPEG_QUALITY=$VideoJpegQuality"
Environment="UAV_VIDEO_SNDBUF_BYTES=262144"
Environment="UAV_DRAW_INTERMEDIATE_BOXES=1"
Environment="UAV_MAX_DRAW_BOXES=60"
Environment="UAV_SHOW_WINDOWS=0"
Environment="DISPLAY=:0"
Environment="XAUTHORITY=/home/Tronlong/.Xauthority"
"@

$EncodedDropIn = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($DropIn))
$DropInPath = "/etc/systemd/system/$Service.d/zzzzz-uav-scheme-a.conf"
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
find "`$dir" -maxdepth 1 -type f \( -name '10-uav-ui.conf' -o -name '20-uav-video-receiver.conf' -o -name '30-*.conf' -o -name '35-uav-scene-mode.conf' -o -name '99-*.conf' -o -name 'zz*.conf' -o -name '*.conf.bak*' \) ! -name 'zzzzz-uav-scheme-a.conf' -delete
"@
$CleanupCommand = $CleanupCommand -replace "`r`n", "`n"

foreach ($BoardIp in $BoardIps) {
    $Remote = "$User@$BoardIp"
    Write-Host "=== $BoardIp -> scheme A complex-scene 500m ==="

    if ($DeployModel) {
        if (-not (Test-Path -LiteralPath $ComplexModelLocal)) {
            throw "Local complex model not found: $ComplexModelLocal"
        }
        $ComplexModelName = Split-Path -Leaf $ComplexModelRemote
        $ComplexTmpRemote = "/tmp/$ComplexModelName"
        scp @ScpOptions $ComplexModelLocal "${Remote}:$ComplexTmpRemote"
        ssh @SshOptions $Remote "mkdir -p '$RemoteModelDir' && mv '$ComplexTmpRemote' '$ComplexModelRemote' && chmod 644 '$ComplexModelRemote'"
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
        ssh @SshOptions $Remote "tr '\0' '\n' < /proc/$MainPid/environ | grep -E 'UAV_(SCHEME_NAME|ALGORITHM_NOTE|SCENE_MODE|LAYER_MODE|H265|CAPTURE|DIFF|PROCESS|RKNN_MODEL|YOLO_CONF|LOW_|MOTION_ROI_STABILITY|MAX_ROIS|DRAW|TRACKER|HOVER|VIDEO_STREAM|VIDEO_SEND|VIDEO_JPEG)' | sort"
    }
}
