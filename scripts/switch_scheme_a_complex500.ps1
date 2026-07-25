param(
    [string[]]$BoardIps = @("192.168.0.1", "192.168.0.2"),
    [string]$User = "root",
    [string]$Service = "python_autostar.service",
    [string]$RemoteModelDir = "/home/Tronlong/rknn_model_zoo/examples/yolov5/model",
    [string]$VideoTargetIp = "192.168.0.200",
    [int]$VideoPort = 9999,
    [int]$VideoWidth = 320,
    [int]$VideoHeight = 180,
    [int]$VideoSendEvery = 2,
    [int]$VideoJpegQuality = 15,
    [switch]$DeployModel,
    [switch]$NoRestart,
    [switch]$NoClean
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$DayModelLocal = Join-Path $RepoRoot "model\yolov5s_day_20260626.rknn"
$SelectedModelRemote = "$RemoteModelDir/yolov5s_day_20260626.rknn"

$DropIn = @"
[Service]
Environment="UAV_SCHEME_NAME=A_complex_scene_500m"
Environment="UAV_LOW_LAYER_PROFILE_NAME=complex_scene_500m_scheme_a"
Environment="UAV_SCENE_MODE=day"
Environment="UAV_LAYER_MODE=auto"
Environment="UAV_AUTO_CAM_LAYER=1"
Environment="UAV_EDGE_REGION_LAYER=1"
Environment="UAV_RKNN_MODEL=$SelectedModelRemote"
Environment="UAV_CAPTURE_W=2560"
Environment="UAV_CAPTURE_H=1440"
Environment="UAV_DIFF_W=1920"
Environment="UAV_DIFF_H=1080"
Environment="UAV_PROCESS_EVERY_N_FRAMES=2"
Environment="UAV_MAX_ROIS_PER_UPDATE=6"
Environment="UAV_ROI_GRID_MAX_PER_CELL=2"
Environment="UAV_ROI_OVERLAP_SUPPRESS=0"
Environment="UAV_GRU_GRAY_SEED_ROIS=0"
Environment="UAV_YOLO_CONF=0.30"
Environment="UAV_STATIC_CONFIRM=1"
Environment="UAV_STATIC_CONFIRM_YOLO_HITS=4"
Environment="UAV_STATIC_CONFIRM_RECENT_HITS=3"
Environment="UAV_STATIC_CONFIRM_MAX_MISSES=2"
Environment="UAV_STATIC_CONFIRM_KEEP_MISSES=18"
Environment="UAV_STATIC_CONFIRM_MAX_YOLO_GAP_FRAMES=90"
Environment="UAV_STATIC_CONFIRM_MAX_AGE_FRAMES=90"
Environment="UAV_STATIC_CONFIRM_MIN_SCORE=0.36"
Environment="UAV_STATIC_CONFIRM_MIN_MEAN_CONF=0.34"
Environment="UAV_STATIC_CONFIRM_MAX_CENTER_JITTER=36"
Environment="UAV_STATIC_CONFIRM_MAX_BOX_JITTER_RATIO=0.80"
Environment="UAV_STATIC_CONFIRM_MAX_SPEED_PX_PER_FRAME=3.5"
Environment="UAV_STATIC_CONFIRM_MAX_RECENT_NET_MOTION_PX=42"
Environment="UAV_HOVER_HOLD=1"
Environment="UAV_HOVER_ENTER_YOLO_HITS=4"
Environment="UAV_HOVER_ENTER_RECENT_HITS=3"
Environment="UAV_HOVER_RECHECK_INTERVAL_FRAMES=12"
Environment="UAV_HOVER_MAX_RECHECK_MISSES=12"
Environment="UAV_HOVER_MAX_YOLO_GAP_FRAMES=90"
Environment="UAV_HOVER_MIN_TRACK_SCORE=0.34"
Environment="UAV_TRACK_MAX_SEARCH_MISSES=12"
Environment="UAV_DIRECT_FULL_FRAME_INFERENCE=0"
Environment="UAV_MOTION_ZOOM_CROP_SIZE=0"
Environment="UAV_MOTION_ZOOM_MAX_ROIS=0"
Environment="UAV_MOTION_ZOOM_ONLY=0"
Environment="UAV_TRACK_ZOOM_CROP_SIZE=0"
Environment="UAV_TRACK_SEARCH_ROIS=0"
Environment="UAV_FULLFRAME_SEARCH_FALLBACK=0"
Environment="UAV_FULLFRAME_FALLBACK_ONLY=0"
Environment="UAV_NIGHT_STATIC_POINT_SUPPRESS=0"
Environment="UAV_VIDEO_TARGET_IP=$VideoTargetIp"
Environment="UAV_VIDEO_BASE_PORT=$VideoPort"
Environment="UAV_VIDEO_STREAM_CAMS="
Environment="UAV_VIDEO_STREAM_W=$VideoWidth"
Environment="UAV_VIDEO_STREAM_H=$VideoHeight"
Environment="UAV_VIDEO_SEND_EVERY_N_FRAMES=$VideoSendEvery"
Environment="UAV_VIDEO_JPEG_QUALITY=$VideoJpegQuality"
Environment="UAV_VIDEO_SNDBUF_BYTES=262144"
Environment="UAV_DRAW_INTERMEDIATE_BOXES=0"
Environment="UAV_MAX_DRAW_BOXES=8"
Environment="UAV_SHOW_WINDOWS=0"
Environment="DISPLAY=:0"
Environment="XAUTHORITY=/home/Tronlong/.Xauthority"
"@

$EncodedDropIn = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($DropIn))
$DropInPath = "/etc/systemd/system/$Service.d/zzzzz-uav-scheme-a.conf"
$CleanupCommand = @"
dir="/etc/systemd/system/$Service.d"
mkdir -p "`$dir"
find "`$dir" -maxdepth 1 -type f \( -name '10-uav-ui.conf' -o -name '20-uav-video-receiver.conf' -o -name '30-*.conf' -o -name '35-uav-scene-mode.conf' -o -name '99-*.conf' -o -name 'zz*.conf' \) ! -name 'zzzzz-uav-scheme-a.conf' -delete
"@

foreach ($BoardIp in $BoardIps) {
    $Remote = "$User@$BoardIp"
    Write-Host "=== $BoardIp -> scheme A complex-scene 500m ==="

    if ($DeployModel) {
        if (-not (Test-Path -LiteralPath $DayModelLocal)) {
            throw "Local day model not found: $DayModelLocal"
        }
        $ModelName = Split-Path -Leaf $SelectedModelRemote
        $TmpRemote = "/tmp/$ModelName"
        scp $DayModelLocal "${Remote}:$TmpRemote"
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
        ssh $Remote "tr '\0' '\n' < /proc/$MainPid/environ | grep -E 'UAV_(SCHEME_NAME|LOW_LAYER_PROFILE_NAME|SCENE_MODE|LAYER_MODE|AUTO_CAM_LAYER|EDGE_REGION_LAYER|RKNN_MODEL|CAPTURE|DIFF|PROCESS|HOVER|VIDEO_STREAM|VIDEO_SEND|VIDEO_JPEG|DRAW|YOLO_CONF)' | sort"
    }
}
