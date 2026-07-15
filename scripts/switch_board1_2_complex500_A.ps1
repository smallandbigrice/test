param(
    [string[]]$BoardIps = @("192.168.0.1", "192.168.0.2"),
    [string]$User = "root",
    [string]$Service = "python_autostar.service",
    [switch]$NoRestart
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ProfileDir = Join-Path $RepoRoot "deploy_profiles\scheme_A_complex500"
$MainLocal = Join-Path $ProfileDir "main.py"
$YoloLocal = Join-Path $ProfileDir "yololib.py"
$CommsLocal = Join-Path $ProfileDir "comms.py"
$DropInLocal = Join-Path $ProfileDir "zzzzz-uav-scene-mode.conf"

foreach ($Path in @($MainLocal, $YoloLocal, $CommsLocal, $DropInLocal)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing profile file: $Path"
    }
}

$RemotePythonDir = "/home/Tronlong/rknn_model_zoo/examples/yolov5/python"
$RemoteDropIn = "/etc/systemd/system/$Service.d/zzzzz-uav-scene-mode.conf"

foreach ($BoardIp in $BoardIps) {
    $Remote = "$User@$BoardIp"
    Write-Host "=== $BoardIp -> Scheme A complex500 ==="

    ssh $Remote @"
set -e
ts=`$(date +%Y%m%d_%H%M%S)
bk=/userdata/uav_backups/scheme_A_complex500_before_`$ts
mkdir -p "`$bk" /etc/systemd/system/$Service.d
cp $RemotePythonDir/main.py "`$bk/main.py" 2>/dev/null || true
cp $RemotePythonDir/yololib.py "`$bk/yololib.py" 2>/dev/null || true
cp $RemotePythonDir/comms.py "`$bk/comms.py" 2>/dev/null || true
cp $RemoteDropIn "`$bk/zzzzz-uav-scene-mode.conf" 2>/dev/null || true
echo backup=`$bk
"@

    scp $MainLocal "${Remote}:$RemotePythonDir/main.py"
    scp $YoloLocal "${Remote}:$RemotePythonDir/yololib.py"
    scp $CommsLocal "${Remote}:$RemotePythonDir/comms.py"
    scp $DropInLocal "${Remote}:$RemoteDropIn"

    ssh $Remote "python3 -m py_compile $RemotePythonDir/main.py && systemctl daemon-reload"
    if (-not $NoRestart) {
        ssh $Remote "systemctl restart $Service && sleep 3"
    }
    ssh $Remote "systemctl is-active $Service; systemctl show $Service -p MainPID"
}
