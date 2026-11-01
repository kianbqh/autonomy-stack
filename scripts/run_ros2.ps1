param(
    [ValidateRange(1, 86400)][int]$Seconds = 180,
    [string]$DockerExecutable = 'docker'
)
$ErrorActionPreference = 'Stop'
$projectPath = Split-Path $PSScriptRoot -Parent
$dockerArgs = @(
    'run', '--rm', '--mount', "type=bind,source=$projectPath,target=/work",
    '--workdir', '/work', '--env', "RUN_SECONDS=$Seconds",
    '--env', 'HOST_UID=0', '--env', 'HOST_GID=0',
    '--env', 'OPENBLAS_NUM_THREADS=1', '--env', 'OMP_NUM_THREADS=1', '--env', 'ROS_LOCALHOST_ONLY=1',
    'ros:humble-ros-base', 'bash', 'scripts/run_ros2_container.sh'
)
& $DockerExecutable @dockerArgs
if ($LASTEXITCODE -ne 0) { throw "ROS 2 run exited with code $LASTEXITCODE" }
Get-Content (Join-Path $projectPath 'results\ros2_metrics.json') -Encoding UTF8
