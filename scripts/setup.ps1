param([string]$Python = 'python', [switch]$Cpu)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonVersion = & $Python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($pythonVersion -notin @('3.10', '3.11')) {
    throw 'Use Python 3.10 or 3.11. Example: powershell -File scripts/setup.ps1 -Python C:\Python310\python.exe'
}
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create virtual environment' }
}
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
& $venvPython -m pip install --upgrade pip 'setuptools<81'
if ($LASTEXITCODE -ne 0) { throw 'pip setup failed' }
$torchIndex = if ($Cpu) { 'https://download.pytorch.org/whl/cpu' } else { 'https://download.pytorch.org/whl/cu121' }
& $venvPython -m pip install torch==2.2.2 torchvision==0.17.2 --index-url $torchIndex
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
& $venvPython -m pip install -r requirements/engine.txt
if ($LASTEXITCODE -ne 0) { throw 'Engine dependency installation failed' }
& $venvPython -m pip install -e '.[export]'
if ($LASTEXITCODE -ne 0) { throw 'Tool installation failed' }
Write-Host 'Ready. Run start.bat to open the training interface.'
