[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
$ingestScript = Join-Path $projectRoot 'ingest.py'
$logDirectory = Join-Path $projectRoot 'logs'
$logPath = Join-Path $logDirectory 'refresh.log'

if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw "Dedicated Python environment is missing. Run: uv sync --dev"
}

New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
if ((Test-Path -LiteralPath $logPath) -and (Get-Item -LiteralPath $logPath).Length -gt 5MB) {
    $archiveName = 'refresh-{0}.log' -f (Get-Date -Format 'yyyyMMdd-HHmmss')
    Move-Item -LiteralPath $logPath -Destination (Join-Path $logDirectory $archiveName)
}

function Protect-LogText {
    param([string]$Value)
    $safe = $Value
    $safe = $safe -replace '(?is)-----BEGIN(?: [A-Z0-9]+)? PRIVATE KEY-----.*?-----END(?: [A-Z0-9]+)? PRIVATE KEY-----', '[REDACTED]'
    $safe = $safe -replace '\bsk-(?:proj-|live-)?[A-Za-z0-9_-]{16,}\b', '[REDACTED]'
    $safe = $safe -replace '\bsk-ant-[A-Za-z0-9_-]{16,}\b', '[REDACTED]'
    $safe = $safe -replace '\bgh[opusr]_[A-Za-z0-9]{20,}\b', '[REDACTED]'
    $safe = $safe -replace '\bxox[baprs]-[A-Za-z0-9-]{10,}\b', '[REDACTED]'
    $safe = $safe -replace '(?im)(password|passwd|pwd|api[_-]?key|access[_-]?token|secret)\s*[:=]\s*\S+', '$1=[REDACTED]'
    return $safe
}

$startedAt = Get-Date
$output = & $pythonExe $ingestScript --incremental 2>&1 | Out-String
$exitCode = $LASTEXITCODE
$record = @(
    ('[{0}] refresh_start' -f $startedAt.ToString('o'))
    (Protect-LogText -Value $output.Trim())
    ('[{0}] refresh_end exit_code={1}' -f (Get-Date).ToString('o'), $exitCode)
) -join [Environment]::NewLine
Add-Content -LiteralPath $logPath -Value $record -Encoding utf8
exit $exitCode
