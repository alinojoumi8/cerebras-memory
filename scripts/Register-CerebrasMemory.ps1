[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [switch]$SkipClients,
    [switch]$SkipTask,
    [switch]$NoRestartChatGPT
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
$serverScript = Join-Path $projectRoot 'mcp_server.py'
$refreshScript = Join-Path $projectRoot 'scripts\Refresh-CerebrasMemory.ps1'
$hermesHelper = Join-Path $projectRoot 'scripts\hermes_noninteractive.py'
$modelWarmup = Join-Path $projectRoot 'scripts\warm_models.py'
$taskName = 'CerebrasMemoryRefresh'
$serverName = 'cerebras-memory'
$profilePath = [Environment]::GetFolderPath('UserProfile')

if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw "Dedicated Python environment is missing: $pythonExe"
}
if (-not (Test-Path -LiteralPath $serverScript -PathType Leaf)) {
    throw "MCP server is missing: $serverScript"
}
if (-not (Test-Path -LiteralPath $modelWarmup -PathType Leaf)) {
    throw "Model warm-up helper is missing: $modelWarmup"
}

# Validate imports before touching any client configuration.
& $pythonExe -c "import mcp_server; assert mcp_server.mcp.name == 'cerebras-memory'"
if ($LASTEXITCODE -ne 0) {
    throw 'The MCP server failed its import check.'
}

# Ordinary searches are deliberately offline-only. Installation is the one
# explicit point at which the optional cross-encoder may be downloaded.
& $pythonExe $modelWarmup
if ($LASTEXITCODE -ne 0) {
    throw 'The local reranker failed to download or load during warm-up.'
}

$backupRoot = Join-Path $projectRoot ('backups\registration-{0}' -f (Get-Date -Format 'yyyyMMdd-HHmmss'))

function Backup-Configuration {
    param(
        [string]$Source,
        [string]$Label
    )
    if (Test-Path -LiteralPath $Source -PathType Leaf) {
        New-Item -ItemType Directory -Path $backupRoot -Force | Out-Null
        Copy-Item -LiteralPath $Source -Destination (Join-Path $backupRoot $Label) -Force
    }
}

function Invoke-Checked {
    param(
        [string]$Program,
        [string[]]$Arguments
    )
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Program exited with code $LASTEXITCODE"
    }
}

function Add-HermesServer {
    param([string[]]$Arguments)
    # Hermes asks which discovered tools to enable. This installer is itself
    # the user's explicit approval of the four documented tools. A Python
    # stdin bridge avoids Windows PowerShell 5.1's UTF-16 native pipeline.
    & $pythonExe $hermesHelper hermes @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "hermes exited with code $LASTEXITCODE"
    }
}

function Remove-HermesServer {
    & $pythonExe $hermesHelper hermes mcp remove cerebras-memory
    if ($LASTEXITCODE -ne 0) {
        throw "hermes exited with code $LASTEXITCODE"
    }
}

function Set-CodexServerBlock {
    param(
        [string]$ConfigPath,
        [string]$PythonPath,
        [string]$ServerPath
    )
    $directory = Split-Path -Parent $ConfigPath
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
    $contents = if (Test-Path -LiteralPath $ConfigPath) {
        [IO.File]::ReadAllText($ConfigPath)
    } else {
        ''
    }
    $newline = if ($contents.Contains("`r`n")) { "`r`n" } else { "`n" }
    $pattern = '(?ms)^\[mcp_servers\.cerebras-memory\]\r?\n.*?(?=^\[|\z)'
    $contents = [regex]::Replace($contents, $pattern, '')
    $pythonLiteral = $PythonPath.Replace("'", "''")
    $serverLiteral = $ServerPath.Replace("'", "''")
    $block = @(
        '[mcp_servers.cerebras-memory]'
        ("command = '{0}'" -f $pythonLiteral)
        ("args = ['{0}']" -f $serverLiteral)
    ) -join $newline
    $updated = $contents.TrimEnd("`r", "`n") + $newline + $newline + $block + $newline
    [IO.File]::WriteAllText($ConfigPath, $updated, [Text.UTF8Encoding]::new($false))
}

function Test-Entry {
    param(
        [string]$Program,
        [string[]]$Arguments
    )
    $value = (& $Program @Arguments 2>&1 | Out-String)
    return $value -match [regex]::Escape($serverName)
}

if (-not $SkipClients) {
    $hermesConfig = (& hermes config path 2>$null | Select-Object -First 1)
    if ($hermesConfig) {
        Backup-Configuration -Source $hermesConfig.Trim() -Label 'hermes-config.yaml'
    }
    Backup-Configuration -Source (Join-Path $profilePath '.claude.json') -Label 'claude-user.json'
    Backup-Configuration -Source (Join-Path $profilePath '.claude\.mcp.json') -Label 'claude-mcp.json'
    Backup-Configuration -Source (Join-Path $profilePath '.codex\config.toml') -Label 'codex-config.toml'
    Backup-Configuration -Source (Join-Path $profilePath '.grok\config.toml') -Label 'grok-config.toml'

    if ($PSCmdlet.ShouldProcess('Hermes Agent user MCP configuration', "register $serverName")) {
        if (Test-Entry -Program 'hermes' -Arguments @('mcp', 'list')) {
            Remove-HermesServer
        }
        Add-HermesServer -Arguments @(
            'mcp', 'add', $serverName, '--command', $pythonExe, '--args', $serverScript
        )
    }

    if ($PSCmdlet.ShouldProcess('Claude Code user MCP configuration', "register $serverName")) {
        if (Test-Entry -Program 'claude' -Arguments @('mcp', 'list')) {
            Invoke-Checked -Program 'claude' -Arguments @('mcp', 'remove', '--scope', 'user', $serverName)
        }
        Invoke-Checked -Program 'claude' -Arguments @(
            'mcp', 'add', '--scope', 'user', $serverName, '--', $pythonExe, $serverScript
        )
    }

    if ($PSCmdlet.ShouldProcess('Codex and ChatGPT desktop MCP configuration', "register $serverName")) {
        Set-CodexServerBlock `
            -ConfigPath (Join-Path $profilePath '.codex\config.toml') `
            -PythonPath $pythonExe `
            -ServerPath $serverScript
    }

    if ($PSCmdlet.ShouldProcess('Grok user MCP configuration', "register $serverName")) {
        if (Test-Entry -Program 'grok' -Arguments @('mcp', 'list')) {
            Invoke-Checked -Program 'grok' -Arguments @('mcp', 'remove', '--scope', 'user', $serverName)
        }
        Invoke-Checked -Program 'grok' -Arguments @(
            'mcp', 'add', '--scope', 'user', $serverName, '--', $pythonExe, $serverScript
        )
    }
}

if (-not $SkipTask) {
    if (-not (Get-Command Register-ScheduledTask -ErrorAction SilentlyContinue)) {
        throw 'Windows ScheduledTasks cmdlets are unavailable.'
    }
    $windowsPowerShell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
    $actionArguments = '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "{0}"' -f $refreshScript
    $action = New-ScheduledTaskAction -Execute $windowsPowerShell -Argument $actionArguments -WorkingDirectory $projectRoot
    $trigger = New-ScheduledTaskTrigger -Daily -At '03:00'
    $settings = New-ScheduledTaskSettingsSet `
        -StartWhenAvailable `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -DontStopOnIdleEnd `
        -MultipleInstances IgnoreNew `
        -ExecutionTimeLimit (New-TimeSpan -Hours 6)
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    $principal = New-ScheduledTaskPrincipal -UserId $identity -LogonType Interactive -RunLevel Limited
    $task = New-ScheduledTask `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Description 'Refresh the private local Cerebras Memory index at 03:00 local time.'
    if ($PSCmdlet.ShouldProcess("Windows task $taskName", 'register or update')) {
        Register-ScheduledTask -TaskName $taskName -InputObject $task -Force | Out-Null
    }
}

if (-not $SkipClients -and -not $NoRestartChatGPT) {
    # The Codex desktop host is named ChatGPT.exe, while the legacy desktop
    # package currently runs as "ChatGPT Classic.exe". Match the package path
    # as the decisive boundary so this can never terminate Codex.
    $chatGptDesktop = Get-Process -Name 'ChatGPT', 'ChatGPT Classic' -ErrorAction SilentlyContinue | Where-Object {
        $_.Path -and $_.Path -match '\\OpenAI\.ChatGPT-Desktop_'
    }
    $chatGptApp = Get-StartApps | Where-Object {
        $_.AppID -like 'OpenAI.ChatGPT-Desktop_*!*'
    } | Select-Object -First 1
    if ($chatGptApp -and $PSCmdlet.ShouldProcess('ChatGPT desktop', 'restart after Codex MCP registration')) {
        if ($chatGptDesktop) {
            $chatGptDesktop | Stop-Process
            Start-Sleep -Seconds 2
        }
        Start-Process -FilePath 'explorer.exe' -ArgumentList ('shell:AppsFolder\{0}' -f $chatGptApp.AppID)
    }
}

[pscustomobject]@{
    Server = $serverName
    Python = $pythonExe
    ServerScript = $serverScript
    BackupDirectory = if (Test-Path -LiteralPath $backupRoot) { $backupRoot } else { $null }
    Task = if ($SkipTask) { 'skipped' } else { $taskName }
}
