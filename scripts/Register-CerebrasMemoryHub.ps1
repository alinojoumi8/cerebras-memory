<#
.SYNOPSIS
Register the Cerebras Memory hub listener as a Windows boot task.

.DESCRIPTION
The hub is the process other machines reach. It has to come back on its own
after a restart, so it runs from a scheduled task at startup rather than from a
console someone has to remember to open.

LogonType S4U runs it whether or not anyone is logged on, and without storing a
password. It only needs local resources -- the SQLite file, the ONNX models, and
a loopback socket -- so the S4U restriction on network resources does not apply.
Publishing it to the tailnet stays the job of `tailscale serve`, which is a
separate service and is configured once with --bg.

The task deliberately does not set CEREBRAS_MEMORY_HTTP_TOKENS: the hub reads it
from the ignored .env at startup, so the secret is never written into the task
definition, where it would be readable by anything that can query the scheduler.

.EXAMPLE
.\scripts\Register-CerebrasMemoryHub.ps1 -AllowedHost matrix.taila13ed8.ts.net

.EXAMPLE
.\scripts\Register-CerebrasMemoryHub.ps1 -Unregister
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    # Host headers the hub will accept. Must include the tailnet name, because
    # `tailscale serve` forwards that rather than the loopback address.
    [string[]]$AllowedHost = @(),
    [int]$Port = 8791,
    # Start at logon as the signed-in user instead of at boot. A boot trigger
    # runs before anyone logs on, which Windows only lets an administrator
    # create; this variant needs no elevation but the hub is only up once you
    # have logged in.
    [switch]$AtLogon,
    [switch]$Unregister,
    [switch]$NoStart
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
$hubScript = Join-Path $projectRoot 'http_server.py'
$taskName = 'CerebrasMemoryHub'

if (-not (Get-Command Register-ScheduledTask -ErrorAction SilentlyContinue)) {
    throw 'Windows ScheduledTasks cmdlets are unavailable.'
}

if ($Unregister) {
    if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
        if ($PSCmdlet.ShouldProcess("Windows task $taskName", 'unregister')) {
            Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
            Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
        }
    }
    [pscustomobject]@{ Task = $taskName; State = 'unregistered' }
    return
}

if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw "Dedicated Python environment is missing: $pythonExe"
}
if (-not (Test-Path -LiteralPath $hubScript -PathType Leaf)) {
    throw "Hub entry point is missing: $hubScript"
}
if (-not $AllowedHost -or $AllowedHost.Count -eq 0) {
    throw 'Specify -AllowedHost, for example -AllowedHost matrix.taila13ed8.ts.net'
}

# Refuse to serve from inside the corpus. The hub enforces this itself at
# startup, but a task whose working directory is wrong would only fail after a
# reboot, which is the worst time to find out.
$settings = & $pythonExe -c "import json, sys; sys.path.insert(0, r'$projectRoot'); from config import load_settings; print(json.dumps({'projects_root': str(load_settings().projects_root)}))"
if ($LASTEXITCODE -ne 0) {
    throw 'Could not load configuration to validate the working directory.'
}
$projectsRoot = (ConvertFrom-Json $settings).projects_root
if ($projectRoot -eq $projectsRoot -or $projectRoot.StartsWith($projectsRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Working directory $projectRoot is inside projects_root $projectsRoot; the hub would refuse to start."
}

$identityName = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$hostArguments = ($AllowedHost | ForEach-Object { '--allowed-host "{0}"' -f $_ }) -join ' '
$actionArguments = '"{0}" --port {1} {2}' -f $hubScript, $Port, $hostArguments

$isElevated = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
)
if (-not $AtLogon -and -not $isElevated) {
    throw @"
Registering a boot task needs an elevated session, because a task that starts
before logon can only be created by an administrator.

Either run this script from an elevated PowerShell:
    .\scripts\Register-CerebrasMemoryHub.ps1 -AllowedHost $($AllowedHost -join ',')

or register it to start at logon instead, which needs no elevation:
    .\scripts\Register-CerebrasMemoryHub.ps1 -AtLogon -AllowedHost $($AllowedHost -join ',')
"@
}

$action = New-ScheduledTaskAction -Execute $pythonExe -Argument $actionArguments -WorkingDirectory $projectRoot
$trigger = if ($AtLogon) {
    New-ScheduledTaskTrigger -AtLogOn -User $identityName
} else {
    New-ScheduledTaskTrigger -AtStartup
}
$taskSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -DontStopOnIdleEnd `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero)

# S4U runs the task whether or not anyone is logged on and stores no password,
# but registering it is itself a privileged operation. The logon variant needs
# only Interactive, which any user may register for themselves.
$principal = if ($AtLogon) {
    New-ScheduledTaskPrincipal -UserId $identityName -LogonType Interactive -RunLevel Limited
} else {
    New-ScheduledTaskPrincipal -UserId $identityName -LogonType S4U -RunLevel Limited
}
$task = New-ScheduledTask `
    -Action $action `
    -Trigger $trigger `
    -Settings $taskSettings `
    -Principal $principal `
    -Description 'Serve the private Cerebras Memory knowledge base on loopback for tailnet clients.'

$started = $false
if ($PSCmdlet.ShouldProcess("Windows task $taskName", 'register or update')) {
    Register-ScheduledTask -TaskName $taskName -InputObject $task -Force | Out-Null
    # Confirm rather than assume. Register-ScheduledTask can fail with a
    # non-terminating CIM error, and a summary that reports success anyway is
    # worse than no summary at all.
    if (-not (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue)) {
        throw "Registration reported no error but $taskName does not exist."
    }
    if (-not $NoStart) {
        Start-ScheduledTask -TaskName $taskName
        $started = $true
    }
}

[pscustomobject]@{
    Task = $taskName
    Trigger = if ($AtLogon) { 'at logon' } else { 'at startup' }
    LogonType = if ($AtLogon) { 'Interactive (runs once you log on)' } else { 'S4U (runs whether or not you are logged on)' }
    Port = $Port
    AllowedHosts = ($AllowedHost -join ', ')
    WorkingDirectory = $projectRoot
    Started = $started
}
