<#
  一键运行这个智能体项目。

    .\run.ps1 "帮我调研一下国产 Agent 框架"
    .\run.ps1                       # 不带参数会提示输入
    .\run.ps1 -UseFramework "..."   # 走框架原生入口（需先 -Install）
    .\run.ps1 -Install              # 只安装 requirements.txt 里的框架依赖

  这个目录是自包含的：runtime_sdk.py / runner.py 都在这里，
  换成任何一台装了 Python 3.9+ 的机器都能直接跑，不依赖 MineAgent 平台。
#>
param(
    [switch]$Install,
    [switch]$UseFramework,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$InputArgs
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$python = 'python'
if (Test-Path (Join-Path $Root '.venv\Scripts\python.exe')) {
    $python = Join-Path $Root '.venv\Scripts\python.exe'
}

if ($Install) {
    Write-Host '>>> 安装 requirements.txt…' -ForegroundColor Cyan
    & $python -m pip install -r requirements.txt
    if (-not $UseFramework) { exit 0 }
}

$entry = if ($UseFramework) { 'agent.py' } else { 'runner.py' }
$text = if ($InputArgs -and $InputArgs.Count -gt 0) { $InputArgs -join ' ' }
        else { Read-Host '请输入要交给智能体的任务' }
if (-not $text) { Write-Host '没有输入，退出。' -ForegroundColor Yellow; exit 1 }

Write-Host ">>> 用 $entry 开始运行" -ForegroundColor Cyan
& $python $entry --input $text
