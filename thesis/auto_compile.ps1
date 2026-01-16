# 获取脚本所在目录（无论从哪里运行都能正确定位）
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$texFile = "thesis.tex"
$compileCommand = "xelatex -interaction=nonstopmode $texFile"
$bibCommand = "bibtex thesis"

# 清理临时文件
function Clean-TexFiles {
    Write-Host "清理临时文件..." -ForegroundColor Magenta
    $extensions = @("*.aux", "*.log", "*.out", "*.toc", "*.lof", "*.lot", "*.bbl", "*.blg",
                    "*.synctex.gz", "*.fls", "*.fdb_latexmk", "*.dvi", "*.xdv",
                    "*.nav", "*.snm", "*.vrb", "*.bcf", "*.run.xml")
    Get-ChildItem -Path $scriptDir -Include $extensions -Recurse -ErrorAction SilentlyContinue |
        Remove-Item -Force -ErrorAction SilentlyContinue
    Write-Host "清理完成!" -ForegroundColor Magenta
}

# 编译函数
function Compile-Thesis {
    Write-Host "开始编译..." -ForegroundColor Green
    Invoke-Expression $compileCommand
    Invoke-Expression $bibCommand
    Invoke-Expression $compileCommand
    Invoke-Expression $compileCommand
    Write-Host "编译完成!" -ForegroundColor Green
}

# 初次编译
Compile-Thesis

# 监控文件变化
$watcher = New-Object System.IO.FileSystemWatcher
$watcher.Path = $scriptDir
$watcher.IncludeSubdirectories = $true
$watcher.EnableRaisingEvents = $true

# 防抖：记录上次编译时间
$script:lastCompileTime = Get-Date

$action = {
    $path = $Event.SourceEventArgs.FullPath
    $name = $Event.SourceEventArgs.Name
    $now = Get-Date

    # 防抖：2秒内不重复编译
    if (($now - $script:lastCompileTime).TotalSeconds -lt 2) { return }
    $script:lastCompileTime = $now

    Write-Host "[$($now.ToString('HH:mm:ss'))] 检测到变化: $name" -ForegroundColor Yellow

    Set-Location $scriptDir
    Compile-Thesis
    Clean-TexFiles
}

# 监控的文件类型：tex, bib, cls, sty, cfg
$filters = @("*.tex", "*.bib", "*.cls", "*.sty", "*.cfg")

# 为每种文件类型创建监控器
$watchers = @()
$events = @()

foreach ($filter in $filters) {
    $w = New-Object System.IO.FileSystemWatcher
    $w.Path = $scriptDir
    $w.Filter = $filter
    $w.IncludeSubdirectories = $true
    $w.EnableRaisingEvents = $true

    $watchers += $w
    $events += Register-ObjectEvent $w "Changed" -Action $action
    $events += Register-ObjectEvent $w "Created" -Action $action
}

Write-Host ""
Write-Host "正在监控文件变化 (tex/bib/cls/sty/cfg)..." -ForegroundColor Cyan
Write-Host "按 Ctrl+C 停止" -ForegroundColor Cyan
Write-Host ""

try {
    while ($true) { Start-Sleep -Seconds 1 }
} finally {
    foreach ($e in $events) {
        Unregister-Event -SourceIdentifier $e.Name -ErrorAction SilentlyContinue
    }
    foreach ($w in $watchers) {
        $w.Dispose()
    }
    Write-Host "监控已停止" -ForegroundColor Red
}
