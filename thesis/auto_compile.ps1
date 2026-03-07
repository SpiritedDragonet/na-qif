# 获取脚本所在目录
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# 优先使用 thesis.tex（与当前项目历史结构保持一致）
if (Test-Path "thesis.tex") {
    $texBase = "thesis"
} elseif (Test-Path "main.tex") {
    $texBase = "main"
} else {
    throw "未找到 main.tex 或 thesis.tex"
}

function Compile-Thesis {
    Write-Host "`n========== 开始编译 ==========" -ForegroundColor Green
    xelatex -interaction=nonstopmode "$texBase.tex"
    bibtex $texBase
    xelatex -interaction=nonstopmode "$texBase.tex"
    xelatex -interaction=nonstopmode "$texBase.tex"
    Write-Host "========== 编译完成 ==========`n" -ForegroundColor Green
}

function Get-FilesHash {
    $sourceFiles = Get-ChildItem -Path $scriptDir -Include @("*.tex", "*.bib", "*.cls", "*.sty", "*.cfg") -Recurse -ErrorAction SilentlyContinue
    $figuresDir = Join-Path $scriptDir "figures"
    $figureFiles = @()
    if (Test-Path $figuresDir) {
        $figureFiles = Get-ChildItem -Path $figuresDir -File -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -notmatch "\\__pycache__\\" }
    }
    $files = @($sourceFiles + $figureFiles) | Sort-Object -Property FullName -Unique
    $hashString = ""
    foreach ($file in $files) {
        $hashString += "$($file.FullName)|$($file.LastWriteTimeUtc.Ticks)|$($file.Length)|"
    }
    return $hashString
}

Compile-Thesis

Write-Host "正在监控文件变化 (tex/bib/cls/sty/cfg + figures/*)..." -ForegroundColor Cyan
Write-Host "按 Ctrl+C 停止" -ForegroundColor Cyan
Write-Host ""

$lastHash = Get-FilesHash
try {
    while ($true) {
        Start-Sleep -Milliseconds 500
        $currentHash = Get-FilesHash
        if ($currentHash -ne $lastHash) {
            $lastHash = $currentHash
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] 检测到文件变化" -ForegroundColor Yellow
            Start-Sleep -Milliseconds 300
            Compile-Thesis
        }
    }
} finally {
    Write-Host "监控已停止" -ForegroundColor Red
}
