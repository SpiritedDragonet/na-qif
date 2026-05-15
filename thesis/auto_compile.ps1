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
    Invoke-Tool "xelatex" @("-interaction=nonstopmode", "$texBase.tex")
    Invoke-Tool "bibtex" @($texBase)
    Invoke-Tool "xelatex" @("-interaction=nonstopmode", "$texBase.tex")
    Invoke-Tool "xelatex" @("-interaction=nonstopmode", "$texBase.tex")
    Write-Host "========== 编译完成 ==========`n" -ForegroundColor Green
    Compile-WordThesis
}

function Compile-WordThesis {
    $wordScript = Join-Path $scriptDir "compile_word.ps1"
    Invoke-Tool "powershell" @("-ExecutionPolicy", "Bypass", "-File", $wordScript)
}

function Invoke-Tool {
    param(
        [string]$Command,
        [string[]]$Arguments
    )
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Command failed with exit code $LASTEXITCODE"
    }
}

function Test-InExcludedDir {
    param([string]$Path)
    $relative = $Path.Substring($scriptDir.Length).TrimStart("\", "/")
    foreach ($dir in @(".git", ".codex_tmp", ".pytest_cache")) {
        if ($relative -eq $dir -or $relative.StartsWith("$dir\") -or $relative.StartsWith("$dir/")) {
            return $true
        }
    }
    return $false
}

function Get-FilesHash {
    $sourceFiles = foreach ($pattern in @("*.tex", "*.bib", "*.cls", "*.sty", "*.cfg")) {
        Get-ChildItem -Path $scriptDir -Filter $pattern -File -Recurse -ErrorAction SilentlyContinue |
            Where-Object { -not (Test-InExcludedDir $_.FullName) }
    }
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
