# 获取脚本所在目录
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$texFile = "thesis.tex"

# 编译函数
function Compile-Thesis {
    Write-Host "`n========== 开始编译 ==========" -ForegroundColor Green
    xelatex -interaction=nonstopmode $texFile
    bibtex thesis
    xelatex -interaction=nonstopmode $texFile
    xelatex -interaction=nonstopmode $texFile
    Write-Host "========== 编译完成 ==========`n" -ForegroundColor Green
}

# 获取所有监控文件的哈希
function Get-FilesHash {
    $files = Get-ChildItem -Path $scriptDir -Include @("*.tex", "*.bib", "*.cls", "*.sty", "*.cfg") -Recurse -ErrorAction SilentlyContinue
    $hashString = ""
    foreach ($file in $files) {
        $hashString += "$($file.FullName)|$($file.LastWriteTime)|"
    }
    return $hashString
}

# 初次编译
Compile-Thesis

Write-Host "正在监控文件变化 (tex/bib/cls/sty/cfg)..." -ForegroundColor Cyan
Write-Host "按 Ctrl+C 停止" -ForegroundColor Cyan
Write-Host ""

# 轮询方式监控（更可靠）
$lastHash = Get-FilesHash

try {
    while ($true) {
        Start-Sleep -Milliseconds 500

        $currentHash = Get-FilesHash

        if ($currentHash -ne $lastHash) {
            $lastHash = $currentHash
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] 检测到文件变化" -ForegroundColor Yellow

            # 等待文件写入完成
            Start-Sleep -Milliseconds 300

            Compile-Thesis
        }
    }
} finally {
    Write-Host "监控已停止" -ForegroundColor Red
}
