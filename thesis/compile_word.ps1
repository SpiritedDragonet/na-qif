param(
    [string]$Output = "",
    [switch]$NoRefresh
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

if ([string]::IsNullOrWhiteSpace($Output)) {
    $outputPath = Join-Path $scriptDir "thesis.docx"
} elseif ([System.IO.Path]::IsPathRooted($Output)) {
    $outputPath = $Output
} else {
    $outputPath = Join-Path $scriptDir $Output
}

$docxArgs = @("-m", "docx_export", "--root", $scriptDir, "--output", $outputPath)
if (-not $NoRefresh) {
    $docxArgs += "--refresh-fields"
}

Write-Host "`n========== Building Word DOCX ==========" -ForegroundColor Green
python @docxArgs
if ($LASTEXITCODE -ne 0) {
    throw "DOCX export failed with exit code $LASTEXITCODE"
}
Write-Host "========== Word DOCX build complete ==========`n" -ForegroundColor Green
