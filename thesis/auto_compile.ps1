$texFile = "thesis.tex"
$compileCommand = "& xelatex -interaction=nonstopmode $texFile"
$bibCommand = "& bibtex thesis"
$pdfFile = "thesis.pdf"
$docxFile = "thesis.docx"
$pandocExe = "C:\Program Files\Pandoc\pandoc.exe"
$templateDocx = "thesis_template.docx"

# Function to clean up temporary files
function Clean-TexFiles {
    Write-Host "Cleaning up temporary files..." -ForegroundColor Magenta
    
    # Remove common LaTeX temporary files
    Get-ChildItem -Path $PWD -Include "*.aux", "*.log", "*.out", "*.toc", "*.lof", "*.lot", "*.bbl", "*.blg", "*.synctex.gz", "*.fls", "*.fdb_latexmk", "*.dvi", "*.xdv", "*.nav", "*.snm", "*.vrb", "*.bcf", "*.run.xml" -Recurse | ForEach-Object {
        Remove-Item $_.FullName -Force
        Write-Host "Removed: $($_.Name)" -ForegroundColor Gray
    }
    
    Write-Host "Cleanup completed!" -ForegroundColor Magenta
}

# Function to convert PDF to DOCX using Pandoc
function Convert-ToDocx {
    if (Test-Path $pdfFile) {
        Write-Host "Converting PDF to DOCX using Pandoc..." -ForegroundColor Cyan
        try {
            # 检查是否存在模板文件
            if (Test-Path $templateDocx) {
                Write-Host "Using custom Word template: $templateDocx" -ForegroundColor Cyan
                # 使用参考模板进行转换
                & $pandocExe -f latex -t docx -o $docxFile --reference-doc=$templateDocx --toc --number-sections $texFile
            } else {
                # 标准转换
                & $pandocExe -f latex -t docx -o $docxFile --toc --number-sections $texFile
                Write-Host "No custom template found. For better formatting, consider creating a Word template." -ForegroundColor Yellow
            }
            
            if (Test-Path $docxFile) {
                Write-Host "Successfully converted to DOCX: $docxFile" -ForegroundColor Green
            } else {
                Write-Host "Failed to create DOCX file" -ForegroundColor Red
            }
        } catch {
            Write-Host "Error during Pandoc conversion: $_" -ForegroundColor Red
        }
    } else {
        Write-Host "PDF file not found, cannot convert to DOCX" -ForegroundColor Red
    }
}

# 创建一个基本的Word模板函数
function Create-DocxTemplate {
    # 如果还没有模板，创建一个基本模板
    if (-not (Test-Path $templateDocx)) {
        try {
            Write-Host "Creating a basic Word template..." -ForegroundColor Cyan
            # 首先创建一个普通的docx
            & $pandocExe -f latex -t docx -o $templateDocx $texFile
            
            Write-Host "Basic template created: $templateDocx" -ForegroundColor Green
            Write-Host "Please edit this template in Word to set your desired styles," -ForegroundColor Yellow
            Write-Host "then save it and run the conversion again for better results." -ForegroundColor Yellow
        } catch {
            Write-Host "Error creating template: $_" -ForegroundColor Red
        }
    }
}

# Initial compilation
Write-Host "Initial compilation..." -ForegroundColor Green
Invoke-Expression $compileCommand
Invoke-Expression $bibCommand
Invoke-Expression $compileCommand
Invoke-Expression $compileCommand
Write-Host "Initial compilation completed!" -ForegroundColor Green

# Create Word template if needed
Create-DocxTemplate

# Convert to DOCX after initial compilation
Convert-ToDocx

# Optional cleanup after initial compilation
# Clean-TexFiles

# Watch for file changes
$watcher = New-Object System.IO.FileSystemWatcher
$watcher.Path = $PWD
$watcher.Filter = "*.tex"
$watcher.IncludeSubdirectories = $true
$watcher.EnableRaisingEvents = $true

$action = {
    $path = $Event.SourceEventArgs.FullPath
    $name = $Event.SourceEventArgs.Name
    $changeType = $Event.SourceEventArgs.ChangeType
    $timeStamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    
    Write-Host "[$timeStamp] File $name $changeType" -ForegroundColor Yellow
    Write-Host "Starting compilation..." -ForegroundColor Green
    
    # Compilation commands
    Invoke-Expression $compileCommand
    Invoke-Expression $bibCommand
    Invoke-Expression $compileCommand
    Invoke-Expression $compileCommand
    
    Write-Host "Compilation completed!" -ForegroundColor Green
    
    # Convert to DOCX after compilation
    Convert-ToDocx
    
    # Clean up temporary files after successful compilation
    Clean-TexFiles
}

# Register events
$changed = Register-ObjectEvent $watcher "Changed" -Action $action
$created = Register-ObjectEvent $watcher "Created" -Action $action

Write-Host "Watching for file changes, press Ctrl+C to stop..." -ForegroundColor Cyan
Write-Host "The script will automatically compile to PDF and convert to DOCX" -ForegroundColor Cyan

try {
    # Keep script running
    while ($true) { Start-Sleep -Seconds 1 }
} finally {
    # Cleanup
    Unregister-Event -SourceIdentifier $changed.Name
    Unregister-Event -SourceIdentifier $created.Name
    $watcher.Dispose()
    Write-Host "Monitoring stopped" -ForegroundColor Red
} 