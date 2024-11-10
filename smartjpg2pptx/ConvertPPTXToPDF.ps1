Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName Microsoft.Office.Interop.PowerPoint
$Application = New-Object -ComObject PowerPoint.Application

# 创建一个 FolderBrowserDialog 对象来选择文件夹
$folderBrowser = New-Object System.Windows.Forms.FolderBrowserDialog
$folderBrowser.Description = "选择包含 PPTX 文件的文件夹"
$folderBrowser.ShowNewFolderButton = $false
if ($folderBrowser.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
    $folderPath = $folderBrowser.SelectedPath
    # 获取所有 PPTX 文件
    $pptxFiles = Get-ChildItem -Path $folderPath -Filter *.pptx
    foreach ($file in $pptxFiles) {
        $fullPath = $file.FullName
        $pdfPath = [System.IO.Path]::ChangeExtension($fullPath, 'pdf')
        Write-Host "正在转换: $fullPath"
        $presentation = $Application.Presentations.Open($fullPath)
        $presentation.SaveAs($pdfPath, [Microsoft.Office.Interop.PowerPoint.PpSaveAsFileType]::ppSaveAsPDF)
        $presentation.Close()
        Write-Host "已保存为: $pdfPath"
    }
    $Application.Quit()
} else {
    Write-Host "未选择任何文件夹。"
}

[System.Runtime.InteropServices.Marshal]::ReleaseComObject($Application)
[System.GC]::Collect()
[System.GC]::WaitForPendingFinalizers()
