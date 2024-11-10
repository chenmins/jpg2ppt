Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName Microsoft.Office.Interop.PowerPoint
$Application = New-Object -ComObject PowerPoint.Application

# ����һ�� FolderBrowserDialog ������ѡ���ļ���
$folderBrowser = New-Object System.Windows.Forms.FolderBrowserDialog
$folderBrowser.Description = "ѡ����� PPTX �ļ����ļ���"
$folderBrowser.ShowNewFolderButton = $false
if ($folderBrowser.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
    $folderPath = $folderBrowser.SelectedPath
    # ��ȡ���� PPTX �ļ�
    $pptxFiles = Get-ChildItem -Path $folderPath -Filter *.pptx
    foreach ($file in $pptxFiles) {
        $fullPath = $file.FullName
        $pdfPath = [System.IO.Path]::ChangeExtension($fullPath, 'pdf')
        Write-Host "����ת��: $fullPath"
        $presentation = $Application.Presentations.Open($fullPath)
        $presentation.SaveAs($pdfPath, [Microsoft.Office.Interop.PowerPoint.PpSaveAsFileType]::ppSaveAsPDF)
        $presentation.Close()
        Write-Host "�ѱ���Ϊ: $pdfPath"
    }
    $Application.Quit()
} else {
    Write-Host "δѡ���κ��ļ��С�"
}

[System.Runtime.InteropServices.Marshal]::ReleaseComObject($Application)
[System.GC]::Collect()
[System.GC]::WaitForPendingFinalizers()
