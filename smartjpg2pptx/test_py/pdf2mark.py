import os
import io
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

def create_watermark(content):
    # 创建一个 PDF 文件流
    packet = io.BytesIO()
    # 设置水印页面大小为A4
    can = canvas.Canvas(packet, pagesize=A4)
    # 设置水印文本属性（位置、内容、字体、大小）
    can.setFont("Helvetica", 16)  # 设置字体和大小
    can.setFillColor("gray")      # 设置水印颜色
    # 计算水印文本位置（这里假设在页面中央，稍微向下调整）
    can.drawString(297.5, 420, content)  # A4纸张大小的中心点（点位置需自行调整）
    can.save()
    packet.seek(0)
    return packet

def add_watermark_to_pdf(input_pdf_path, output_pdf_path, watermark_content):
    watermark = create_watermark(watermark_content)
    watermark_reader = PdfReader(watermark)
    watermark_page = watermark_reader.pages[0]

    pdf_reader = PdfReader(input_pdf_path)
    pdf_writer = PdfWriter()

    # 将水印应用到每一页
    for page in pdf_reader.pages:
        page.merge_page(watermark_page)
        pdf_writer.add_page(page)

    with open(output_pdf_path, 'wb') as out:
        pdf_writer.write(out)

def scan_folder_and_convert(folder_path):
    watermark_content = "内部培训资料严禁传播"
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                output_pdf_path = pdf_path.replace('.pdf', '_watermarked.pdf')
                print(f"Processing: {pdf_path}")
                add_watermark_to_pdf(pdf_path, output_pdf_path, watermark_content)
                print(f"Watermarked file saved as: {output_pdf_path}")



if __name__ == "__main__":
    # folder_path = input("Enter the path of the folder containing PDF files: ")
    # folder_path = r"D:/培训课件"
    folder_path = r"D:/BaiduNetdiskDownload/Office脚本激活工具/"
    scan_folder_and_convert(folder_path)
    print("Processing completed.")
