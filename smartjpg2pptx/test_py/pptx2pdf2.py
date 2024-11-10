import os
import comtypes.client
from pypdfium2 import PdfDocument

def pptx_to_pdf(pptx_path, pdf_path):
    try:
        powerpoint = comtypes.client.CreateObject("Powerpoint.Application")
        powerpoint.Visible = 1

        deck = powerpoint.Presentations.Open(pptx_path)
        deck.SaveAs(pdf_path, FileFormat=32)  # 32 stands for "PDF"
        deck.Close()
        powerpoint.Quit()
    except Exception as e:
        print(f"Error converting {pptx_path} to PDF: {str(e)}")
        powerpoint.Quit()


def set_pdf_permissions(pdf_path):
    pdf = PdfDocument(pdf_path)
    permissions = {
        "print": False,
        "modify": False,
        "copy": False,
        "annot_forms": False,
        "fill_forms": False,
        "extract": False,
        "assemble": False,
        "print_high_res": False
    }
    # Set password protection (owner_password, user_password)
    pdf.set_permissions("owner_password", **permissions)
    pdf.save(pdf_path)
    pdf.close()

def scan_folder_and_convert(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pptx"):
                pptx_path = os.path.join(root, file)
                pdf_path = os.path.splitext(pptx_path)[0] + '.pdf'
                print(f"Converting: {pptx_path} to {pdf_path}")
                pptx_to_pdf(pptx_path, pdf_path)
                set_pdf_permissions(pdf_path)

if __name__ == "__main__":
    # folder_path = input("Enter the path of the folder containing PPTX files: ")
    folder_path = r"D:/培训课件"
    scan_folder_and_convert(folder_path)
    print("Conversion completed.")
