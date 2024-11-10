import os
import comtypes.client

def pptx_to_pdf(pptx_path, pdf_path):
    powerpoint = comtypes.client.CreateObject("Powerpoint.Application")
    powerpoint.Visible = 1

    deck = powerpoint.Presentations.Open(pptx_path)
    deck.SaveAs(pdf_path, FileFormat=32)  # 32 stands for "PDF"
    deck.Close()
    powerpoint.Quit()

def scan_folder_and_convert(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pptx"):
                pptx_path = os.path.join(root, file)
                pdf_path = os.path.splitext(pptx_path)[0] + '.pdf'
                print(f"Converting: {pptx_path} to {pdf_path}")
                pptx_to_pdf(pptx_path, pdf_path)

if __name__ == "__main__":
    # folder_path = input("Enter the path of the folder containing PPTX files: ")
    folder_path=r"D:/培训课件"
    scan_folder_and_convert(folder_path)
    print("Conversion completed.")
