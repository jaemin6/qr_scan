import subprocess
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

def create_pdf(script_path, output_path):
    """
    파이썬 스크립트 내용과 실행 결과를 PDF로 저장합니다.
    """
    try:
        # 1. 파이썬 스크립트 실행 및 결과 캡처
        # subprocess를 사용하여 스크립트를 실행하고, 표준 출력을 캡처합니다.
        # stderr=subprocess.STDOUT를 사용해 오류 메시지도 함께 캡처합니다.
        print(f"Executing script: {script_path}...")
        result = subprocess.run(['python', script_path], capture_output=True, text=True, encoding='utf-8')
        output = result.stdout + result.stderr

        print("Script execution complete. Creating PDF...")

        # 2. PDF 문서 생성
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # 사용자 정의 스타일 (헤더, 본문)
        title_style = ParagraphStyle('TitleStyle', parent=styles['Normal'], fontSize=16, leading=20, alignment=TA_CENTER)
        code_style = ParagraphStyle('CodeStyle', parent=styles['Normal'], fontName='Courier', fontSize=10, leading=12, backColor='#f0f0f0')
        output_style = ParagraphStyle('OutputStyle', parent=styles['Normal'], fontName='Courier', fontSize=10, leading=12, backColor='#fff0e6')

        elements = []
        
        # PDF 제목 추가
        elements.append(Paragraph("Python Script and Execution Result", title_style))
        elements.append(Spacer(1, 12))

        # 3. 파이썬 스크립트 내용 추가
        elements.append(Paragraph(f"<b>Code from '{script_path}':</b>", styles['Normal']))
        elements.append(Spacer(1, 6))
        with open(script_path, 'r', encoding='utf-8') as f:
            code_content = f.read()
            elements.append(Paragraph(code_content, code_style))
        
        elements.append(Spacer(1, 24))

        # 4. 실행 결과 추가
        elements.append(Paragraph("<b>Execution Output:</b>", styles['Normal']))
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(output, output_style))

        # 5. PDF 파일 빌드 및 저장
        doc.build(elements)
        print(f"PDF file created successfully at: {output_path}")
        
    except FileNotFoundError:
        print(f"Error: The file '{script_path}' was not found. Please check the file path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    # PDF로 만들고 싶은 파이썬 파일명과 결과 PDF 파일명을 지정하세요.
    script_to_run = 'qr_scanner_web.py'
    output_pdf_file = 'project_report.pdf'
    create_pdf(script_to_run, output_pdf_file)
