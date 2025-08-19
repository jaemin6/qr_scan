# 필요한 라이브러리 가져오기
import cv2 # OpenCV 라이브러리: 이미지 및 비디오 처리에 사용됩니다.
import numpy as np # NumPy 라이브러리: 숫자 연산을 효율적으로 처리합니다.
import webbrowser # 웹 브라우저를 제어하는 라이브러리: 링크를 여는 데 사용됩니다.
from PIL import ImageFont, ImageDraw, Image # Pillow 라이브러리: 이미지에 한글을 쓰기 위해 사용됩니다.

# OpenCV용 한글 폰트 경로 지정 (Windows 기준: 맑은 고딕)
try:
    fontpath = "C:/Windows/Fonts/malgun.ttf" # 맑은 고딕 폰트 파일의 경로를 지정합니다.
    font = ImageFont.truetype(fontpath, 20) # PIL(Pillow)에서 사용할 폰트 객체를 생성합니다.
    is_font_loaded = True # 폰트가 성공적으로 로드되었음을 나타내는 플래그를 True로 설정합니다.
except IOError:
    is_font_loaded = False # 폰트 로드에 실패하면 플래그를 False로 설정합니다.
    font = None # 폰트 객체를 None으로 초기화합니다.

def put_text_on_frame(frame, text, pos, color=(255, 0, 0)):
    """한글 텍스트를 프레임에 표시하는 유틸리티 함수"""
    if is_font_loaded: # 한글 폰트가 로드되었는지 확인합니다.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # OpenCV의 BGR 색상 채널을 RGB로 변환합니다.
        pil_img = Image.fromarray(frame_rgb) # NumPy 배열을 PIL 이미지 객체로 변환합니다.
        draw = ImageDraw.Draw(pil_img) # 이미지에 그림을 그릴 수 있는 객체를 생성합니다.
        draw.text(pos, text, font=font, fill=color) # 지정된 위치에 한글 텍스트를 그립니다.
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR) # PIL 이미지를 다시 OpenCV의 BGR 형식으로 변환합니다.
    else: # 한글 폰트가 로드되지 않았을 경우
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2) # 기본 OpenCV 폰트로 영문 텍스트를 그립니다.
        return frame # 텍스트가 추가된 프레임을 반환합니다.

def adjust_lighting(frame):
    """조명 환경 개선을 위한 CLAHE 적용"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB) # BGR 이미지를 LAB 색상 공간으로 변환합니다.
    l, a, b = cv2.split(lab) # LAB 채널을 L(명도), A, B로 분리합니다.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) # CLAHE 알고리즘 객체를 생성합니다.
    cl = clahe.apply(l) # L 채널에 CLAHE를 적용하여 명암을 보정합니다.
    limg = cv2.merge((cl, a, b)) # 보정된 L 채널과 기존 A, B 채널을 다시 병합합니다.
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR) # LAB 이미지를 다시 BGR로 변환하여 반환합니다.
    
def remove_noise(frame):
    """노이즈 제거를 위한 미디언 필터 적용"""
    return cv2.medianBlur(frame, 5) # 이미지에 미디언 블러 필터를 적용하여 노이즈를 제거합니다.

def is_valid_url(url):
    """문자열이 유효한 URL 형식인지 간단하게 확인합니다."""
    # http:// 또는 https:// 로 시작하는지 확인합니다.
    return url.startswith("http://") or url.startswith("https://")

def main():
    """
    메인 함수: 웹캠을 통해 QR 코드를 인식하고 유효한 URL이면 엽니다.
    """
    cap = cv2.VideoCapture(0) # 웹캠(0번 카메라)을 엽니다.
    if not cap.isOpened(): # 웹캠이 성공적으로 열렸는지 확인합니다.
        print("웹캠을 열 수 없습니다.") # 실패 시 오류 메시지를 출력합니다.
        return # 프로그램 종료합니다.

    detector = cv2.QRCodeDetector() # QR 코드 탐지 및 디코딩을 위한 객체를 생성합니다.
    
    print("--- QR 코드 인식 준비 완료. 카메라에 QR 코드를 비추세요. ---")
    
    while True: # 무한 루프를 실행하여 실시간으로 프레임을 처리합니다.
        ret, frame = cap.read() # 웹캠에서 프레임을 읽습니다.
        if not ret: # 프레임을 읽는 데 실패하면
            print("프레임을 읽을 수 없습니다.")
            break # 루프를 종료합니다.
        
        # 노이즈 제거 및 조명 보정 적용
        processed_frame = remove_noise(frame.copy()) # 프레임에 노이즈 제거를 적용합니다.
        processed_frame = adjust_lighting(processed_frame) # 노이즈 제거된 프레임에 조명 보정을 적용합니다.
        
        # QR 코드 디텍터로 바운딩 박스 찾아내기
        _, points, _ = detector.detectAndDecode(processed_frame) # 전처리된 프레임에서 QR 코드의 위치를 찾습니다.

        if points is not None and points.size > 0: # QR 코드의 위치 정보(points)가 유효한지 확인합니다.
            # 바운딩 박스 그리기
            cv2.polylines(frame, points.astype(int), True, (0, 255, 255), 3) # 원본 프레임에 QR 코드의 윤곽선을 그립니다.

            # 바운딩 박스 영역만 잘라서 디코딩 시도
            x, y, w, h = cv2.boundingRect(points.astype(int)) # QR 코드 위치의 사각형 경계(x, y, 너비, 높이)를 가져옵니다.
            
            # ROI(관심 영역) 좌표가 유효한지 안전하게 확인
            y_safe = max(0, y) # y 좌표가 0보다 작아지지 않도록 보정합니다.
            x_safe = max(0, x) # x 좌표가 0보다 작아지지 않도록 보정합니다.
            y2_safe = min(processed_frame.shape[0], y + h) # y + h 값이 이미지 높이를 넘지 않도록 보정합니다.
            x2_safe = min(processed_frame.shape[1], x + w) # x + w 값이 이미지 너비를 넘지 않도록 보정합니다.
            
            if (x2_safe - x_safe) > 0 and (y2_safe - y_safe) > 0: # 잘라낼 영역의 너비와 높이가 0보다 큰지 확인합니다.
                roi_to_decode = processed_frame[y_safe:y2_safe, x_safe:x2_safe] # 전처리된 프레임에서 ROI를 잘라냅니다.
                data, _, _ = detector.detectAndDecode(roi_to_decode) # 잘라낸 ROI에서 QR 코드를 디코딩합니다.
        
        else: # QR 코드가 프레임에서 감지되지 않았을 경우
            data = None # 데이터를 None으로 초기화합니다.
        
        if data: # QR 코드 데이터가 존재하면
            if is_valid_url(data): # 데이터가 유효한 URL 형식인지 확인합니다.
                print(f"QR 코드 인식 성공! 링크를 엽니다: {data}")
                webbrowser.open(data) # 기본 웹 브라우저로 링크를 엽니다.
                break # 링크를 열었으므로 루프를 종료하고 프로그램을 끝냅니다.
            else:
                display_frame = put_text_on_frame(frame, f"QR 코드 인식됨: {data}", (10, 60), (0, 255, 0))
                display_frame = put_text_on_frame(display_frame, "하지만 유효한 URL이 아닙니다.", (10, 90), (0, 0, 255))
        else:
            display_frame = put_text_on_frame(frame, "QR 코드 찾는 중...", (10, 30), (255, 0, 0))

        cv2.imshow("QR 코드 인식", display_frame) # 화면에 프레임을 보여줍니다.
        
        if cv2.waitKey(1) & 0xFF == ord('q'): # 'q' 키를 누르면
            break # 루프를 종료하고 프로그램을 끝냅니다.
            
    cap.release() # 웹캠 객체를 해제합니다.
    cv2.destroyAllWindows() # 모든 창을 닫습니다.

if __name__ == "__main__":
    main() # main 함수를 실행합니다.
