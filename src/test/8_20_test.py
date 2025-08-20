# 필요한 라이브러리를 가져옵니다.
import cv2 # OpenCV 라이브러리: 이미지 및 비디오 처리에 사용됩니다.
import webbrowser # 웹 브라우저를 제어하는 라이브러리: QR 코드 링크를 자동으로 엽니다.
import time # 시간 관련 라이브러리: QR 코드 중복 인식을 방지하기 위한 딜레이를 처리합니다.
import numpy as np # NumPy 라이브러리: 배열 및 행렬 연산을 효율적으로 처리합니다.
from PIL import ImageFont, ImageDraw, Image # Pillow 라이브러리: 이미지에 한글을 쓰기 위해 사용됩니다.

# 한글 폰트가 시스템에 있는지 확인하고, 없으면 기본 폰트로 대체합니다.
try:
    fontpath = "C:/Windows/Fonts/malgun.ttf" # Windows의 기본 한글 폰트 경로입니다.
    font = ImageFont.truetype(fontpath, 20) # PIL(Pillow)에서 사용할 폰트 객체를 생성합니다.
    is_font_loaded = True # 폰트가 성공적으로 로드되었음을 나타내는 플래그입니다.
except IOError:
    is_font_loaded = False # 폰트 로드에 실패하면 플래그를 False로 설정합니다.
    font = None # 폰트 객체를 None으로 초기화합니다.

def put_text_on_frame(frame, text, pos, color=(255, 0, 0)):
    """한글 텍스트를 프레임에 표시하는 유틸리티 함수"""
    if is_font_loaded: # 한글 폰트가 로드되었는지 확인합니다.
        # OpenCV의 BGR 색상 채널을 PIL의 RGB로 변환합니다.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # NumPy 배열을 PIL 이미지 객체로 변환합니다.
        pil_img = Image.fromarray(frame_rgb)
        # 이미지에 그릴 수 있는 객체를 생성합니다.
        draw = ImageDraw.Draw(pil_img)
        # 지정된 위치에 한글 텍스트를 그립니다.
        draw.text(pos, text, font=font, fill=color)
        # PIL 이미지를 다시 OpenCV의 BGR 형식으로 변환하여 반환합니다.
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else: # 한글 폰트가 로드되지 않았을 경우
        # 기본 OpenCV 폰트로 영문 텍스트를 그립니다.
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame

def enhance_image_for_qr(frame):
    """
    QR 코드 인식을 향상시키기 위해 컨투어 기반의 이미지 전처리를 수행합니다.
    """
    # 1. 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. 노이즈 제거: 미디언 블러를 적용하여 영상의 잡음을 줄입니다.
    blurred = cv2.medianBlur(gray, 5)

    # 3. 조명 보정: CLAHE(대비 제한 적응 히스토그램 평활화)를 적용하여 대비를 향상시킵니다.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)

    # 4. 이진화: QR 코드를 흑백으로 명확하게 구분합니다.
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. 컨투어 찾기: 이미지에서 객체의 외곽선을 찾습니다.
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 6. QR 코드 영역 추출: 가장 큰 사각형 형태의 컨투어를 찾습니다.
    largest_contour = None
    max_area = 0

    for contour in contours:
        perimeter = cv2.arcLength(contour, True) # 컨투어의 둘레를 계산합니다.
        # 다각형 근사화: 컨투어를 근사한 다각형으로 만듭니다.
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True) 
        area = cv2.contourArea(contour)

        # 사각형 형태의 컨투어이면서, 일정 크기 이상일 경우
        if len(approx) == 4 and area > 1000:
            if area > max_area:
                max_area = area
                largest_contour = approx

    # 가장 큰 사각형 컨투어를 찾으면 해당 영역을 잘라내어 반환합니다.
    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)
        # QR 코드 영역을 잘라냅니다.
        cropped_qr = frame[y:y+h, x:x+w]
        # 잘라낸 영역의 크기를 일정하게 조정하여 인식률을 높일 수 있습니다.
        if cropped_qr.shape[0] > 0 and cropped_qr.shape[1] > 0:
            return cropped_qr
    
    # QR 코드 영역을 찾지 못하면 원본 그레이스케일 이미지를 반환합니다.
    return gray

def main():
    """
    메인 함수: 웹캠을 통해 QR 코드를 인식하고 자동으로 링크를 엽니다.
    """
    cap = cv2.VideoCapture(0) # 0번 카메라(기본 웹캠)를 엽니다.
    if not cap.isOpened(): # 웹캠이 제대로 열렸는지 확인합니다.
        print("웹캠을 열 수 없습니다.")
        return

    detector = cv2.QRCodeDetector() # QR 코드 탐지 및 디코딩을 위한 객체를 생성합니다.
    print("QR 코드 스캐너 시작 ('q'를 눌러 종료)")

    last_data = None # 마지막으로 인식된 데이터를 저장하여 중복 인식을 방지합니다.
    last_open_time = 0 # 마지막으로 링크를 연 시간을 기록합니다.
    DELAY_TIME = 5 # 같은 QR 코드를 다시 열기까지의 최소 대기 시간(초)을 설정합니다.

    while True: # 무한 루프를 실행하여 실시간으로 프레임을 처리합니다.
        ret, frame = cap.read() # 웹캠에서 한 프레임을 읽어옵니다.
        if not ret: # 프레임을 읽는 데 실패하면 루프를 종료합니다.
            break
        
        # 1. 원본 프레임에서 인식 시도: 가장 빠르고 효율적인 방법입니다.
        data, points, _ = detector.detectAndDecode(frame)
        
        # 2. 실패하면 컨투어 기반으로 전처리된 프레임으로 재시도
        if not data:
            enhanced_frame = enhance_image_for_qr(frame.copy())
            # 디코더는 BGR 또는 그레이스케일 프레임을 받으므로, 바로 적용합니다.
            data, points, _ = detector.detectAndDecode(enhanced_frame)
            
        # 결과에 따른 화면 표시 및 링크 연결 로직
        display_frame = frame.copy()
        display_msg = "QR 코드를 찾는 중입니다..."
        color = (255, 0, 0) # 파란색

        if data:
            is_same_qr = (data == last_data)
            is_delay_passed = (time.time() - last_open_time) > DELAY_TIME
            
            if not is_same_qr or is_delay_passed:
                display_msg = "QR 코드 인식 성공! 링크를 엽니다."
                color = (0, 255, 0) # 초록색
                print(f"QR 코드 인식 성공: {data}")
                
                # URL 자동 보정 (http 또는 https가 없으면 추가)
                url = data.strip()
                if not url.startswith("http"):
                    url = "http://" + url
                webbrowser.open(url)
                
                # 마지막 인식 데이터와 시간 업데이트
                last_data = data
                last_open_time = time.time()
                
            else:
                display_msg = f"인식 완료: {data}\n딜레이({DELAY_TIME}s) 동안 대기 중..."
                color = (0, 255, 255) # 노란색
            
            # QR 코드 경계 상자 그리기
            # 컨투어 방식은 points가 없을 수도 있으므로 확인합니다.
            if points is not None and len(points) > 0:
                points = np.int32(points).reshape(-1, 2)
                cv2.polylines(display_frame, [points], True, color, 3)
        
        # 프레임에 텍스트 표시
        display_frame = put_text_on_frame(display_frame, display_msg, (10, 30), color)
        
        # 화면에 최종 결과 표시
        cv2.imshow("QR Code Scanner", display_frame)

        # 'q' 키를 누르면 프로그램 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
