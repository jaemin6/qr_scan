# 필요한 라이브러리를 가져옵니다.
import cv2 # OpenCV 라이브러리: 이미지 및 비디오 처리에 사용됩니다.
import webbrowser # 웹 브라우저를 제어하는 라이브러리: QR 코드 링크를 자동으로 엽니다.
import time # 시간 관련 라이브러리: 시간 측정 및 딜레이를 처리합니다.
import numpy as np # NumPy 라이브러리: 배열 및 행렬 연산을 효율적으로 처리합니다.
import re # 정규 표현식(Regular Expression) 라이브러리: URL 유효성을 검사합니다.
from PIL import ImageFont, ImageDraw, Image # Pillow 라이브러리: 이미지에 한글을 쓰기 위해 사용됩니다.

# 한글 폰트가 시스템에 있는지 확인하고, 없으면 기본 폰트로 대체합니다.
try:
    fontpath = "C:/Windows/Fonts/malgun.ttf" # Windows의 기본 한글 폰트 경로입니다.
    font = ImageFont.truetype(fontpath, 20) # PIL(Pillow)에서 사용할 폰트 객체를 생성합니다.
    is_font_loaded = True # 폰트가 성공적으로 로드되었음을 나타내는 플래그입니다.
except IOError:
    is_font_loaded = False # 폰트 로드에 실패하면 플래그를 False로 설정합니다.
    font = None # 폰트 객체를 None으로 초기화합니다.

# 정규 표현식을 사용하여 URL의 유효성을 검사합니다.
# http 또는 https로 시작하는 문자열을 찾습니다.
url_pattern = re.compile(r'^(http|https)://[^\s/$.?#].[^\s]*$')

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

def show_statistics(total_frames, recognized_frames, recognition_times):
    """
    통계 정보를 별도의 창에 표시합니다.
    """
    # 통계 정보를 계산합니다.
    recognition_rate = (recognized_frames / total_frames) * 100 if total_frames > 0 else 0
    avg_recognition_time = np.mean(recognition_times) if len(recognition_times) > 0 else 0

    # 통계 메시지를 생성합니다.
    stats_msg1 = f"총 프레임 수: {total_frames}"
    stats_msg2 = f"QR 코드 인식 프레임 수: {recognized_frames}"
    stats_msg3 = f"인식률: {recognition_rate:.2f}%"
    stats_msg4 = f"평균 인식 시간: {avg_recognition_time*1000:.2f}ms"

    # 통계 정보를 담을 새로운 이미지(창)를 생성합니다.
    height, width = 300, 500
    stats_image = np.zeros((height, width, 3), dtype=np.uint8)
    stats_image.fill(255) # 흰색 배경

    # 한글 텍스트를 이미지에 표시합니다.
    stats_image = put_text_on_frame(stats_image, "--- 스캔 통계 ---", (100, 50), (0, 0, 0))
    stats_image = put_text_on_frame(stats_image, stats_msg1, (50, 100), (0, 0, 0))
    stats_image = put_text_on_frame(stats_image, stats_msg2, (50, 140), (0, 0, 0))
    stats_image = put_text_on_frame(stats_image, stats_msg3, (50, 180), (0, 0, 0))
    stats_image = put_text_on_frame(stats_image, stats_msg4, (50, 220), (0, 0, 0))
    
    cv2.imshow("Statistics", stats_image)
    cv2.waitKey(0) # 아무 키나 누를 때까지 창을 유지합니다.
    cv2.destroyAllWindows()


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

    total_frames = 0 # 총 프레임 수를 계산합니다.
    recognized_frames = 0 # QR 코드를 인식한 프레임 수를 계산합니다.
    recognition_times = [] # 인식에 걸린 시간을 저장합니다.

    while True: # 무한 루프를 실행하여 실시간으로 프레임을 처리합니다.
        ret, frame = cap.read() # 웹캠에서 한 프레임을 읽어옵니다.
        if not ret: # 프레임을 읽는 데 실패하면 루프를 종료합니다.
            break
        
        total_frames += 1
        
        start_time = time.time() # 인식 시작 시간을 기록합니다.
        
        display_frame = frame.copy()
        found_data = False
        data = None
        points = None
        
        # 1. 원본 프레임에서 인식 시도
        data, points, _ = detector.detectAndDecode(frame)
        if data:
            found_data = True
        
        # 2. 원본에서 인식 실패 시, 이미지 전처리를 통해 다시 인식 시도
        if not found_data:
            # 이미지를 흑백으로 변환하고 대비를 높입니다.
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            enhanced_frame = cv2.equalizeHist(gray_frame)
            
            data, points, _ = detector.detectAndDecode(enhanced_frame)
            if data:
                found_data = True
        
        # 결과에 따른 화면 표시 및 링크 연결 로직
        display_msg = "QR 코드를 찾는 중입니다..."
        color = (255, 0, 0) # 파란색
        
        if found_data:
            end_time = time.time() # 인식 종료 시간을 기록합니다.
            recognition_times.append(end_time - start_time)
            recognized_frames += 1
            
            is_same_qr = (data == last_data)
            is_delay_passed = (time.time() - last_open_time) > DELAY_TIME
            
            # URL 유효성 검사
            is_valid_url = url_pattern.match(data)
            
            if is_valid_url and (not is_same_qr or is_delay_passed):
                display_msg = "QR 코드 인식 성공! 링크를 엽니다."
                color = (0, 255, 0) # 초록색
                print(f"QR 코드 인식 성공: {data}")
                
                try:
                    webbrowser.open(data)
                except Exception as e:
                    print(f"URL 열기 실패: {e}")
                
                last_data = data
                last_open_time = time.time()
                
            elif is_valid_url:
                display_msg = f"인식 완료: {data}\n딜레이({DELAY_TIME}s) 동안 대기 중..."
                color = (0, 255, 255) # 노란색
            
            else:
                display_msg = f"유효한 URL이 아닙니다: {data}"
                color = (0, 0, 255) # 빨간색
            
            # QR 코드 경계 상자 그리기
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

        # CPU 사용량을 줄이고 비디오 프레임 속도를 조절하기 위해 1초 딜레이를 추가합니다.
        time.sleep(0.01)

    # 자원 해제 및 통계 표시
    cap.release()
    cv2.destroyAllWindows()
    show_statistics(total_frames, recognized_frames, recognition_times)

if __name__ == "__main__":
    main()
