# 필요한 라이브러리 가져오기
import cv2               # 컴퓨터 비전 작업을 위한 OpenCV 라이브러리
import time              # 시간 관련 작업을 위한 라이브러리: FPS 측정
import numpy as np       # 배열 및 행렬 연산을 위한 라이브러리
from PIL import ImageFont, ImageDraw, Image  # 파이썬 이미지 처리: 한글 폰트를 이미지에 그리기 위해

# 한글 폰트 경로 지정 (Windows 기준: 맑은 고딕)
try:
    fontpath = "C:/Windows/Fonts/malgun.ttf"
    font = ImageFont.truetype(fontpath, 20)
    is_font_loaded = True
except IOError:
    is_font_loaded = False
    font = None

def put_text_on_frame(frame, text, pos, color=(255, 0, 0)):
    """한글 텍스트를 프레임에 표시하는 유틸리티 함수"""
    if is_font_loaded:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_img)
        draw.text(pos, text, font=font, fill=color)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame

def main():
    """
    메인 함수: 웹캠을 통해 두 가지 QR 인식 방식의 성능을 실시간으로 비교합니다.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    detector = cv2.QRCodeDetector()
    
    # 1단계: 기본 인식 방법 성능 측정
    print("--- 1단계: 기본 인식 방법 성능 측정 (10초) ---")
    
    start_time = time.time()
    frame_count = 0
    success_count = 0
    
    while time.time() - start_time < 10:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 기본 인식 방법: 전처리 없이 바로 인식
        data, _, _ = detector.detectAndDecode(frame)
        if data:
            success_count += 1

        # 화면에 정보 표시
        fps = frame_count / (time.time() - start_time)
        display_msg = "기본 방법 테스트 중"
        color = (255, 0, 0)
        
        frame = put_text_on_frame(frame, f"FPS: {fps:.2f}", (10, 30), (0, 255, 255))
        frame = put_text_on_frame(frame, display_msg, (10, 60), color)
        
        if data:
            frame = put_text_on_frame(frame, f"QR 인식됨: {data}", (10, 90), (0, 255, 0))
            
        cv2.imshow("QR 성능 테스트", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    total_fps_base = frame_count / (time.time() - start_time)
    success_rate_base = success_count / frame_count * 100 if frame_count > 0 else 0
    
    print(f"\n--- 기본 방법 결과 ---")
    print(f"평균 FPS: {total_fps_base:.2f}")
    print(f"인식 성공률: {success_rate_base:.2f}%")
    
    # 2단계: 개선된 인식 방법 성능 측정
    print("\n--- 2단계: 개선된 인식 방법 성능 측정 (10초) ---")
    
    start_time = time.time()
    frame_count = 0
    success_count = 0
    
    while time.time() - start_time < 10:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1

        # 개선된 인식 방법: 전처리 후 인식
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binarized = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        data, _, _ = detector.detectAndDecode(binarized)
        
        if data:
            success_count += 1
        
        # 화면에 정보 표시
        fps = frame_count / (time.time() - start_time)
        display_msg = "개선된 방법 테스트 중"
        color = (0, 255, 0)
        
        frame = put_text_on_frame(frame, f"FPS: {fps:.2f}", (10, 30), (0, 255, 255))
        frame = put_text_on_frame(frame, display_msg, (10, 60), color)

        if data:
            frame = put_text_on_frame(frame, f"QR 인식됨: {data}", (10, 90), (0, 255, 0))
            
        cv2.imshow("QR 성능 테스트", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    total_fps_improved = frame_count / (time.time() - start_time)
    success_rate_improved = success_count / frame_count * 100 if frame_count > 0 else 0

    print(f"\n--- 개선된 방법 결과 ---")
    print(f"평균 FPS: {total_fps_improved:.2f}")
    print(f"인식 성공률: {success_rate_improved:.2f}%")

    print("\n--- 분석 완료: q를 눌러 종료하거나, 웹캠을 계속 사용하려면 놔두세요 ---")

    # 모든 테스트가 끝난 후, 개선된 방법으로 계속 실시간 스캔
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binarized = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        data, _, _ = detector.detectAndDecode(binarized)

        if data:
            frame = put_text_on_frame(frame, f"QR 인식됨: {data}", (10, 30), (0, 255, 0))
        else:
            frame = put_text_on_frame(frame, "QR 미인식", (10, 30), (255, 0, 0))
            
        cv2.imshow("QR 성능 테스트", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()