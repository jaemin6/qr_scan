# 필요한 라이브러리 가져오기
import cv2
import webbrowser
import time
import numpy as np
from PIL import ImageFont, ImageDraw, Image

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    detector = cv2.QRCodeDetector()
    print("QR 코드 인식 시작 ('q'로 종료)")

    last_data = None
    last_open_time = 0
    DELAY_TIME = 10  # 같은 QR 다시 열리기까지 최소 대기 시간(초)

    # 한글 폰트 경로 지정 (Windows 기준: 맑은 고딕)
    try:
        fontpath = "C:/Windows/Fonts/malgun.ttf"
        font = ImageFont.truetype(fontpath, 20)
    except IOError:
        print("한글 폰트(malgun.ttf)를 찾을 수 없습니다. 기본 폰트로 대체합니다.")
        fontpath = None
        font = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        data, bbox, _ = detector.detectAndDecode(frame)
        
        # QR 코드 인식 상태에 따라 메시지 및 색상 설정
        display_msg = "QR 미인식"
        color = (255, 0, 0) # 빨강

        if data:
            # 인식된 데이터가 이전에 열었던 데이터와 같은지 확인
            is_same_qr = (data == last_data)
            # 마지막으로 열었던 시간으로부터 딜레이 시간이 지났는지 확인
            is_delay_passed = (time.time() - last_open_time) > DELAY_TIME

            # 새로운 QR 코드를 인식했거나(is_same_qr가 False),
            # 같은 QR 코드를 인식했더라도(is_same_qr가 True) 딜레이 시간이 지났다면 처리
            if not is_same_qr or is_delay_passed:
                print(f"QR 코드 인식됨: {data}")
                
                # URL 자동 보정 + 열기
                url = data.strip()
                if not url.startswith("http"):
                    url = "http://" + url
                webbrowser.open(url)
                
                # 마지막 인식 데이터와 시간 업데이트
                last_data = data
                last_open_time = time.time()
            
            # QR 경계 상자 그리기
            if bbox is not None:
                pts = bbox.astype(int).reshape(-1, 2)
                for i in range(len(pts)):
                    cv2.line(frame, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]),
                             (0, 255, 0), 2)

            display_msg = f"QR 인식됨: {data}"
            color = (0, 255, 0) # 초록
        
        # OpenCV → PIL 변환 (한글 폰트 출력용)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_img)

        # 폰트가 정상적으로 로드된 경우에만 텍스트 그리기
        if font:
            draw.text((10, 30), display_msg, font=font, fill=color)
        else:
            # 폰트 로드 실패 시 OpenCV의 기본 폰트로 텍스트 그리기
            cv2.putText(frame, display_msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.imshow("QR Code Scanner", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()