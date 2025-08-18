import cv2
import webbrowser
import time
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

    #  한글 폰트 경로 지정 (Windows 기준: 맑은 고딕)
    fontpath = "C:/Windows/Fonts/malgun.ttf"
    font = ImageFont.truetype(fontpath, 20)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        data, bbox, _ = detector.detectAndDecode(frame)

        if data:
            # 새 데이터거나 일정 시간 지난 경우만 처리
            if data != last_data or (time.time() - last_open_time) > DELAY_TIME:
                print(f"QR 코드 인식됨: {data}")

                # URL 자동 보정 + 열기
                url = data.strip()
                if not url.startswith("http"):
                    url = "http://" + url
                webbrowser.open(url)

                last_data = data
                last_open_time = time.time()

            # QR 경계 상자 그리기
            if bbox is not None:
                pts = bbox.astype(int).reshape(-1, 2)
                for i in range(len(pts)):
                    cv2.line(frame, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]),
                             (0, 255, 0), 2)

            display_msg = f"QR 인식됨: {data}"
            color = (0, 255, 0)

        else:
            display_msg = "QR 미인식"
            color = (255, 0, 0)  # 빨강
            last_data = None

        #  OpenCV → PIL 변환 (한글 폰트 출력용)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_img)
        draw.text((10, 30), display_msg, font=font, fill=color)  # 한글 표시
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        cv2.imshow("QR Code Scanner", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import numpy as np  # PIL 변환을 위해 numpy 필요
    main()
