import cv2
import webbrowser
import time

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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        data, bbox, _ = detector.detectAndDecode(frame)

        if data:
            # 새 데이터거나, 일정 시간 지난 경우만 처리
            if data != last_data or (time.time() - last_open_time) > DELAY_TIME:
                print(f"QR 코드 인식됨: {data}")

                if data.startswith("http"):
                    webbrowser.open(data)

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
            color = (0, 0, 255)
            last_data = None  # QR 사라지면 상태 초기화

        cv2.putText(frame, display_msg, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("QR Code Scanner", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
