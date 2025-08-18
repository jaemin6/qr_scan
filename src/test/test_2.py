import cv2
import webbrowser
import time

# 1. QRCodeDetector 생성
detector = cv2.QRCodeDetector()

# 2. 카메라 열기
cap = cv2.VideoCapture(0)

# 3. 한글 폰트 설정 (윈도우 기준: 말굿체)
font = cv2.FONT_HERSHEY_SIMPLEX  # 기본 폰트
# 참고: OpenCV 기본 폰트는 한글 미지원 → ??? 로 나옴
# 따라서 아래처럼 putText 대신 rectangle + imshow 에 직접 넣거나,
# cv2.putText로는 영문만 쓰고, 한글은 PIL 같은 라이브러리 써야 함.
# 여기서는 링크는 영문이므로 그대로 putText 사용.

last_open_time = 0  # 마지막으로 링크를 연 시각
delay_time = 10     # 10초 딜레이

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 4. QR 코드 인식
    data, bbox, _ = detector.detectAndDecode(frame)

    if data:
        # QR 영역 그리기
        if bbox is not None:
            pts = bbox.astype(int).reshape(-1, 2)
            for j in range(len(pts)):
                cv2.line(frame, tuple(pts[j]), tuple(pts[(j+1) % len(pts)]), (0, 255, 0), 2)

        # 화면에 표시
        cv2.putText(frame, f"QR: {data}", (20, 50),
                    font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

        # 일정 시간 지나야 다시 링크 열림
        current_time = time.time()
        if current_time - last_open_time > delay_time:
            print(f"QR 링크 열기: {data}")
            webbrowser.open(data)
            last_open_time = current_time  # 마지막 실행 시각 갱신

    cv2.imshow("QR Scanner", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
