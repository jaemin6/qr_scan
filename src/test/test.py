# 필요한 라이브러리를 가져옴
import cv2 # 컴퓨터 비전 작업을 위한 OpenCV 라이브러리
import webbrowser  # 웹 브라우저를 제어하기 위한 라이브러리

# 메인 함수 정의
def main():
    cap = cv2.VideoCapture(0) # 0번 카메라(기본 웹캠)를 열기
    if not cap.isOpened(): # 웹캠이 제대로 열렸는지 확인
        print("❌ 웹캠을 열 수 없습니다. 카메라 연결 상태를 확인해 주세요.")
        return # 함수를 종료

    detector = cv2.QRCodeDetector()
    print("QR 코드 인식 시작 ('q'로 종료)")

    last_data = None  # 마지막으로 인식된 QR 코드 데이터를 저장할 변수

    while True:
        ret, frame = cap.read() # 웹캠에서 한 프레임 읽기
        if not ret:  # 프레임을 읽지 못하면 루프 종료
            break

        data, bbox, _ = detector.detectAndDecode(frame)

        # QR 코드가 성공적으로 인식되었을 경우
        if data:
            # 새로 인식된 데이터가 이전과 다를 때만 처리
            if data != last_data:
                msg = f"QR 인식됨: {data}"
                print(msg)
                
                # 데이터가 URL이면 웹 브라우저 열기
                if data.startswith("http") or data.startswith("https"):
                    webbrowser.open(data)
                
                last_data = data # 현재 데이터를 마지막 데이터로 저장

            # QR 코드가 인식된 상태에서는 경계 박스와 메시지 색상을 초록색으로 설정
            display_msg = f"QR: {data}" # UI에 표시할 메시지
            color = (0, 255, 0)

            # QR 코드 경계 박스 그리기
            if bbox is not None:
                pts = bbox.astype(int).reshape(-1, 2)
                for i in range(len(pts)):
                    cv2.line(frame, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]), color, 2)

        # QR 코드가 인식되지 않았을 경우 (else 블록 추가)
        else:
            display_msg = "QR: 인식 대기 중..."
            color = (0, 0, 255) # 메시지 색상을 빨간색으로 설정
            last_data = None  # QR 코드가 사라지면 마지막 데이터를 초기화

        # 화면 하단에 메시지 표시
        # 배경에 검은색 박스를 그려 텍스트 가독성을 높입니다.
        text_bg_height = 40
        cv2.rectangle(frame, (0, frame.shape[0] - text_bg_height), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.putText(frame, display_msg, (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("QR Code Scanner", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()