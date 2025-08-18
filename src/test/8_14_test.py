"""
1. 캠에서 qr 코드를 인식하고, 인식된 qr 코드가 URL인 경우 해당 URL을 브라우저로 이동 할 수 있게하는 코드
2. 캠에서 qr을 인식하면 터미널에 인식 된 url주소가 뜸
3. 터미널에 인식된 주소를 ctrl + click하면 브라우저로 이동
4. q 키를 누르면 프로그램 종료
"""
# 필요한 라이브러리를 가져옴
import cv2          # 컴퓨터 비전 작업을 위한 OpenCV 라이브러리. 영상 처리, 웸캠 제어 등 사용
# import webbrowser   # 웹 브라우저를 제어하기 위한 라이브러리. URL을 열기 위해 사용(별도 설치 없이 사용 가능)

# 메인 함수 정의
def main():
    cap = cv2.VideoCapture(0)           # 0번 카메라(기본 웹캠)를 열기  (웹캠이 여러개라면 1, 2 등 숫자를 바꿔 시도)
    if not cap.isOpened():              # 카메라가 열리지 않으면 오류 메시지 출력, 웹캠이 제대로 열렸는지 확인
        print("웹캠을 열 수 없습니다.")
        return                          # 함수를 종료하여 프로그램을 중단

    # QR 코드 디텍터 객체 생성
    detector = cv2.QRCodeDetector()
    print("QR 코드 인식 시작 ('q'로 종료)")

    # 마지막으로 인식된 qr코드의 데이터를 저장할 변수, 이 변수를 사용하여 같은 qr코드가 반복 인식되는 것을 방지
    last_data = None

    # 무한 루프를 시작하여 웹캠에서 프레임을 읽고 QR 코드를 인식
    while True:
        # ret는 성공 여부 (True/False), frame은 읽은 프레임 이미지
        ret, frame = cap.read() # cap.read()는 웹캠에서 프레임을 읽어오는 함수
        if not ret: # 프레임을 읽지 못하면 루프를 종료
            break

        # data: 인식된 QR 코드의 데이터(없으면 빈 문자열) 
        # bbox: QR 코드의 경계 상자, _는 신뢰도 값(사용하지 않음)
        data, bbox, _ = detector.detectAndDecode(frame) # detector.detectAndDecode()는 QR 코드를 인식하고 데이터를 추출하는 함수

        # qr 코드가 성공적으로 인식되었을 경우
        if data:
            # 새로 인식된 데이터가 이전 데이터와 다를 경우에만 처리
            if data != last_data:  # 이 조건을 통해 같은 qr 코드가 반복 인식되는 것을 방지
                msg = f" QR 인식됨: {data}"
                print(msg)
                
                # 웹 링크인 경우 브라우저 열기 인식된 주소가 'http' 또는 'https'로 시작하는 경우
                # if data.startswith("http") or data.startswith("https"):
                   # webbrowser.open(data)  # 웹 브라우저로 URL 열기
                
                last_data = data # 현재 인식된 데이터를 저장
            # QR 코드의 경계 상자가 존재하면, 경계 상자를 그려서 시각적으로 표시 (녹색)
            if bbox is not None:
                # bbox는 QR 코드의 위치를 나타내는 좌표 배열
                pts = bbox.astype(int).reshape(-1, 2)
                # 경계 상자를 그리기 위해 네 개의 선을 차례대로 그림
                for i in range(len(pts)):
                    cv2.line(frame, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]), (0, 255, 0), 2)
            display_msg = "QR 인식 성공"
            color = (0, 255, 0)  # 녹색 표시

        else:
            # QR 코드가 인식되지 않은 경우
            display_msg = "X QR 미인식"
            color = (0, 0, 255) # 빨간색 표시
            last_data = None    # QR 코드가 사라지면 마지막 데이터를 초기화
        # QR 코드 인식 결과를 화면에 표시
        cv2.putText(frame, display_msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow("QR Code Scanner", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 루프가 끝나면 웹캠을 해제하고 모든 창을 닫음
    cap.release()
    # openCV에서 사용한 모든 창을 닫음
    cv2.destroyAllWindows()

# 프로그램이 직접 실행될 때만 main() 함수를 호출
if __name__ == "__main__":
    main()