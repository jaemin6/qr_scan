# 필요한 라이브러리 가져오기
import cv2            # 컴퓨터 비전 작업을 위한 OpenCV 라이브러리
import webbrowser     # 웹 브라우저를 제어하기 위한 라이브러리: qr 코드 인식 후 URL 열기
import time           # 시간 관련 작업을 위한 라이브러리: QR 코드 인식 후 딜레이 처리
import numpy as np    # 배열 및 행렬 연산을 위한 라이브러리: OpenCV 이미지와 PIL 이미지 변환에 사용
from PIL import ImageFont, ImageDraw, Image   # 파이썬 이미지 처리: 한글 폰트를 이미지에 그리기 위해

def main():                           # 메인 함수 정의
    cap = cv2.VideoCapture(0)         # 0번 카메라(기본 웹캠)를 열기
    if not cap.isOpened():            # 웹캠이 제대로 열렸는지 확인, 열리지 않는다면
        print("웹캠을 열 수 없습니다.")  # 오류 메시지 출력
        return                        # 함수를 종료하여 프로그램을 중단

    # QR 코드 디텍터 객체 생성
    detector = cv2.QRCodeDetector()       # QR 코드 인식하고 디코딩하는 객체 생성
    print("QR 코드 인식 시작 ('q'로 종료)")  # QR 코드 인식 시작 메시지 출력

    last_data = None                      # 마지막으로 인식된 QR 코드 데이터를 저장할 변수(중복 인식 방지)
    last_open_time = 0                    # 마지막으로 링크를 연 시각 저장
    DELAY_TIME = 10  # 같은 QR 다시 열리기까지 최소 대기 시간(초)

    # 한글 폰트 경로 지정 (Windows 기준: 맑은 고딕)
    try:
        fontpath = "C:/Windows/Fonts/malgun.ttf"  # 윈도우 기본 한글 폰트 경로 지정
        font = ImageFont.truetype(fontpath, 20)   # 폰트 파일과 크기를 지정하여 폰트 생성
    except IOError:           # 폰트 파일을 찾을 수 없는 경우
        print("한글 폰트(malgun.ttf)를 찾을 수 없습니다. 기본 폰트로 대체합니다.") # 폰트 로드 실패 메시지 출력
        fontpath = None       # 폰트 경로를 초기화
        font = None           # 폰트 객체를 None으로 설정(기본 폰트 사용)

    while True:               # 무한 루프 시작 (프로그램 핵심 실행 부분)
        ret, frame = cap.read() # 웹캠에서 한 프레임 읽기(ret: 성공 여부, frame: 읽은 이미지)
        if not ret:             # 프레임을 읽지 못하면
            break               # 루프 종료
 
        # 이미지 전처리 단계 추가, 프레임을 그레이스케일로 변환 (QR 코드 인식에 최적화)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 프레임을 그레이스케일로 변환 (QR 코드 인식에 최적화)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # 가우시안 블러 적용 (노이즈 감소)
        _, binarized = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 이진화 처리
        data, bbox, _ = detector.detectAndDecode(binarized)  # QR 코드 인식 및 디코딩

        if not data:
            data, bbox, _ = detector.detectAndDecode(frame)  # 현재 프레임에서 qr인식 후 데이터 추출
        
        
        
        # QR 코드 인식 상태에 따라 메시지 및 색상 설정
        display_msg = "QR 미인식"
        color = (255, 0, 0) # 빨강

        # QR 코드가 인식된 경우
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
                last_data = data             # 현재 인식된 데이터를 이전 데이터로 저장
                last_open_time = time.time() # 현재 시각을 마지막 열림 시간으로 저장
            
            # QR 경계 상자 그리기
            if bbox is not None:
                pts = bbox.astype(int).reshape(-1, 2)
                for i in range(len(pts)):
                    cv2.line(frame, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]),
                             (0, 255, 0), 2)  # qr코드의 모서리를 선으로 연결, 녹색 사각형

            display_msg = f"QR 인식됨: {data}"
            color = (0, 255, 0) # 초록
        
        # OpenCV → PIL 변환 (한글 폰트 출력용)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # opencv 이미지를 pil 이미지로 변환
        pil_img = Image.fromarray(frame_rgb)                # numpy 배열을 PIL 이미지로 변환
        draw = ImageDraw.Draw(pil_img)                      # pil 이미지에 그리기 위한 객체 생성

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