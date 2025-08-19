# 필요한 라이브러리 가져오기
import cv2                 # 컴퓨터 비전 작업을 위한 OpenCV 라이브러리
import webbrowser          # 웹 브라우저를 제어하기 위한 라이브러리: qr 코드 인식 후 URL 열기
import time                # 시간 관련 작업을 위한 라이브러리: QR 코드 인식 후 딜레이 처리
import numpy as np         # 배열 및 행렬 연산을 위한 라이브러리: OpenCV 이미지와 PIL 이미지 변환에 사용
from PIL import ImageFont, ImageDraw, Image  # 파이썬 이미지 처리: 한글 폰트를 이미지에 그리기 위해

def main():
    """
    메인 함수: 웹캠을 통해 QR 코드를 실시간으로 스캔하고,
    인식된 정보를 화면에 표시하며 URL을 엽니다.
    """
    # 0번 카메라(기본 웹캠)를 열기
    cap = cv2.VideoCapture(0)
    # 웹캠이 제대로 열렸는지 확인
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    # QR 코드 디텍터 객체 생성 및 FPS 계산을 위한 변수 초기화
    detector = cv2.QRCodeDetector()
    print("QR 코드 인식 시작 ('q'로 종료)")

    # 성능 측정 변수
    start_time = time.time()
    frame_count = 0

    # 중복 인식 방지를 위한 변수
    last_data = None
    last_open_time = 0
    DELAY_TIME = 10 # 같은 QR 다시 열리기까지 최소 대기 시간(초)

    # 한글 폰트 경로 지정 (Windows 기준: 맑은 고딕)
    try:
        fontpath = "C:/Windows/Fonts/malgun.ttf"
        font = ImageFont.truetype(fontpath, 20)
        # 폰트 로드 성공 여부 플래그
        is_font_loaded = True
    except IOError:
        print("한글 폰트(malgun.ttf)를 찾을 수 없습니다. 기본 폰트로 대체합니다.")
        is_font_loaded = False
        font = None

    while True:
        # 웹캠에서 한 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            # 프레임을 읽지 못하면 루프 종료
            print("프레임을 읽을 수 없습니다. 프로그램을 종료합니다.")
            break

        # FPS 계산
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        # 이미지 전처리: 성능 향상
        # 이미지를 그레이스케일로 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 노이즈 감소를 위한 가우시안 블러
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # 이진화 처리
        _, binarized = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # QR 코드 인식 및 디코딩 (전처리된 이미지 사용)
        data, bbox, _ = detector.detectAndDecode(binarized)

        # 인식 결과에 따라 메시지 및 색상 설정
        display_msg = "QR 미인식"
        color = (255, 0, 0)  # 빨강
        
        # 인식 결과가 있을 경우
        if data:
            # URL 유효성 검사 및 웹 브라우저 열기 (안정성 개선)
            try:
                # 인식된 데이터가 이전에 열었던 데이터와 같은지 확인
                is_same_qr = (data == last_data)
                # 마지막으로 열었던 시간으로부터 딜레이 시간이 지났는지 확인
                is_delay_passed = (time.time() - last_open_time) > DELAY_TIME

                if not is_same_qr or is_delay_passed:
                    print(f"QR 코드 인식됨: {data}")
                    url = data.strip()
                    if not url.startswith(("http://", "https://")):
                        url = "http://" + url
                    webbrowser.open(url)

                    # 마지막 인식 데이터와 시간 업데이트
                    last_data = data
                    last_open_time = time.time()
                    
            except Exception as e:
                # URL 열기 실패 시 오류 메시지 출력 (안정성 개선)
                print(f"URL 열기 오류: {e}")
                
            # QR 경계 상자 그리기
            if bbox is not None:
                pts = bbox.astype(int).reshape(-1, 2)
                for i in range(len(pts)):
                    cv2.line(frame, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]), (0, 255, 0), 2)
            
            # 신뢰도(Confidence Score)는 OpenCV의 기본 QR 디텍터에서 직접 제공되지 않으므로,
            # 편의상 '신뢰도 100%'로 가정하고 메시지를 표시합니다.
            display_msg = f"QR 인식됨: {data}"
            color = (0, 255, 0) # 초록
        
        # 화면에 FPS, 인식 메시지 표시 (사용성 개선)
        # OpenCV → PIL 변환 (한글 폰트 출력용)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_img)

        # 폰트가 정상적으로 로드된 경우에만 텍스트 그리기
        if is_font_loaded:
            draw.text((10, 30), f"FPS: {fps:.2f}", font=font, fill=(0, 255, 255))
            draw.text((10, 60), display_msg, font=font, fill=color)
        else:
            # 폰트 로드 실패 시 OpenCV의 기본 폰트로 텍스트 그리기
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, display_msg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # PIL → OpenCV 변환 후 화면에 표시
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.imshow("QR Code Scanner", frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()