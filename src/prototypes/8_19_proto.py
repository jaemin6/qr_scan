# 필요한 라이브러리 가져오기
import cv2 # OpenCV 라이브러리: 이미지 및 비디오 처리에 사용됩니다.
import time # 시간 관련 라이브러리: 성능 측정을 위해 사용됩니다.
import numpy as np # NumPy 라이브러리: 숫자 연산을 효율적으로 처리합니다.
import matplotlib # Matplotlib 라이브러리: 그래프를 그리는 데 사용됩니다.
import matplotlib.pyplot as plt # Matplotlib의 하위 모듈: 그래프를 쉽게 만듭니다.
from matplotlib.backends.backend_agg import FigureCanvasAgg # 그래프를 이미지로 변환하기 위한 모듈입니다.
from PIL import ImageFont, ImageDraw, Image # Pillow 라이브러리: 이미지에 한글을 쓰기 위해 사용됩니다.
import webbrowser # 웹 브라우저를 제어하는 라이브러리: 링크를 여는 데 사용됩니다.

# matplotlib 백엔드를 'Agg'로 명시적으로 설정하여 이미지 변환을 안정적으로 만듦
matplotlib.use('Agg') # GUI 없이 그래프를 이미지 파일로 저장하도록 설정합니다.

# 한글 폰트 설정 (matplotlib용)
try:
    plt.rcParams['font.family'] = 'Malgun Gothic' # Windows에서 한글 폰트를 '맑은 고딕'으로 설정합니다.
    plt.rcParams['axes.unicode_minus'] = False # 그래프에서 마이너스 기호가 깨지는 현상을 방지합니다.
except Exception:
    pass # 폰트 설정이 실패하면 에러 없이 건너뜁니다.

# OpenCV용 한글 폰트 경로 지정 (Windows 기준: 맑은 고딕)
try:
    fontpath = "C:/Windows/Fonts/malgun.ttf" # 맑은 고딕 폰트 파일의 경로를 지정합니다.
    font = ImageFont.truetype(fontpath, 20) # PIL(Pillow)에서 사용할 폰트 객체를 생성합니다.
    is_font_loaded = True # 폰트가 성공적으로 로드되었음을 나타내는 플래그를 True로 설정합니다.
except IOError:
    is_font_loaded = False # 폰트 로드에 실패하면 플래그를 False로 설정합니다.
    font = None # 폰트 객체를 None으로 초기화합니다.

def put_text_on_frame(frame, text, pos, color=(255, 0, 0)):
    """한글 텍스트를 프레임에 표시하는 유틸리티 함수"""
    if is_font_loaded: # 한글 폰트가 로드되었는지 확인합니다.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # OpenCV의 BGR 색상 채널을 RGB로 변환합니다.
        pil_img = Image.fromarray(frame_rgb) # NumPy 배열을 PIL 이미지 객체로 변환합니다.
        draw = ImageDraw.Draw(pil_img) # 이미지에 그림을 그릴 수 있는 객체를 생성합니다.
        draw.text(pos, text, font=font, fill=color) # 지정된 위치에 한글 텍스트를 그립니다.
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR) # PIL 이미지를 다시 OpenCV의 BGR 형식으로 변환합니다.
    else: # 한글 폰트가 로드되지 않았을 경우
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2) # 기본 OpenCV 폰트로 영문 텍스트를 그립니다.
        return frame # 텍스트가 추가된 프레임을 반환합니다.

def create_graph_image(baseline_rate, improved_rate, fps_base, fps_improved):
    """
    matplotlib을 사용하여 최종 성능 그래프를 생성합니다.
    """
    plt.style.use('dark_background') # 그래프 배경을 어둡게 설정하여 가독성을 높입니다.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), tight_layout=True) # 1행 2열의 서브플롯을 생성합니다.
    
    # 인식률 그래프
    labels_rate = ['기본', '향상된 방법'] # 인식률 그래프의 x축 라벨을 정의합니다.
    rates = [baseline_rate, improved_rate] # 인식률 데이터를 준비합니다.
    bars1 = ax1.bar(labels_rate, rates, color=['#ff9999', '#66b3ff']) # 막대 그래프를 그립니다.
    ax1.set_title('인식 성공률 (%)') # 그래프의 제목을 설정합니다.
    ax1.set_ylim(0, 100) # y축 범위를 0부터 100까지로 설정합니다.
    for bar in bars1: # 각 막대에 대해 반복합니다.
        yval = bar.get_height() # 막대의 높이(값)를 가져옵니다.
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', va='bottom') # 막대 위에 값을 표시합니다.
    
    # FPS 그래프
    labels_fps = ['기본', '향상된 방법'] # FPS 그래프의 x축 라벨을 정의합니다.
    fps_values = [fps_base, fps_improved] # FPS 데이터를 준비합니다.
    bars2 = ax2.bar(labels_fps, fps_values, color=['#ff9999', '#66b3ff']) # 막대 그래프를 그립니다.
    ax2.set_title('평균 FPS') # 그래프의 제목을 설정합니다.
    ax2.set_ylim(0, max(fps_values) * 1.2 if max(fps_values) > 0 else 10) # y축 범위를 자동으로 설정합니다.
    for bar in bars2: # 각 막대에 대해 반복합니다.
        yval = bar.get_height() # 막대의 높이(값)를 가져옵니다.
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.1f}', ha='center', va='bottom') # 막대 위에 값을 표시합니다.
    
    # 그래프를 메모리 버퍼로 변환 (안정적인 방법)
    canvas = FigureCanvasAgg(fig) # FigureCanvasAgg 객체를 생성합니다.
    canvas.draw() # 그래프를 그립니다.
    
    buf = canvas.buffer_rgba() # 그래프의 픽셀 데이터를 버퍼로 가져옵니다.
    graph_img = np.asarray(buf, dtype=np.uint8) # 버퍼를 NumPy 배열로 변환합니다.
    graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGBA2BGR) # RGBa 색상 채널을 OpenCV의 BGR로 변환합니다.
    
    plt.close(fig) # 생성된 그래프 객체를 닫아 메모리를 해제합니다.
    return graph_img # 그래프 이미지를 반환합니다.

def adjust_lighting(frame):
    """조명 환경 개선을 위한 CLAHE 적용"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB) # BGR 이미지를 LAB 색상 공간으로 변환합니다.
    l, a, b = cv2.split(lab) # LAB 채널을 L(명도), A, B로 분리합니다.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) # CLAHE 알고리즘 객체를 생성합니다.
    cl = clahe.apply(l) # L 채널에 CLAHE를 적용하여 명암을 보정합니다.
    limg = cv2.merge((cl, a, b)) # 보정된 L 채널과 기존 A, B 채널을 다시 병합합니다.
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR) # LAB 이미지를 다시 BGR로 변환하여 반환합니다.

def sharpen_image(frame):
    """이미지 선명화를 위한 샤프닝 필터 적용"""
    # 샤프닝 커널(필터)을 정의합니다.
    # 이 커널은 중앙 픽셀의 값을 강조하여 주변 픽셀과의 차이를 크게 만듭니다.
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    # `cv2.filter2D` 함수를 사용하여 커널을 이미지에 적용합니다.
    sharpened = cv2.filter2D(frame, -1, kernel)
    return sharpened

def remove_noise(frame):
    """노이즈 제거를 위한 미디언 필터 적용"""
    return cv2.medianBlur(frame, 5) # 이미지에 미디언 블러 필터를 적용하여 노이즈를 제거합니다.

def is_valid_url(url):
    """문자열이 유효한 URL 형식인지 간단하게 확인합니다."""
    # http:// 또는 https:// 로 시작하는지 확인합니다.
    return url.startswith("http://") or url.startswith("https://")

def main():
    """
    메인 함수: 웹캠을 통해 두 가지 QR 인식 방식의 성능을 비교하고, 유효한 URL이면 엽니다.
    """
    cap = cv2.VideoCapture(0) # 웹캠(0번 카메라)을 엽니다.
    if not cap.isOpened(): # 웹캠이 성공적으로 열렸는지 확인합니다.
        print("웹캠을 열 수 없습니다.") # 실패 시 오류 메시지를 출력합니다.
        return # 프로그램 종료합니다.

    detector = cv2.QRCodeDetector() # QR 코드 탐지 및 디코딩을 위한 객체를 생성합니다.
    
    # ----------------------------------------------------
    # 1단계: 성능 비교 테스트
    # ----------------------------------------------------
    
    # 기본 인식 방법 성능 측정
    print("--- 1단계: 기본 인식 방법 성능 측정 시작 (10초) ---") # 테스트 시작 메시지를 출력합니다.
    start_time_base = time.time() # 현재 시간을 기록하여 테스트 시간을 측정합니다.
    frame_count_base = 0 # 처리된 프레임 수를 0으로 초기화합니다.
    success_count_base = 0 # 인식에 성공한 QR 코드 수를 0으로 초기화합니다.
    
    while time.time() - start_time_base < 10: # 10초 동안 반복합니다.
        ret, frame = cap.read() # 웹캠에서 프레임을 읽습니다.
        if not ret: break # 프레임을 읽는 데 실패하면 루프를 종료합니다.
        
        frame_count_base += 1 # 프레임 수를 1 증가시킵니다.
        
        # 기본 인식 방법: 전처리 없이 바로 인식
        data, _, _ = detector.detectAndDecode(frame) # 전처리 없이 프레임 전체에서 QR 코드를 인식합니다.
        if data: # QR 코드 데이터가 존재하면
            success_count_base += 1 # 성공 횟수를 1 증가시킵니다.
        
        # 프레임에 메시지 표시
        display_frame = put_text_on_frame(frame.copy(), "기본 방법 테스트 중", (10, 30), (255, 0, 0)) # 프레임에 테스트 중임을 표시합니다.
        if data: # QR 코드 데이터가 존재하면
            display_frame = put_text_on_frame(display_frame, f"QR 인식됨: {data}", (10, 60), (0, 255, 0)) # 인식된 데이터를 표시합니다.
        
        cv2.imshow("QR 성능 테스트", display_frame) # 화면에 프레임을 보여줍니다.
        if cv2.waitKey(1) & 0xFF == ord('q'): break # 'q' 키를 누르면 루프를 종료합니다.
        
    cv2.destroyAllWindows() # 모든 창을 닫습니다.
    
    elapsed_base = time.time() - start_time_base # 경과 시간을 계산합니다.
    fps_base = frame_count_base / elapsed_base if elapsed_base > 0 else 0 # 초당 프레임 수(FPS)를 계산합니다.
    success_rate_base = success_count_base / frame_count_base * 100 if frame_count_base > 0 else 0 # 인식 성공률을 계산합니다.
    
    # 향상된 방법 성능 측정
    print("--- 2단계: 향상된 방법 테스트 시작 (10초) ---") # 테스트 시작 메시지를 출력합니다.
    start_time_improved = time.time() # 현재 시간을 기록하여 테스트 시간을 측정합니다.
    frame_count_improved = 0 # 처리된 프레임 수를 0으로 초기화합니다.
    success_count_improved = 0 # 성공 횟수를 0으로 초기화합니다.
    
    while time.time() - start_time_improved < 10: # 10초 동안 반복합니다.
        ret, frame = cap.read() # 웹캠에서 프레임을 읽습니다.
        if not ret: break # 프레임을 읽는 데 실패하면 루프를 종료합니다.
        
        frame_count_improved += 1 # 프레임 수를 1 증가시킵니다.
        
        # 노이즈 제거 및 조명 보정 적용
        processed_frame = remove_noise(frame.copy()) # 프레임에 노이즈 제거를 적용합니다.
        processed_frame = adjust_lighting(processed_frame) # 노이즈 제거된 프레임에 조명 보정을 적용합니다.
        processed_frame = sharpen_image(processed_frame) # 초점 개선을 위한 선명화 적용

        # QR 코드 디텍터로 바운딩 박스 찾아내기
        _, points, _ = detector.detectAndDecode(processed_frame) # 전처리된 프레임에서 QR 코드의 위치를 찾습니다.

        if points is not None and points.size > 0: # QR 코드의 위치 정보(points)가 유효한지 확인합니다.
            cv2.polylines(frame, points.astype(int), True, (0, 255, 255), 3) # 원본 프레임에 QR 코드의 윤곽선을 그립니다.

            x, y, w, h = cv2.boundingRect(points.astype(int)) # QR 코드 위치의 사각형 경계(x, y, 너비, 높이)를 가져옵니다.
            
            y_safe = max(0, y) # y 좌표가 0보다 작아지지 않도록 보정합니다.
            x_safe = max(0, x) # x 좌표가 0보다 작아지지 않도록 보정합니다.
            y2_safe = min(processed_frame.shape[0], y + h) # y + h 값이 이미지 높이를 넘지 않도록 보정합니다.
            x2_safe = min(processed_frame.shape[1], x + w) # x + w 값이 이미지 너비를 넘지 않도록 보정합니다.
            
            if (x2_safe - x_safe) > 0 and (y2_safe - y_safe) > 0: # 잘라낼 영역의 너비와 높이가 0보다 큰지 확인합니다.
                roi_to_decode = processed_frame[y_safe:y2_safe, x_safe:x2_safe] # 전처리된 프레임에서 ROI를 잘라냅니다.
                data, _, _ = detector.detectAndDecode(roi_to_decode) # 잘라낸 ROI에서 QR 코드를 디코딩합니다.
                if data: # QR 코드 데이터가 존재하면
                    success_count_improved += 1 # 성공 횟수를 1 증가시킵니다.
        
        display_frame = put_text_on_frame(frame, "향상된 방법 테스트 중", (10, 30), (0, 255, 0)) # 프레임에 테스트 중임을 표시합니다.
        cv2.imshow("QR 성능 테스트", display_frame) # 화면에 프레임을 보여줍니다.
        if cv2.waitKey(1) & 0xFF == ord('q'): break # 'q' 키를 누르면 루프를 종료합니다.
    
    cv2.destroyAllWindows() # 모든 창을 닫습니다.

    elapsed_improved = time.time() - start_time_improved # 경과 시간을 계산합니다.
    fps_improved = frame_count_improved / elapsed_improved if elapsed_improved > 0 else 0 # 초당 프레임 수(FPS)를 계산합니다.
    success_rate_improved = success_count_improved / frame_count_improved * 100 if frame_count_improved > 0 else 0 # 인식 성공률을 계산합니다.

    # 최종 결과 표시
    final_graph = create_graph_image(success_rate_base, success_rate_improved, fps_base, fps_improved) # 최종 결과 그래프 이미지를 생성합니다.
    cv2.imshow("QR 성능 최종 결과", final_graph) # 그래프를 화면에 보여줍니다.
    print("\n--- 분석 완료. 아무 키나 누르면 다음 단계로 넘어갑니다. ---")
    cv2.waitKey(0) # 아무 키나 누를 때까지 대기합니다.
    cv2.destroyAllWindows() # 모든 창을 닫습니다.

    # ----------------------------------------------------
    # 2단계: QR 코드 인식 및 링크 열기
    # ----------------------------------------------------
    print("--- QR 코드 인식 준비 완료. 카메라에 QR 코드를 비추세요. ---")
    
    while True: # 무한 루프를 실행하여 실시간으로 프레임을 처리합니다.
        ret, frame = cap.read() # 웹캠에서 프레임을 읽습니다.
        if not ret: break # 프레임을 읽는 데 실패하면 루프를 종료합니다.

        data = None # 디코딩 결과를 저장할 변수 초기화
        points = None

        # 1. 원본 프레임에서 인식 시도
        data, points, _ = detector.detectAndDecode(frame)

        if not data:
            # 2. 실패하면 전처리된 프레임으로 인식 시도
            processed_frame = remove_noise(frame.copy())
            processed_frame = adjust_lighting(processed_frame)
            processed_frame = sharpen_image(processed_frame)
            data, points, _ = detector.detectAndDecode(processed_frame)

        # 결과에 따른 UI 업데이트
        display_frame = frame.copy()
        if points is not None and points.size > 0:
            cv2.polylines(display_frame, points.astype(int), True, (0, 255, 255), 3)

        if data:
            if is_valid_url(data):
                display_frame = put_text_on_frame(display_frame, f"QR 코드 인식 성공! 링크를 엽니다...", (10, 60), (0, 255, 0))
                cv2.imshow("QR 코드 인식", display_frame)
                cv2.waitKey(1)
                print(f"QR 코드 인식 성공! 링크를 엽니다: {data}")
                webbrowser.open(data)
                break
            else:
                display_frame = put_text_on_frame(display_frame, f"QR 코드 인식됨: {data}", (10, 60), (0, 255, 0))
                display_frame = put_text_on_frame(display_frame, "하지만 유효한 URL이 아닙니다.", (10, 90), (0, 0, 255))
        else:
            display_frame = put_text_on_frame(display_frame, "QR 코드 찾는 중...", (10, 30), (255, 0, 0))
            display_frame = put_text_on_frame(display_frame, "초점을 맞추거나 QR코드를 조금 더 가까이/멀리 가져가세요", (10, display_frame.shape[0] - 30), (0, 255, 255))
            
        cv2.imshow("QR 코드 인식", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
