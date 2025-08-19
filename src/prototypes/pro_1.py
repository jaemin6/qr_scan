'''
1. 1단계 기본 웹캠으로 qr코드를 10초간 인식함
2. 2단계 향상된 방법으로 웹캠으로 qr코드를 10초간 인식함
3. 1단계와 2단계의 인식률과 FPS를 비교함
4. matplotlib을 사용하여 인식률과 FPS를 그래프로 시각화함
5. 최종 결과를 화면에 보여줌
'''

# 필요한 라이브러리 가져오기
import cv2                                                  # OpenCV 라이브러리: 이미지 및 비디오 처리에 사용
import time                                                 # 시간 관련 라이브러리: 성능 측정을 위해 사용
import numpy as np                                          # NumPy 라이브러리: 숫자 연산을 효율적으로 처리
import matplotlib                                           # Matplotlib 라이브러리: 그래프를 그리는 데 사용
import matplotlib.pyplot as plt                             # Matplotlib의 하위 모듈: 그래프를 만들기
from matplotlib.backends.backend_agg import FigureCanvasAgg # 그래프를 이미지로 변환하기 위한 모듈
from PIL import ImageFont, ImageDraw, Image                 # Pillow 라이브러리: 이미지에 한글을 쓰기 위해 사용

# matplotlib 백엔드를 'Agg'로 명시적으로 설정하여 이미지 변환을 안정적으로 만듦
matplotlib.use('Agg') # GUI 없이 그래프를 이미지 파일로 저장하도록 설정

# 한글 폰트 설정 (matplotlib용)
# 시스템에 설치된 폰트를 찾아 사용하거나, 경로를 직접 지정
try:
    plt.rcParams['font.family'] = 'Malgun Gothic' # Windows에서 한글 폰트를 '맑은 고딕'으로 설정
    plt.rcParams['axes.unicode_minus'] = False # 그래프에서 마이너스 기호가 깨지는 현상을 방지
except Exception:
    pass # 폰트 설정이 실패하면 에러 없이 건너뜀

# OpenCV용 한글 폰트 경로 지정 (Windows 기준: 맑은 고딕)
try:
    fontpath = "C:/Windows/Fonts/malgun.ttf" # 맑은 고딕 폰트 파일의 경로를 지정
    font = ImageFont.truetype(fontpath, 20) # PIL(Pillow)에서 사용할 폰트 객체를 생성
    is_font_loaded = True # 폰트가 성공적으로 로드되었음을 나타내는 플래그를 True로 설정
except IOError:
    is_font_loaded = False # 폰트 로드에 실패하면 플래그를 False로 설정
    font = None # 폰트 객체를 None으로 초기화

def put_text_on_frame(frame, text, pos, color=(255, 0, 0)):
    """한글 텍스트를 프레임에 표시하는 유틸리티 함수"""
    if is_font_loaded: 
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # OpenCV의 BGR 색상 채널을 RGB로 변환
        pil_img = Image.fromarray(frame_rgb) # NumPy 배열을 PIL 이미지 객체로 변환
        draw = ImageDraw.Draw(pil_img) # 이미지에 그림을 그릴 수 있는 객체를 생성
        draw.text(pos, text, font=font, fill=color) # 지정된 위치에 한글 텍스트를 그림
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR) # PIL 이미지를 다시 OpenCV의 BGR 형식으로 변환
    else: # 한글 폰트가 로드되지 않았을 경우 
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2) # 기본 OpenCV 폰트로 영문 텍스트를 그림
        return frame # 텍스트가 추가된 프레임을 반환

def create_graph_image(baseline_rate, improved_rate, fps_base, fps_improved):
    """
    matplotlib을 사용하여 최종 성능 그래프를 생성함
    """
    plt.style.use('dark_background') # 그래프 배경을 어둡게 설정
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), tight_layout=True) # 1행 2열의 서브플롯을 생성
    
    # 인식률 그래프
    labels_rate = ['기본', '향상된 방법'] # 인식률 그래프의 x축 라벨
    rates = [baseline_rate, improved_rate] # 인식률 데이터를 준비
    bars1 = ax1.bar(labels_rate, rates, color=['#ff9999', '#66b3ff']) # 막대 그래프를 그립니다.
    ax1.set_title('인식 성공률 (%)') # 그래프의 제목
    ax1.set_ylim(0, 100) # y축 범위를 0부터 100까지
    for bar in bars1: # 각 막대에 대해 반복
        yval = bar.get_height() # 막대의 높이(값)를 불ㄹㅓ옴
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', va='bottom') # 막대 위에 값을 표시
    
    # FPS 그래프
    labels_fps = ['기본', '향상된 방법'] # FPS 그래프의 x축 라벨
    fps_values = [fps_base, fps_improved] # FPS 데이터를 준비
    bars2 = ax2.bar(labels_fps, fps_values, color=['#ff9999', '#66b3ff']) # 막대 그래프를 그림
    ax2.set_title('평균 FPS') # 그래프의 제목
    ax2.set_ylim(0, max(fps_values) * 1.2 if max(fps_values) > 0 else 10) # y축 범위
    for bar in bars2: # 각 막대에 대해 반복
        yval = bar.get_height() # 막대의 높이값을 가져옴
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.1f}', ha='center', va='bottom') # 막대 위에 값을 표시
    
    # 그래프를 메모리 버퍼로 변환 (안정적인 방법)
    canvas = FigureCanvasAgg(fig) # FigureCanvasAgg 객체를 생성
    canvas.draw() # 그래프 그림
    
    buf = canvas.buffer_rgba() # 그래프의 픽셀 데이터를 버퍼로 가져옴
    graph_img = np.asarray(buf, dtype=np.uint8) # 버퍼를 NumPy 배열로 변환
    graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGBA2BGR) # RGBa 색상 채널을 OpenCV의 BGR로 변환
    
    plt.close(fig) # 생성된 그래프 객체를 닫아 메모리를 해제
    return graph_img # 그래프 이미지를 반환

def adjust_lighting(frame):
    """조명 환경 개선을 위한 CLAHE 적용"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB) # BGR 이미지를 LAB 색상 공간으로 변환
    l, a, b = cv2.split(lab) # LAB 채널을 L(명도), A, B로 분리
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) # CLAHE 알고리즘 객체를 생성
    cl = clahe.apply(l) # L 채널에 CLAHE를 적용하여 명암을 보정함
    limg = cv2.merge((cl, a, b)) # 보정된 L 채널과 기존 A, B 채널을 다시 병합
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR) # LAB 이미지를 다시 BGR로 변환하여 반환
    
def remove_noise(frame):
    """노이즈 제거를 위한 미디언 필터 적용"""
    return cv2.medianBlur(frame, 5) # 이미지에 블러 필터를 적용하여 노이즈를 제거

def main():
    """
    메인 함수: 웹캠을 통해 두 가지 QR 인식 방식의 성능을 실시간으로 비교하고 시각화
    """
    cap = cv2.VideoCapture(0) # 웹캠(0번 카메라)을 엶 기본카메라
    if not cap.isOpened(): # 웹캠이 성공적으로 열렸는지 확인
        print("웹캠을 열 수 없습니다.") # 실패 시 오류 메시지
        return # 프로그램 종료합니다.

    detector = cv2.QRCodeDetector() # QR 코드 탐지 및 디코딩을 위한 객체를 생성
    
    # 1단계: 기본 인식 방법 성능 측정
    print("--- 1단계: 기본 인식 방법 성능 측정 시작 (10초) ---") # 테스트 시작 메시지를 출력
    start_time_base = time.time() # 현재 시간을 기록하여 테스트 시간을 측정
    frame_count_base = 0 # 처리된 프레임 수를 0으로 초기화
    success_count_base = 0 # 인식에 성공한 QR 코드 수를 0으로 초기화
    
    while time.time() - start_time_base < 10: # 10초 동안 반복
        ret, frame = cap.read() # 웹캠에서 프레임을 읽음
        if not ret: break # 실패시 루프 종료
        
        frame_count_base += 1 # 프레임 수를 1 증가
        
        # 기본 인식 방법: 전처리 없이 바로 인식
        data, _, _ = detector.detectAndDecode(frame) # 전처리 없이 프레임 전체에서 QR 코드를 인식
        if data: # QR 코드 데이터가 존재하면
            success_count_base += 1 # 성공 횟수를 1 증가
        
        # 프레임에 메시지 표시
        display_frame = put_text_on_frame(frame.copy(), "기본 방법 테스트 중", (10, 30), (255, 0, 0)) # 프레임에 테스트 중임을 표시합니다.
        if data: # QR 코드 데이터가 존재하면
            display_frame = put_text_on_frame(display_frame, f"QR 인식됨: {data}", (10, 60), (0, 255, 0)) # 인식된 데이터를 표시합니다.
        
        cv2.imshow("QR 성능 테스트", display_frame) # 화면에 프레임을 나타냄
        if cv2.waitKey(1) & 0xFF == ord('q'): break # 'q' 키를 누르면 루프를 종료
        
    cv2.destroyAllWindows() 
    
    # 최종 FPS 및 인식률 계산
    elapsed_base = time.time() - start_time_base # 경과 시간을 계산
    fps_base = frame_count_base / elapsed_base if elapsed_base > 0 else 0 # 초당 프레임 수(FPS)를 계산
    success_rate_base = success_count_base / frame_count_base * 100 if frame_count_base > 0 else 0 # 인식 성공률을 계산
    
    # 2단계: 향상된 방법 성능 측정
    print("--- 2단계: 향상된 방법 테스트 시작 (10초) ---") # 테스트 시작 메시지를 출력
    start_time_improved = time.time() # 현재 시간을 기록하여 테스트 시간을 측정
    frame_count_improved = 0 # 처리된 프레임 수를 0으로 초기화
    success_count_improved = 0 # 성공 횟수를 0으로 초기화
    
    while time.time() - start_time_improved < 10: # 10초 동안 반복
        ret, frame = cap.read() # 웹캠에서 프레임을 읽음
음        if not ret: break # 실패시 루프 종료
        
        frame_count_improved += 1 # 프레임 수를 1 증가
        
        data = None # 디코딩된 데이터를 초기화
        
        # 노이즈 제거 및 조명 보정 적용
        processed_frame = remove_noise(frame.copy()) # 프레임에 노이즈 제거를 적용
        processed_frame = adjust_lighting(processed_frame) # 노이즈 제거된 프레임에 조명 보정을 적용
        
        # QR 코드 디텍터로 바운딩 박스 찾아내기
        _, points, _ = detector.detectAndDecode(processed_frame) # 전처리된 프레임에서 QR 코드의 위치를 찾아냄

        if points is not None and points.size > 0: # qr코드 가 인식되었는지 확인
            # 바운딩 박스 그리기
            cv2.polylines(frame, points.astype(int), True, (0, 255, 255), 3) # 원본 프레임에 QR 코드의 윤곽선을 그림

            # 바운딩 박스 영역만 잘라서 디코딩 시도
            x, y, w, h = cv2.boundingRect(points.astype(int)) # QR 코드 위치의 x, y, 너비, 높이를 가져옴
            
            # ROI(관심 영역) 좌표가 유효한지 안전하게 확인
            y_safe = max(0, y) # y 좌표가 0보다 작아지지 않도록 보정
            x_safe = max(0, x) # x 좌표가 0보다 작아지지 않도록 보정
            y2_safe = min(processed_frame.shape[0], y + h) # y + h 값이 이미지 높이를 넘지 않도록 보정
            x2_safe = min(processed_frame.shape[1], x + w) # x + w 값이 이미지 너비를 넘지 않도록 보정
            
            if (x2_safe - x_safe) > 0 and (y2_safe - y_safe) > 0: # 잘라낼 영역의 너비와 높이가 0보다 큰지 확인
                roi_to_decode = processed_frame[y_safe:y2_safe, x_safe:x2_safe] # 전처리된 프레임에서 roi를 잘라냄
                data, _, _ = detector.detectAndDecode(roi_to_decode) # 잘라낸 ROI에서 QR 코드를 디코딩
        
        if data: # QR 코드 데이터가 존재하면
            success_count_improved += 1 # 성공 횟수를 1 증가
        
        # 프레임에 메시지 표시
        display_frame = put_text_on_frame(frame, "향상된 방법 테스트 중", (10, 30), (0, 255, 0)) # 프레임에 테스트 중임을 표시
        if data: # QR 코드 데이터가 존재하면
            display_frame = put_text_on_frame(display_frame, f"QR 인식됨: {data}", (10, 60), (0, 255, 0)) # 인식된 데이터를 표시
        
        cv2.imshow("QR 성능 테스트", display_frame) # 화면에 프레임을 보여줌
        if cv2.waitKey(1) & 0xFF == ord('q'): break # 'q' 키를 누르면 루프를 종료
    
    cv2.destroyAllWindows() 

    # 최종 FPS 및 인식률 계산
    elapsed_improved = time.time() - start_time_improved # 경과 시간을 계산
    fps_improved = frame_count_improved / elapsed_improved if elapsed_improved > 0 else 0 # 초당 프레임 수를 계산
    success_rate_improved = success_count_improved / frame_count_improved * 100 if frame_count_improved > 0 else 0 # 인식 성공률을 계산

    # 최종 결과 표시
    final_graph = create_graph_image(success_rate_base, success_rate_improved, fps_base, fps_improved) # 최종 결과 그래프 이미지를 생성
    cv2.imshow("QR 성능 최종 결과", final_graph) # 그래프를 화면에 보여줌

    print("\n--- 분석 완료 ---") # 완료 메시지를 출력
    print(f"기본 방법: 평균 FPS {fps_base:.2f}, 인식률 {success_rate_base:.2f}%") # 기본 방법의 최종 결과를 출력
    print(f"향상된 방법: 평균 FPS {fps_improved:.2f}, 인식률 {success_rate_improved:.2f}%") # 향상된 방법의 최종 결과를 출력
    
    cv2.waitKey(0) 
    cap.release() 
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    main()
