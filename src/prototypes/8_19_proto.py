# 필요한 라이브러리 가져오기
import cv2                         # 컴퓨터 비전 작업을 위한 OpenCV 라이브러리
import numpy as np                 # 배열 및 행렬 연산을 위한 라이브러리
import matplotlib.pyplot as plt    # 데이터 시각화를 위한 라이브러리

# 사용자 안내: 아래 QR 코드 이미지 경로를 실제 파일 경로로 변경해주세요.
# 예시: 'C:/Users/YourName/Desktop/test_qr.png'
QR_CODE_IMAGE_PATH = "./sample_qr.png" 

def load_and_preprocess_image(path):
    """
    이미지를 불러오고, 전처리하여 실험에 사용할 테스트 이미지 리스트를 생성합니다.
    (노이즈 및 블러를 추가하여 다양한 환경을 시뮬레이션)
    """
    try:
        # QR 코드 이미지 불러오기
        image = cv2.imread(path)
        if image is None:
            print(f"오류: '{path}' 경로의 이미지를 찾을 수 없습니다.")
            return []

        test_images = [image] # 원본 이미지 추가

        # 다양한 난이도의 테스트 이미지를 생성 (노이즈, 블러)
        for i in range(1, 6):
            # 노이즈 추가
            noise = np.random.normal(0, i * 15, image.shape).astype(np.uint8)
            noisy_image = cv2.add(image, noise)
            test_images.append(noisy_image)

            # 블러 추가
            blur_kernel = (i * 2 + 1, i * 2 + 1)
            blurred_image = cv2.GaussianBlur(image, blur_kernel, 0)
            test_images.append(blurred_image)
            
        print(f"총 {len(test_images)}개의 테스트 이미지가 준비되었습니다.")
        return test_images
        
    except Exception as e:
        print(f"이미지 불러오기 또는 전처리 중 오류 발생: {e}")
        return []


def detect_qr_baseline(image):
    """
    기본 QR 코드 인식 방법: 전처리 없이 이미지 전체를 디텍터에 전달합니다.
    """
    detector = cv2.QRCodeDetector()
    data, _, _ = detector.detectAndDecode(image)
    return data is not None


def detect_qr_improved(image):
    """
    개선된 QR 코드 인식 방법: 주신 코드와 같이 전처리 후 인식합니다.
    """
    detector = cv2.QRCodeDetector()
    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 가우시안 블러 (노이즈 감소)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 이진화 처리 (명암 대비 강화)
    _, binarized = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    data, _, _ = detector.detectAndDecode(binarized)
    return data is not None


def visualize_results(baseline_successes, improved_successes):
    """
    두 방법의 인식 성공률을 막대 그래프로 시각화합니다.
    """
    total_tests = len(baseline_successes)
    
    # 성공률 계산
    baseline_rate = sum(baseline_successes) / total_tests * 100
    improved_rate = sum(improved_successes) / total_tests * 100
    
    print(f"\n--- 결과 ---")
    print(f"총 테스트 이미지 수: {total_tests}")
    print(f"기본 방법 인식 성공률: {baseline_rate:.2f}%")
    print(f"개선된 방법 인식 성공률: {improved_rate:.2f}%")
    
    # 그래프 데이터 설정
    labels = ['기본 방법', '개선된 방법']
    rates = [baseline_rate, improved_rate]
    
    # 그래프 그리기
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, rates, color=['#ff9999', '#66b3ff'])
    
    # 막대 위에 값 표시
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom', fontsize=12)
        
    plt.ylim(0, 100) # y축 범위를 0-100%로 설정
    plt.title('QR 코드 인식 성능 비교', fontsize=16)
    plt.ylabel('인식 성공률 (%)', fontsize=12)
    plt.show()


def main():
    # 테스트 이미지 생성
    test_images = load_and_preprocess_image(QR_CODE_IMAGE_PATH)
    if not test_images:
        return
        
    baseline_results = []
    improved_results = []

    # 각 테스트 이미지에 대해 두 가지 방법으로 인식 테스트
    for i, img in enumerate(test_images):
        print(f"테스트 이미지 {i+1}/{len(test_images)} 처리 중...")
        
        # 기본 방법으로 테스트
        baseline_results.append(detect_qr_baseline(img))
        
        # 개선된 방법으로 테스트
        improved_results.append(detect_qr_improved(img))

    # 결과 시각화
    visualize_results(baseline_results, improved_results)


if __name__ == "__main__":
    main()