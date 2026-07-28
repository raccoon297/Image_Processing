import cv2
import matplotlib.pyplot as plt

# 1. 이미지 로드 (컬러)
image_path = "images/input.jpg" # 사용할 이미지 이름으로 변경하세요
img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

if img_bgr is None:
    print("이미지를 찾을 수 없습니다.")
    exit()

# 2. 색공간 변환 (BGR -> Lab)
# Lab 색공간은 밝기(L)와 색상(a, b)이 완전히 분리되어 있어 CLAHE 처리에 최적입니다.
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)

# 3. 채널 분리
l, a, b = cv2.split(img_lab)

# 4. CLAHE 객체 생성 및 L 채널 적용
# clipLimit: 대비 한계 (보통 2.0 ~ 4.0), tileGridSize: 타일 분할 크기
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
l_clahe = clahe.apply(l)

# 5. 채널 병합 및 원래 색공간으로 복귀
img_lab_clahe = cv2.merge((l_clahe, a, b))
img_bgr_clahe = cv2.cvtColor(img_lab_clahe, cv2.COLOR_Lab2BGR)

# 시각화를 위해 RGB로 변환
img_rgb_original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_rgb_clahe = cv2.cvtColor(img_bgr_clahe, cv2.COLOR_BGR2RGB)

# 6. 결과 시각화
plt.figure(figsize=(10, 8))

# 원본 이미지 및 히스토그램
plt.subplot(2, 2, 1)
plt.imshow(img_rgb_original)
plt.title('Original Color Image')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.hist(l.ravel(), bins=256, range=[0, 256], color='gray')
plt.title('Original L-Channel Histogram')

# 컬러 CLAHE 결과 및 히스토그램
plt.subplot(2, 2, 2)
plt.imshow(img_rgb_clahe)
plt.title('Color CLAHE (Lab Space)')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.hist(l_clahe.ravel(), bins=256, range=[0, 256], color='gray')
plt.title('CLAHE L-Channel Histogram')

plt.tight_layout()
plt.show()