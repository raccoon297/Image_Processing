import cv2
import matplotlib.pyplot as plt

# 1. 이미지 로드 (컬러)
image_path = "images/input.jpg" # 사용할 이미지 이름으로 변경하세요
# OpenCV는 기본적으로 BGR 포맷으로 이미지를 읽어옵니다.
img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

if img_bgr is None:
    print("이미지를 찾을 수 없습니다.")
    exit()

# 2. 색공간 변환 및 밝기 채널 분리 (BGR -> YCrCb)
# RGB 채널의 상관관계로 인한 색상 왜곡을 막기 위해 YCrCb 색공간으로 변환합니다.
img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

# Y(밝기), Cr(붉은색 색차), Cb(푸른색 색차) 채널로 분리
y, cr, cb = cv2.split(img_ycrcb)

# 3. Histogram Equalization 적용 (Y 채널에만)
y_eq = cv2.equalizeHist(y)

# 4. 채널 병합 및 원래 색공간으로 복귀
# 평활화된 Y채널과 기존의 Cr, Cb 채널을 다시 합칩니다.
img_ycrcb_eq = cv2.merge((y_eq, cr, cb))

# 결과를 화면에 출력하기 위해 다시 BGR 색공간으로 변환 (OpenCV용)
img_bgr_eq = cv2.cvtColor(img_ycrcb_eq, cv2.COLOR_YCrCb2BGR)

# Matplotlib 출력을 위해 BGR을 RGB로 변환
img_rgb_original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_rgb_eq = cv2.cvtColor(img_bgr_eq, cv2.COLOR_BGR2RGB)

# 5. 결과 시각화
plt.figure(figsize=(10, 8))

# 원본 컬러 이미지
plt.subplot(2, 2, 1)
plt.imshow(img_rgb_original)
plt.title('Original Color Image')
plt.axis('off')

# 원본 Y(밝기) 채널 히스토그램
plt.subplot(2, 2, 3)
plt.hist(y.ravel(), bins=256, range=[0, 256], color='gray')
plt.title('Original Y-Channel Histogram')
plt.xlim([0, 256])

# Color HE 결과 이미지
plt.subplot(2, 2, 2)
plt.imshow(img_rgb_eq)
plt.title('Color Histogram Equalization (HE)')
plt.axis('off')

# 평활화된 Y(밝기) 채널 히스토그램
plt.subplot(2, 2, 4)
plt.hist(y_eq.ravel(), bins=256, range=[0, 256], color='gray')
plt.title('Equalized Y-Channel Histogram')
plt.xlim([0, 256])

plt.tight_layout()
plt.show()