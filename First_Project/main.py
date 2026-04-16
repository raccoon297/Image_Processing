import cv2
import numpy as np
import matplotlib.pyplot as plt

##-----------------------------------------------------P-HDR-------------------------------------------------------------##

def adjust_gamma(image, gamma=1.0):
    """감마 보정을 통해 이미지 밝기를 조절하는 함수"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# 1. 이미지 로드 (컬러)
image_path = "input.jpg" # 사용할 이미지 이름으로 변경하세요
img = cv2.imread(image_path)

if img is None:
    print("이미지를 찾을 수 없습니다.")
    exit()

# Matplotlib 출력을 위해 BGR을 RGB로 변환
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. 가상의 다중 노출 이미지 생성 (감마 보정 활용)
img_underexposed = adjust_gamma(img_rgb, gamma=0.4) # 어두운 이미지 (명부 디테일용)
img_overexposed = adjust_gamma(img_rgb, gamma=2.2)  # 밝은 이미지 (암부 디테일용)

# 3. Mertens Fusion 알고리즘으로 블렌딩
merge_mertens = cv2.createMergeMertens()
exposure_images = [img_underexposed, img_rgb, img_overexposed]
hdr_mertens = merge_mertens.process(exposure_images)

# 출력을 위해 0~255 범위의 8비트 이미지로 변환
hdr_8bit = np.clip(hdr_mertens * 255, 0, 255).astype('uint8')

##----------------------------------------------------Color_HE-----------------------------------------------------------##

# 1. 이미지 로드 (컬러)

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

##--------------------------------------------------Color_CLAHE----------------------------------------------------------##

# 1. 이미지 로드 (컬러)

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

##-----------------------------------------------------시각화-------------------------------------------------------------##

# 4. 결과 시각화
plt.figure(figsize=(12, 8))

# 원본 이미지 
plt.subplot(3, 3, 2)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')
# P-HDR 결과 이미지
plt.subplot(3, 3, 4)
plt.imshow(hdr_8bit)
plt.title('HDR Result')
plt.axis('off')
# Color HE 결과 이미지
plt.subplot(3, 3, 5)
plt.imshow(img_rgb_eq)
plt.title('Color Histogram Equalization (HE)')
plt.axis('off')
# 평활화된 Y(밝기) 채널 히스토그램
plt.subplot(3, 3, 8)
plt.hist(y_eq.ravel(), bins=256, range=[0, 256], color='gray')
plt.title('Equalized Y-Channel Histogram')
plt.xlim([0, 256])
# 원본 히스토그램
plt.subplot(3, 3, 7)
plt.hist(l.ravel(), bins=256, range=[0, 256], color='gray')
plt.title('Original Histogram')
# 컬러 CLAHE 결과 이미지
plt.subplot(3, 3, 6)
plt.imshow(img_rgb_clahe)
plt.title('Color CLAHE (Lab Space)')
plt.axis('off')
# 컬러 CLAHE 히스토그램
plt.subplot(3, 3, 9)
plt.hist(l_clahe.ravel(), bins=256, range=[0, 256], color='gray')
plt.title('CLAHE L-Channel Histogram')

plt.tight_layout()
plt.show()