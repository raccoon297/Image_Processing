import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 불러오기
img = cv2.imread('input.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. 밝기 다른 이미지 3개 만들기
def adjust_brightness(image, factor):
    return np.clip(image * factor, 0, 255).astype(np.uint8)

img_dark = adjust_brightness(img, 0.5)   # 어둡게
img_mid  = adjust_brightness(img, 1.0)   # 원본
img_bright = adjust_brightness(img, 1.5) # 밝게

# OpenCV는 BGR 사용 → 다시 변환
img_dark_bgr = cv2.cvtColor(img_dark, cv2.COLOR_RGB2BGR)
img_mid_bgr = cv2.cvtColor(img_mid, cv2.COLOR_RGB2BGR)
img_bright_bgr = cv2.cvtColor(img_bright, cv2.COLOR_RGB2BGR)

# 3. HDR Merge (Debevec 방법)
images = [img_dark_bgr, img_mid_bgr, img_bright_bgr]

merge_debevec = cv2.createMergeDebevec()
hdr = merge_debevec.process(images, times=np.array([1/30.0, 0.25, 2.5], dtype=np.float32))

# 4. Tone Mapping (HDR → 일반 이미지로 변환)
tonemap = cv2.createTonemap(gamma=2.2)
ldr = tonemap.process(hdr)

# 값 보정
ldr = np.clip(ldr * 255, 0, 255).astype('uint8')
ldr = cv2.cvtColor(ldr, cv2.COLOR_BGR2RGB)

# 5. 결과 출력
plt.figure(figsize=(12,6))

plt.subplot(1,4,1)
plt.title("Dark")
plt.imshow(img_dark)
plt.axis('off')

plt.subplot(1,4,2)
plt.title("Normal")
plt.imshow(img_mid)
plt.axis('off')

plt.subplot(1,4,3)
plt.title("Bright")
plt.imshow(img_bright)
plt.axis('off')

plt.subplot(1,4,4)
plt.title("HDR Result")
plt.imshow(ldr)
plt.axis('off')

plt.show()


# 6. 히스토그램 평활화

# grayscale 변환
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 기본 히스토그램 평활화
hist_eq = cv2.equalizeHist(gray)

# CLAHE (Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_img = clahe.apply(gray)

# 7. 결과 출력 (비교)
plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.title("Original")
plt.imshow(img)
plt.axis('off')

plt.subplot(2,3,2)
plt.title("HDR")
plt.imshow(ldr)
plt.axis('off')

plt.subplot(2,3,3)
plt.title("Grayscale")
plt.imshow(gray, cmap='gray')
plt.axis('off')

plt.subplot(2,3,4)
plt.title("Histogram Equalization")
plt.imshow(hist_eq, cmap='gray')
plt.axis('off')

plt.subplot(2,3,5)
plt.title("CLAHE")
plt.imshow(clahe_img, cmap='gray')
plt.axis('off')

plt.show()

# 8. 히스토그램 그래프까지 추가

plt.figure(figsize=(12,8))

# (1) 원본 컬러
plt.subplot(3,3,1)
plt.title("Original (Color)")
plt.imshow(img)
plt.axis('off')

# (2) HDR
plt.subplot(3,3,4)
plt.title("HDR")
plt.imshow(ldr)
plt.axis('off')

# (3) 흑백 원본
plt.subplot(3,3,2)
plt.title("Grayscale Original")
plt.imshow(gray, cmap='gray')
plt.axis('off')

# (4) 히스토그램 평활화 결과
plt.subplot(3,3,5)
plt.title("Histogram Equalization")
plt.imshow(hist_eq, cmap='gray')
plt.axis('off')

# (5) CLAHE 결과
plt.subplot(3,3,6)
plt.title("CLAHE")
plt.imshow(clahe_img, cmap='gray')
plt.axis('off')

# (6) 원본 히스토그램
plt.subplot(3,3,7)
plt.title("Histogram (Original)")
plt.hist(gray.ravel(), bins=256)

# (7) 평활화 히스토그램
plt.subplot(3,3,8)
plt.title("Histogram (Equalized)")
plt.hist(hist_eq.ravel(), bins=256)

# (8) CLAHE 히스토그램
plt.subplot(3,3,9)
plt.title("Histogram (CLAHE)")
plt.hist(clahe_img.ravel(), bins=256)

plt.tight_layout()
plt.show()