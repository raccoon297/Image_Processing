import cv2
import matplotlib.pyplot as plt

# 1. 이미지 로드 (흑백)
image_path = "images/input.jpg" # 사용할 이미지 이름으로 변경하세요
img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img_gray is None:
    print("이미지를 찾을 수 없습니다.")
    exit()

# 2. Histogram Equalization 적용
img_he = cv2.equalizeHist(img_gray)

# 3. 결과 시각화
plt.figure(figsize=(10, 8))

# 원본
plt.subplot(2, 2, 1)
plt.imshow(img_gray, cmap='gray', vmin=0, vmax=255)
plt.title('Original Grayscale')
plt.axis('off')

plt.subplot(2, 2, 3)
# plt.plot 대신 plt.hist() 사용
plt.hist(img_gray.ravel(), bins=256, range=[0, 256])
plt.title('Original Histogram')
plt.xlim([0, 256])

# HE 결과
plt.subplot(2, 2, 2)
plt.imshow(img_he, cmap='gray', vmin=0, vmax=255)
plt.title('Histogram Equalization (HE)')
plt.axis('off')

plt.subplot(2, 2, 4)
# plt.plot 대신 plt.hist() 사용
plt.hist(img_he.ravel(), bins=256, range=[0, 256])
plt.title('HE Histogram')
plt.xlim([0, 256])

plt.tight_layout()
plt.show()

