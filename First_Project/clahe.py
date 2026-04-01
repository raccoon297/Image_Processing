import cv2
import matplotlib.pyplot as plt

# 이미지 불러오기
img = cv2.imread('input.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# grayscale 변환
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# CLAHE 적용
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_img = clahe.apply(gray)

# 한 화면에 출력
plt.figure(figsize=(10,8))

# 위쪽: 이미지
plt.subplot(2,2,1)
plt.title("Grayscale Original")
plt.imshow(gray, cmap='gray')
plt.axis('off')

plt.subplot(2,2,2)
plt.title("CLAHE Result")
plt.imshow(clahe_img, cmap='gray')
plt.axis('off')

# 아래쪽: 히스토그램
plt.subplot(2,2,3)
plt.title("Histogram (Original)")
plt.hist(gray.ravel(), bins=256)

plt.subplot(2,2,4)
plt.title("Histogram (CLAHE)")
plt.hist(clahe_img.ravel(), bins=256)

plt.tight_layout()
plt.show()