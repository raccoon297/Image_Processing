import cv2
import matplotlib.pyplot as plt

# 이미지 불러오기
img = cv2.imread('input.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# grayscale 변환
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 히스토그램 평활화
hist_eq = cv2.equalizeHist(gray)

# 한 화면에 출력
plt.figure(figsize=(10,8))

# 위쪽: 이미지
plt.subplot(2,2,1)
plt.title("Grayscale Original")
plt.imshow(gray, cmap='gray')
plt.axis('off')

plt.subplot(2,2,2)
plt.title("Histogram Equalized")
plt.imshow(hist_eq, cmap='gray')
plt.axis('off')

# 아래쪽: 히스토그램
plt.subplot(2,2,3)
plt.title("Histogram (Original)")
plt.hist(gray.ravel(), bins=256)

plt.subplot(2,2,4)
plt.title("Histogram (Equalized)")
plt.hist(hist_eq.ravel(), bins=256)

plt.tight_layout()
plt.show()