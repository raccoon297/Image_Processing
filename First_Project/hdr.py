import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 불러오기
img = cv2.imread('input.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 밝기 조절 함수
def adjust_brightness(image, factor):
    return np.clip(image * factor, 0, 255).astype(np.uint8)

# 밝기 다른 이미지 생성
img_dark = adjust_brightness(img, 0.5)
img_mid = adjust_brightness(img, 1.0)
img_bright = adjust_brightness(img, 1.5)

# BGR 변환
img_dark_bgr = cv2.cvtColor(img_dark, cv2.COLOR_RGB2BGR)
img_mid_bgr = cv2.cvtColor(img_mid, cv2.COLOR_RGB2BGR)
img_bright_bgr = cv2.cvtColor(img_bright, cv2.COLOR_RGB2BGR)

# HDR 생성
images = [img_dark_bgr, img_mid_bgr, img_bright_bgr]
merge_debevec = cv2.createMergeDebevec()
hdr = merge_debevec.process(images, times=np.array([1/30.0, 0.25, 2.5], dtype=np.float32))

# Tone Mapping
tonemap = cv2.createTonemap(gamma=2.2)
ldr = tonemap.process(hdr)
ldr = np.clip(ldr * 255, 0, 255).astype('uint8')
ldr = cv2.cvtColor(ldr, cv2.COLOR_BGR2RGB)

# 출력
plt.figure(figsize=(12,6))

# 원본 + HDR
plt.subplot(2,3,1)
plt.title("Original")
plt.imshow(img)
plt.axis('off')

plt.subplot(2,3,2)
plt.title("HDR Result")
plt.imshow(ldr)
plt.axis('off')

# 밝기별 이미지
plt.subplot(2,3,4)
plt.title("Dark")
plt.imshow(img_dark)
plt.axis('off')

plt.subplot(2,3,5)
plt.title("Normal")
plt.imshow(img_mid)
plt.axis('off')

plt.subplot(2,3,6)
plt.title("Bright")
plt.imshow(img_bright)
plt.axis('off')

plt.tight_layout()
plt.show()
