import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# 4. 결과 시각화
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(img_underexposed)
plt.title('Under-exposed (Gamma 0.4)')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(img_overexposed)
plt.title('Over-exposed (Gamma 2.2)')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(hdr_8bit)
plt.title('HDR Result')
plt.axis('off')

plt.tight_layout()
plt.show()

