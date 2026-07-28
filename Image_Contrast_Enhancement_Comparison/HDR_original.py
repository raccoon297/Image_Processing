import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 3장의 사진 로드
img_paths = ["images/under.jpg", "images/normal.jpg", "images/over.jpg"]
images = [cv2.imread(p) for p in img_paths]

# 이미지 로드 확인
for i, img in enumerate(images):
    if img is None:
        print(f"{img_paths[i]} 이미지를 찾을 수 없습니다.")
        exit()

# 모든 이미지의 크기를 첫 번째 이미지와 동일하게 강제 맞춤
target_height, target_width = images[0].shape[:2]
for i in range(1, len(images)):
    images[i] = cv2.resize(images[i], (target_width, target_height))


# 시각화를 위해 정렬(Alignment) 이전에 원본 이미지들을 RGB로 복사해 둡니다.
# (alignMTB가 원본 리스트를 수정하기 때문입니다)
images_rgb = [cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB) for img in images]

# 2. 이미지 정렬 (Alignment)
alignMTB = cv2.createAlignMTB()
alignMTB.process(images, images)

# 3. Mertens Fusion 알고리즘으로 블렌딩
merge_mertens = cv2.createMergeMertens()
hdr_mertens = merge_mertens.process(images)

# 4. 8비트 이미지로 변환 및 결과 확인
hdr_8bit = np.clip(hdr_mertens * 255, 0, 255).astype('uint8')
hdr_rgb = cv2.cvtColor(hdr_8bit, cv2.COLOR_BGR2RGB)

# 5. 시각화 (2행 3열 구조 활용)
plt.figure(figsize=(15, 8))

# [첫 번째 줄] 1, 2, 3번째 칸에 각각 입력 이미지 배치
plt.subplot(2, 3, 1)
plt.imshow(images_rgb[0])
plt.title('Under-exposed')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(images_rgb[1])
plt.title('Normal-exposed')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(images_rgb[2])
plt.title('Over-exposed')
plt.axis('off')

# [두 번째 줄] 4, 5, 6번째 칸 중 가운데인 5번째 칸에 HDR 배치
plt.subplot(2, 3, 5)
plt.imshow(hdr_rgb)
plt.title('Final HDR Result')
plt.axis('off')

plt.tight_layout()
plt.show()