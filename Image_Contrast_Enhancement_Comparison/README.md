# Exposure and Contrast Enhancement in Digital Images

> 영상신호처리 중간고사 대체 프로젝트  
> **단일·다중 노출 영상과 히스토그램 기반 기법을 이용한 밝기 및 대비 개선 비교**

## 프로젝트 소개

밝은 실외와 어두운 실내가 한 화면에 함께 존재하는 영상은 카메라의 제한된 동적 범위 때문에 어느 한쪽의 세부 정보가 쉽게 손실된다.

이 프로젝트에서는 이러한 문제를 개선하기 위해 다음 영상처리 기법을 직접 구현하고 결과를 비교하였다.

- 단일 영상 기반 **Pseudo-HDR**
- 다중 노출 영상 기반 **Exposure Fusion**
- 전역 대비 개선을 위한 **Histogram Equalization**
- 국부 대비 개선을 위한 **CLAHE**
- 컬러 영상에 적용한 **Color HE**와 **Color CLAHE**

단순히 결과 영상을 생성하는 데 그치지 않고, 각 기법의 동작 원리와 장단점을 이미지 및 히스토그램을 통해 분석하는 것을 목표로 하였다.

## 주요 결과

| 기법 | 처리 방식 | 장점 | 한계 |
|---|---|---|---|
| Pseudo-HDR | 한 장의 영상에서 감마 보정으로 가상 노출 영상을 생성한 뒤 융합 | 별도의 다중 촬영이 필요 없고 움직이는 피사체에 의한 고스트 현상이 적음 | 실제 노출 정보를 추가하는 방식이 아니므로 질감과 색감 복원에 한계가 있음 |
| Multi-exposure Fusion | 서로 다른 노출의 세 영상을 정렬한 뒤 Mertens Exposure Fusion 수행 | 정적인 장면에서 암부 질감과 색감을 자연스럽게 복원 | 촬영 사이에 움직인 피사체에서 고스트 현상이 발생할 수 있음 |
| Histogram Equalization | 영상 전체의 밝기 분포를 전역적으로 재분배 | 구현이 간단하고 빠르며 전체 대비를 크게 향상 | 밝은 영역의 클리핑과 암부 노이즈 증폭 가능 |
| CLAHE | 영상을 타일로 분할하여 국부 평활화를 수행하고 대비를 제한 | 밝은 영역을 비교적 보존하면서 암부 디테일을 개선 | 영상에 맞는 `clipLimit`과 `tileGridSize` 설정 필요 |

실험 결과, **전체 영상에 동일한 변환을 적용하는 HE보다 국부적으로 대비를 조절하는 CLAHE가 밝기 차이가 큰 장면에서 더 균형 잡힌 결과를 보였다.** 또한 정적인 풍경에서는 실제 다중 노출 영상의 융합이 더 풍부한 정보를 제공했지만, 움직이는 피사체가 포함된 환경에서는 단일 영상 기반 Pseudo-HDR이 안정적이었다.

## 결과 이미지



## 구현 내용

### 1. Pseudo-HDR

한 장의 입력 영상에 서로 다른 감마 값을 적용하여 저노출·정상노출·과노출 영상을 가상으로 생성한 뒤, OpenCV의 Mertens Exposure Fusion으로 결합하였다.

```python
img_underexposed = adjust_gamma(img_rgb, gamma=0.4)
img_overexposed = adjust_gamma(img_rgb, gamma=2.2)

merge_mertens = cv2.createMergeMertens()
exposure_images = [img_underexposed, img_rgb, img_overexposed]
hdr_mertens = merge_mertens.process(exposure_images)
```

감마 보정은 LUT(Look-Up Table)를 사용해 모든 픽셀에서 반복되는 계산을 줄였다.

### 2. 다중 노출 영상 융합

실제로 노출이 다른 세 장의 이미지를 입력으로 사용하였다. 이미지 크기를 통일하고 `AlignMTB`로 정렬한 뒤 Mertens Fusion을 수행하였다.

```python
align_mtb = cv2.createAlignMTB()
align_mtb.process(images, images)

merge_mertens = cv2.createMergeMertens()
hdr_mertens = merge_mertens.process(images)
```

이 방식은 정적인 장면에서 암부의 신호 대 잡음비와 색 표현이 우수하지만, 촬영 사이에 이동한 피사체는 정렬만으로 보정되지 않아 잔상으로 남을 수 있다.

### 3. Histogram Equalization

입력 영상을 그레이스케일로 변환한 뒤 전역 히스토그램 평활화를 적용하였다.

```python
img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img_he = cv2.equalizeHist(img_gray)
```

좁은 밝기 구간에 집중된 픽셀을 넓게 재배치하여 대비를 높였지만, 영상 전체에 동일한 변환을 적용하기 때문에 이미 밝은 영역까지 과도하게 강조되는 현상을 확인하였다.

### 4. CLAHE

영상을 작은 타일로 나누어 각 영역의 대비를 독립적으로 개선하고, `clipLimit`으로 과도한 대비 및 노이즈 증폭을 제한하였다.

```python
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8),
)
img_clahe = clahe.apply(img_gray)
```

실험에서는 전역 HE보다 실외의 밝은 영역을 잘 보존하면서 아치 내부의 벽면과 자전거 디테일을 개선하였다.

### 5. 컬러 영상 처리

RGB 각 채널에 직접 평활화를 적용하면 채널 간 비율이 달라져 색 왜곡이 발생할 수 있다. 이를 방지하기 위해 밝기와 색상 정보를 분리한 색 공간을 사용하였다.

- **Color HE:** YCrCb 색 공간의 `Y` 밝기 채널에만 HE 적용
- **Color CLAHE:** Lab 색 공간의 `L` 밝기 채널에만 CLAHE 적용

```python
# Color HE
img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
y, cr, cb = cv2.split(img_ycrcb)
y_eq = cv2.equalizeHist(y)

# Color CLAHE
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
l, a, b = cv2.split(img_lab)
l_clahe = clahe.apply(l)
```

## 프로젝트 구조

```text
First_Project/
├── CLAHE.py                # 그레이스케일 CLAHE
├── Color_CLAHE.py          # Lab 색 공간 기반 컬러 CLAHE
├── Color_HE.py             # YCrCb 색 공간 기반 컬러 HE
├── HDR_original.py         # 실제 다중 노출 영상 융합
├── HE.py                    # 그레이스케일 Histogram Equalization
├── P-HDR.py                 # 단일 영상 기반 Pseudo-HDR
├── main(1).py               # Pseudo-HDR, Color HE, Color CLAHE 비교
├── input.jpg                # 단일 영상 실험 입력
├── input3.jpg               # 추가 실험 영상
├── under.jpg                # 저노출 입력
├── normal.jpg               # 정상 노출 입력
├── over.jpg                 # 과노출 입력
├── requirements.txt
└── 2021742065_윤동환_영산신호처리_중간대체레포트.pdf
```

## 실행 환경

- Python 3
- OpenCV
- NumPy
- Matplotlib

### 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 실행 예시

```bash
python P-HDR.py
python HDR_original.py
python HE.py
python CLAHE.py
python Color_HE.py
python Color_CLAHE.py
```

각 스크립트는 현재 폴더의 입력 이미지를 읽고, 원본과 처리 결과 또는 히스토그램을 Matplotlib 창으로 출력한다.

## 파일별 역할

| 파일 | 설명 |
|---|---|
| `P-HDR.py` | 감마 보정으로 가상 노출 영상을 만든 뒤 Mertens Fusion 수행 |
| `HDR_original.py` | 실제 저노출·정상노출·과노출 영상의 정렬 및 Exposure Fusion 수행 |
| `HE.py` | 그레이스케일 전역 히스토그램 평활화 |
| `CLAHE.py` | 그레이스케일 국부 히스토그램 평활화 |
| `Color_HE.py` | YCrCb의 Y 채널을 이용한 컬러 HE |
| `Color_CLAHE.py` | Lab의 L 채널을 이용한 컬러 CLAHE |
| `main(1).py` | 주요 컬러 처리 결과를 한 화면에서 비교 |

## 분석 및 배운 점

1. **동일한 대비 개선 기법도 영상의 밝기 분포에 따라 결과가 크게 달라진다.**  
   전역 HE는 빠르고 강한 대비 개선이 가능하지만, 밝고 어두운 영역이 함께 존재하는 영상에서는 클리핑과 노이즈 증폭이 발생할 수 있다.

2. **CLAHE는 대비 개선과 왜곡 억제 사이의 균형이 중요하다.**  
   `clipLimit`과 `tileGridSize`를 조정하면 국부 디테일을 개선할 수 있지만, 모든 영상에 동일한 값이 최적인 것은 아니다.

3. **Pseudo-HDR과 실제 다중 노출 융합은 대체 관계가 아니라 상황에 따른 선택 관계다.**  
   정적인 장면에서는 실제 다중 노출 영상이 더 풍부한 신호를 제공하지만, 동적인 장면에서는 단일 입력 기반 방식이 고스트 현상에 더 강하다.

4. **컬러 영상에서는 밝기와 색상 채널을 분리해야 한다.**  
   밝기 채널에만 처리 기법을 적용함으로써 대비를 개선하면서 색 왜곡을 줄일 수 있었다.

## 한계 및 향후 개선

- 현재 비교는 주로 시각적 결과와 히스토그램 분석을 중심으로 이루어졌다.
- 향후에는 처리 시간, 엔트로피, 대비 지표, PSNR 또는 SSIM 등 정량 지표를 추가해 기법별 성능을 비교할 수 있다.
- CLAHE의 파라미터를 입력 영상의 조도 및 밝기 분포에 따라 자동으로 선택하는 방법을 실험할 수 있다.
- 움직이는 피사체가 포함된 다중 노출 영상에서 고스트 현상을 억제하는 정합 및 deghosting 기법을 추가할 수 있다.


---

This project was completed as a midterm replacement project for a Digital Image Processing course.
