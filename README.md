# Image Processing Projects

영상 신호 처리를 학습하며 수행한 세 개의 프로젝트를 정리한 저장소다.

전통적인 영상 대비 향상 기법의 비교부터 딥러닝 기반 무선 신호 분류, 손동작 인식을 활용한 게임 구현까지 서로 다른 형태의 영상 신호 처리 문제를 다룬다.

## Projects

| Project                                                                          | Description                                                                             | Main Technologies                 |
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | --------------------------------- |
| [Image Contrast Enhancement Comparison](./Image_Contrast_Enhancement_Comparison) | HDR, Histogram Equalization, CLAHE 및 컬러 영상 확장 기법의 결과를 비교한 프로젝트다.                        | Python, Image Processing          |
| [Automatic Modulation Classification](./Automatic_Modulation_Classification)     | RadioML I/Q 신호를 성상도 이미지로 변환하고 MobileNetV3 Small과 ResNet-18의 변조 분류 성능과 연산 효율을 비교한 프로젝트다. | Python, PyTorch, Deep Learning    |
| [Hand Gun Fruit Shooter](./fruit_shooting)                                       | MediaPipe 기반 손동작 인식과 웹캠 영상을 이용하여 과일을 조준하고 발사하는 게임을 구현한 프로젝트다.                           | Python, OpenCV, MediaPipe, Pygame |

## Repository Structure

```text
Image_Processing/
├─ Image_Contrast_Enhancement_Comparison/
├─ Automatic_Modulation_Classification/
├─ fruit_shooting/
└─ README.md
```

각 프로젝트의 구현 내용, 실행 방법 및 결과는 해당 폴더의 README에서 확인할 수 있다.
