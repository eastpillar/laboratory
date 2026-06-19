# Portfolio
<hr style="border: 1px solid gray;"></hr>
안동주

email : adj1001@naver.com

tel : +82 10-3825-5246

<hr style="border: 1px solid gray;"></hr>

# Intro
>안녕하세요! "***신기술을 배우고 도전하며 활용하고 싶은***" 안동주입니다!<br>
>1년간 computer vision 인공지능 연구실에서 학부연구생으로 vision 인공지능 기초에 대해 공부했습니다.<br>
>기초 모델부터 멀티모달까지 논문을 읽고 분석하여 저자가 설명하는 모델을 코드로 구현한 경험이 있습니다.<br>
>각 모델들을 구현 및 실험을 진행하고, 연구실 팀원들과 교수님에게 설명하는 형식의 세미나 발표를 진행한 경험이 있습니다<br>
>[![PPT](https://img.shields.io/badge/PPT-세미나%20발표자료%20살펴보기-df2a2a.svg?style=for-the-badge)](./Seminar_presentation_slides)

<hr style="border: 1px solid gray;"></hr>

# Project
>모델 구현 순서는 논문 게재일 순으로 다음과 같습니다. <br>
>`VGG` ➔ `ResNet` ➔ `DenseNet` ➔ `CBAM` ➔ `FCN` ➔ `U-Net` ➔ `ViT` ➔ `DINOv2` ➔ `OpenVLA, ETRI` ➔ **`OpenVLA-ECoT`**  

## 1. Image Classification 모델 구현

- 구현 모델 : `VGG`, `ResNet`, `DenseNet`, `CBAM`, `ViT`, `DINOv2`
- 핵심 역할 : 단독 개발, 200개의 정답 Class로 구성된 데이터셋으로, 훈련을 진행하는 207,005장의 학습 이미지와 성능 평가를 진행하는 51,752장의 평가 이미지를 사용했습니다.<br>
각 평가 이미지가 고양이, 자동차, 동물 등 어떤 정답에 속하는지 예측하는 인공지능 모델입니다.


- Language : `Python`
- Tool : `Pytorch`, `einops`, `numpy`, `opencv-python`, `pillow`, `tqdm`<br>
[![Lib](https://img.shields.io/badge/Lib-전체%20라이브러리%20살펴보기-2ea043.svg?style=for-the-badge)](./requirements.txt)
- 모델 코드 및 상세 설명<br>
[![VGG](https://img.shields.io/badge/VGG-ffaa00.svg?style=for-the-badge&logoColor=white&colorB=ffaa00)](./VGG)
[![ResNet](https://img.shields.io/badge/ResNet-ffb300.svg?style=for-the-badge&logoColor=white&colorB=ffaa00)](./ResNet)
[![DenseNet](https://img.shields.io/badge/DenseNet-ffaa00.svg?style=for-the-badge&logoColor=white&colorB=ffaa00)](./DenseNet)
[![CBAM](https://img.shields.io/badge/CBAM-ffaa00.svg?style=for-the-badge&logoColor=white&colorB=ffaa00)](./CBAM)
[![ViT](https://img.shields.io/badge/ViT-ffaa00.svg?style=for-the-badge&logoColor=white&colorB=ffaa00)](./ViT)
[![DINOv2](https://img.shields.io/badge/DINOv2-ffaa00.svg?style=for-the-badge&logoColor=white&colorB=ffaa00)](./DINOv2/Classification_task)

## 2. Semantic Segmentation 모델 구현

- 구현 모델 : `FCN`, `U-Net`, `DINOv2`
- 핵심 역할 : 단독 개발, 21개의 정답 Class로 구성된 데이터셋으로, 훈련을 진행하는 이미지는 데이터 증강을 통해 1,464장에서 10,582장으로 구성하고 성능 평가를 진행하는 1,464장의 평가 이미지를 사용했습니다.<br>
256x256으로 구성된 이미지의 각 픽셀이 어떤 정답 Class에 속하는지 예측하는 인공지능 모델입니다.<br>
256x256으로 구성된 하나의 이미지마다 65,536개의 정답을 예측하는 모델입니다.


- Language : `Python`
- Tool : `Pytorch`, `einops`, `numpy`, `opencv-python`, `pillow`, `tqdm`<br>
[![Lib](https://img.shields.io/badge/Lib-전체%20라이브러리%20살펴보기-2ea043.svg?style=for-the-badge)](./requirements.txt)<br>
- 모델 코드 및 상세 설명<br>
[![FCN](https://img.shields.io/badge/FCN-ffaa00.svg?style=for-the-badge&logoColor=white&colorB=ffaa00)](./FCN)
[![U-Net](https://img.shields.io/badge/UNet-ffaa00.svg?style=for-the-badge&logoColor=white&colorB=ffaa00)](./U-Net)
[![DINOv2](https://img.shields.io/badge/DINOv2-ffaa00.svg?style=for-the-badge&logoColor=white&colorB=ffaa00)](./DINOv2/Segmentation_task)

## 3. ETRI 연구 과제

- 구현, 활용 모델 : `MTCNN`, `TokenHPE`, `Transformer Model`
- 핵심 역할 : 팀원, 카메라 기준 정면을 바라보고 있는 영유아의 동영상으로 구성된 데이터셋으로, 
보호자, 감독관이 같이 찍힌 영상에서 아이의 얼굴을 감지하는 알고리즘을 구현하고 아이가 바라보는 방향을 모델로 예측하여
아이의 목표 행동 여부에 따라 자폐 스펙트럼 장애 여부를 예측하는 모델입니다.

- Language : `Python`
- Tool : `pytorch`, `facenet-pytorch`, `einops`, `huggingface-hub`, `numpy`, `opencv-python`, `pandas`, `pillow`, `scipy`, `seaborn`, `tifffile`<br>
[![Lib](https://img.shields.io/badge/Lib-전체%20라이브러리%20살펴보기-2ea043.svg?style=for-the-badge)](./ETRI/requirements.txt)<br>
- 모델 코드 및 상세 설명<br>
[![TokenHPE](https://img.shields.io/badge/TokenHPE-ffaa00.svg?style=for-the-badge&logoColor=white&colorB=ffaa00)](./ETRI/TokenHPE2)
[![Transformer model](https://img.shields.io/badge/Transformer_model-ffaa00.svg?style=for-the-badge&logoColor=white&colorB=ffaa00)](./ETRI/etri_transformer_model)

## 4. OpenVLA-ECoT 모델 구현

- 활용 모델 : `OpenVLA`
- 핵심 역할 : 단독 개발, 여러 시나리오로 구성된 로봇 제어 데이터셋 LIBERO를 사용하여 진행했습니다.<br>
입력된 시각 및 언어 정보를 바탕으로 기존 학습 데이터에만 의존해 즉각적인 행동을 출력하는 대신, 학습되지 않은 새로운 시나리오(Unseen)에서도 사전 추론 과정을 거쳐 정확한 로봇 제어값을 예측하는 인공지능 모델입니다.<br>

- Language : `Python`
- Tool : `pytorch`, `einops`, `imageio`, `keras`, `numpy`, `opencv-python`, `openvla`, `pillow`, `scipy`, `tokenizers`, `tqdm`<br>
[![Lib](https://img.shields.io/badge/Lib-전체%20라이브러리%20살펴보기-2ea043.svg?style=for-the-badge)](./OpenVLA-ECoT/OpenVLA-ECoT_Requirements/requirements.txt)<br>
- 모델 코드 및 상세 설명<br>
[![OpenVLA](https://img.shields.io/badge/OpenVLA-ffaa00.svg?style=for-the-badge&logoColor=white&colorB=ffaa00)](./openvla)
[![OpenVLA-ECoT](https://img.shields.io/badge/Openvla_ECoT-ffaa00.svg?style=for-the-badge&logoColor=white&colorB=ffaa00)](./OpenVLA-ECoT)