# 2023 Samsung AI Challenge : Camera-Invariant Domain Adaptation
## 1. 개요
https://dacon.io/competitions/official/236132/overview/description
  - 주제 : 카메라 특성 변화에 강인한 Domain Adaptive Semantic Segmentation 알고리즘 개발
  - Task : Semantic Segmentation, Unsupervised Domain Adaptation
  - 기간 : 2023.08.21 ~ 2023.10.02
  - 결과 : 20등 / 212

## 2. 데이터셋 설명
- train_source_image (폴더) : 왜곡이 존재하지 않는 train 이미지(Source Domain)
![Train 0001](https://github.com/jang3463/samsung_ai/assets/70848146/f3cf5886-e8d1-4abc-982b-46f166504a89)
<img width="800" height="400" alt="image" src="https://github.com/jang3463/samsung_ai/assets/70848146/f3cf5886-e8d1-4abc-982b-46f166504a89">
- train_source_gt (폴더) : 왜곡이 존재하지 않는 train 이미지(Source Domain)의 mask
![output2](https://github.com/jang3463/samsung_ai/assets/70848146/af92d2bf-035d-4f8d-ac11-52b2102aba72)
- train_target_image (폴더) : 왜곡된 이미지(Target Domain) => 200도의 시야각(200° F.O.V)을 가지는 어안렌즈 카메라로 촬영된 이미지
![TRAIN_TARGET_0007](https://github.com/jang3463/samsung_ai/assets/70848146/18bcbec4-120f-424c-9230-a0a31ac82fb8)
- val_source_image (폴더) : 왜곡이 존재하지 않는 validation 이미지(Source Domain)
- val_source_gt (폴더) : 왜곡이 존재하지 않는 validation 이미지(Source Domain)의 mask
- test_image (폴더) : 추론해야하는 이미지(Target Domain)
![TEST_0001](https://github.com/jang3463/samsung_ai/assets/70848146/8d67e571-c3a1-4d94-8955-0105571bc195)

## 3. 수행방법
- 본 과제는 왜곡이 존재하지 않는 이미지(Source Domain)와 레이블을 활용하여, 왜곡된 이미지(Target Domain)에 대해서도 고성능의 이미지 분할(Semantic Segmentation)을 수행하는 AI 알고리즘 개발을 개발하는 것
- 먼저 Unsupervised Domain Adaptation 방법 중 SOTA인 [HRDA(Context-Aware High-Resolution Domain-Adaptive Semantic Segmentation)](https://github.com/lhoyer/HRDA) 적용
- 하지만 위 방법은 이미지의 광도와 질감(Photometry and Texture)의 Domain gap에서는 잘 작동하지만 이미지의 광학 왜곡적인 부분의 Domain에서는 잘 작동하지 않음
- 이 부분을 해결하기 위해 Domain adaptation을 적용하기 전, Opencv를 사용해 Affine 변환, LensDistortion 변환을 이미지에 적용하여 성능을 향상시킴
- 최종적으로 LB miou 0.58739 달성
- 예측 결과 example
![mask_1](https://github.com/jang3463/samsung_ai/assets/70848146/fcceeefe-248d-4929-996c-63503deb7068)

## Team member
장종환 (개인 참가)
