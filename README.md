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
- train_source_gt (폴더) : 왜곡이 존재하지 않는 train 이미지(Source Domain)의 mask
![output2](https://github.com/jang3463/samsung_ai/assets/70848146/af92d2bf-035d-4f8d-ac11-52b2102aba72)
- train_target_image (폴더) : 왜곡된 이미지(Target Domain) => 200도의 시야각(200° F.O.V)을 가지는 어안렌즈 카메라로 촬영된 이미지
![TRAIN_TARGET_0007](https://github.com/jang3463/samsung_ai/assets/70848146/18bcbec4-120f-424c-9230-a0a31ac82fb8)
- val_source_image (폴더) : 왜곡이 존재하지 않는 validation 이미지(Source Domain)
- val_source_gt (폴더) : 왜곡이 존재하지 않는 validation 이미지(Source Domain)의 mask
- test_image (폴더) : 추론해야하는 이미지(Target Domain)
![TRAIN_TARGET_0007](https://github.com/jang3463/samsung_ai/assets/70848146/18bcbec4-120f-424c-9230-a0a31ac82fb8)

## 3. 수행방법
- 본 과제는 블랙박스 영상으로부터 자동차의 충돌 상황을 분석하는 AI 모델을 개발하는 것
- 본 데이터의 LABEL는 위의 이미지처럼 분할이 가능함
- label을 crash, weather, timing으로 분할하여 multi_label_classification 문제로 변환
- 모델로는 slowfast_r101, MVITv2_B_32x3, r3d_18을 사용해본 결과, r3d_18이 성능이 가장 좋았음
- 최종적으로 carsh는 r3d_18 모델 사용, weather, timing은 영상에서 각 5개의 이미지를 random으로 추출하고 convnext_large 모델 사용
- 각 label별로 병렬적으로 임베딩 추출하여 concat
- 최종적으로 LB miou 0.58739 달성
- 예측 결과 example
![mask_1](https://github.com/jang3463/samsung_ai/assets/70848146/fcceeefe-248d-4929-996c-63503deb7068)

## 4. 한계점
- 데이터 특성상 클래스 불균형도 심하고 잘못 labeling 된 데이터도 포함되어 있어서 쉽지 않은 대회였음

## Team member
장종환 (개인 참가)
