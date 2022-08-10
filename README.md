📋 Introduction
---
- 이미지 분류 딥러닝을 구현하고자 했으며, 최대한의 Accurary를 뽑아내자라는 목표를 잡았음

# 랜드마크 이미지 분류
## 개요
제로베이스에서 배운 Deep Learning 강의 기반으로 이미지 분류에 도전해보기 위해 
데이콘 경진대회를 연습삼아 도전합니다.

[데이콘/ 랜드마크 분류 AI 경진대회](https://dacon.io/competitions/official/235585/overview/description)


## 데이터 수집
대회에서는 데이콘에서 제공한 데이터가 있었지만 현재는 다운로드가 불가능함. Q&A를 살펴본 결과 AI Hub에서 다운로드가 가능한것으로 확인됨

[AI Hub/ 랜드마크 이미지](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=56)

전체 데이터는 12TB로 학습하는 입장에서 다루기에 크기 때문에 가장 작은 지역단위인 세종시 데이터만 활용

- 총 48 GB
- 사진크기: 4032 x 3024
- Class: 84개
- Training(12396장)과 Validation(1504장)이미지가 나누어져 있음.


## 🎯 Result
Train과 Validation Accuracy가 매우 높게 나옴.

![Result](https://user-images.githubusercontent.com/100823210/183580705-a1af4afb-6608-4389-b921-3e8f287cb751.png)

일부 향교나 서원와 같은 이미지는 잘 분류하지 못함.

![actual pred](https://user-images.githubusercontent.com/100823210/183580924-f71cab66-f252-409b-bccf-3e5376cf1677.png)


## 전처리
### 사전작업
Google Colab에서는 수집된 데이터의 업로드가 힘들기때문에 Local 컴퓨터에서 OpenCV를 이용하여 데이터 크기를 원본크기의 0.1배로 줄여줌.
```python
import cv2

img = cv2.imread(os.path.join(region_dir, cls, fname))
resized_img = cv2.resize(img, (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
cv2.imwrite(os.path.join(target_dir, fname), resized_img)
```

### Transform
#### Resize
Resize 크기에 따른 Accuracy 측정을 위해 아래 3가지 값으로 변경하며 수행함.
- 256x256, 128x128, 64x64

#### Normalize
이미지넷의 mean, std값을 사용해도 괜찮았지만 좀 더 Accuracy를 올리기 위해 아래 함수를 활용하여 데이터의 Normalize 값을 찾음.
```python
def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std
```

### Dataset & DataLoader
폴더의 구조가 ImageFolder를 사용하기 쉽게 구성되어 있으므로 ImageFolder와 DataLoader를 사용하여 구성함.
```python
image_datasets = {x: ImageFolder(root=os.path.join(data_dir, x),
                                 transform=transform[x]) for x in transform.keys()}
dataloaders = {x: DataLoader(image_datasets[x],
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             num_workers=2) for x in transform.keys()}
```

## 🔎 Modeling
### 전이학습
Resnet50을 이용한 전이학습을 계획함. (requires_grad = False는 conv3_x까지 적용)
![Resnet50](https://user-images.githubusercontent.com/100823210/183578724-b8298ea1-5336-4580-99b0-1c6109194491.png)

#### Resnet50을 선택한 이유
1. 학습 난이도가 매우 낮아진다
2. 깊이가 깊어질수록 높은 정확도 향상을 보임
3. 많은 수의 Layer를 누적하여 깊은 Network를 설계할 때 여러 문제가 발생하는 CNN문제를 보완

### Sweeps ( weights & Biases)
web에서 결과에 대한 시각화 기능을 지원하기에 sweeps를 통한 Accurary와 Loss function 시각화

### 옵티마이저
Adam을 사용

### 스케쥴러
7번째 Epoch마다 0.1배씩 낮아지게 설정
```python
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
```

## 성능 개선
최적의 파라미터 튜닝값을 찾기위해 Weights & Biases의 Sweeps 기능을 이용하여 다양한 변수를 변경하여 반복 수행함.

[Hyperparameter-Sweeps](https://wandb.ai/zbooster/Hyperparameter-Sweeps?workspace=user-zbooster)

## ⚙️ Limitations
- Accuracy가 너무 높음. 
    - 100%의 Accuracy가 나오기도 했음.
        - 데이터의 양이 적어져서 이런가 의문점
- 데이터 특성으로 인한 과적합 문제가 발생하지 않았나?

## 📖 Reference
- 마이크로소프트 연구원 Kaiming He 외 3인(2015), Deep Residual Learning for Image Recognition
    - https://arxiv.org/pdf/1512.03385.pdf
