 📋 소개
---
![1](https://user-images.githubusercontent.com/104749023/183792578-7aac5169-8e8d-4af9-afd3-0da508003fd9.PNG)


- Resnet50을 활용해 이미지 분류 딥러닝을 했습니다. 


# 랜드마크 이미지 분류
## 개요
Deep Learning 기반으로 이미지 분류에 도전해보기 위해 데이콘 경진대회를 참여했습니다.

[데이콘/ 랜드마크 분류 AI 경진대회](https://dacon.io/competitions/official/235585/overview/description)


## 데이터 수집
대회에서는 데이콘에서 제공한 데이터가 있었지만 현재는 다운로드가 불가능함. Q&A를 살펴본 결과 AI Hub에서 다운로드가 가능한것으로 확인했습니다.

[AI Hub/ 랜드마크 이미지](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=56)

전체 데이터는 12TB로 학습하는 입장에서 다루기에 크기 때문에 가장 작은 지역단위인 세종시 데이터만 활용했습니다.

- 총 48 GB
- 사진크기: 4032 x 3024
- Class: 84개
- Training(12396장)과 Validation(1504장)이미지가 나누어져 있습니다.


## 🎯 결과

|Accuracy|pyTorch|Resnet50|
|---|------|---|
|Train|99.85%|99.42%|
|Validation|85.17%|96.68%

![Result](https://user-images.githubusercontent.com/100823210/183580705-a1af4afb-6608-4389-b921-3e8f287cb751.png)

일부 향교나 서원와 같은 이미지는 잘 분류하지 못하는 모습을 확인할 수 있습니다.

![actual pred](https://user-images.githubusercontent.com/100823210/183580924-f71cab66-f252-409b-bccf-3e5376cf1677.png)


## 전처리
### 사전작업
Google Colab에서는 수집된 데이터의 업로드가 힘들기때문에 Local 컴퓨터에서 OpenCV를 이용하여 데이터 크기를 원본크기의 0.1배로 줄였습니다.
```python
import cv2

img = cv2.imread(os.path.join(region_dir, cls, fname))
resized_img = cv2.resize(img, (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
cv2.imwrite(os.path.join(target_dir, fname), resized_img)
```

### Transform
#### Resize
Resize 크기에 따른 Accuracy 측정을 위해 아래 3가지 값으로 변경하며 수행했습니다.
- 256x256, 128x128, 64x64

#### Normalize
이미지넷의 mean, std값을 사용해도 괜찮았지만 좀 더 Accuracy를 올리기 위해 아래 함수를 활용하여 데이터의 Normalize 값을 찾았습니다.
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

|Dataset & DataLoader|설명|
|------|---|
|List처럼 직접 index하는 방법|직접 작성할 수도 있어서 대량의 이미지 파일을 한 번에 메모리에 저장하지 않고, 필요할 때마다 읽어서 학습하는듯|
|DataLoader를 통해 순회하기|학습을 진행하는 for문에 사용되었으며 generator 사용 되어 각 iteration 마다 batch size 만큼 가져와서 사용하는 것|


ImageFolder와 DataLoader를 사용했습니다.
    - 폴더의 구조가 ImageFolder를 사용하기 쉽게 구성되어있습니다.
    
    
    
```python
image_datasets = {x: ImageFolder(root=os.path.join(data_dir, x),
                                 transform=transform[x]) for x in transform.keys()}
dataloaders = {x: DataLoader(image_datasets[x],
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             num_workers=2) for x in transform.keys()}
```

## 🔎 모델링
### 전이학습
Resnet50을 이용한 전이학습을 계획했습니다. (requires_grad = False는 conv3_x까지 적용)
![Resnet50](https://user-images.githubusercontent.com/100823210/183578724-b8298ea1-5336-4580-99b0-1c6109194491.png)

#### Resnet50을 사용한 이유
1. 학습 난이도가 매우 낮아집니다.
2. 깊이가 깊어질수록 높은 정확도 향상을 보입니다.
3. 많은 수의 Layer를 누적하여 깊은 Network를 설계할 때 여러 문제가 발생하는 CNN문제를 보완합니다.

### Sweeps ( weights & Biases)
web에서 결과에 대한 시각화 기능을 지원하기에 sweeps를 통한 Accurary와 Loss function 시각화

### 옵티마이저
Adam을 사용했습니다.

### 스케쥴러
7번째 Epoch마다 0.1배씩 낮아지게 설정했습니다.
```python
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
```

## 성능 개선
최적의 파라미터 튜닝값을 찾기위해 Weights & Biases의 Sweeps 기능을 이용하여 다양한 변수를 변경하여 반복 수행했습니다.

[Hyperparameter-Sweeps](https://wandb.ai/zbooster/Hyperparameter-Sweeps?workspace=user-zbooster)

## ⚙️ 의문점
Ephochs를 돌리면서 Train Accuracy가 100%가 나오는 경우를 봤습니다.
    - Accuracy가 100%가 나오면 데이터를 한번 의심해야 합니다.
        - 세종시 데이터로 축소하면서 데이터가 overfitting된 부분은 없는지 확인할 예정입니다.
        
## 📖 관련 자료
- 마이크로소프트 연구원 Kaiming He 외 3인(2015), Deep Residual Learning for Image Recognition
    - https://arxiv.org/pdf/1512.03385.pdf
- AI 연구원 Aroddary, (ResNet) Deep residual learning for image recognition 번역 및 추가 설명과 Keras 구현
    - https://sike6054.github.io/blog/paper/first-post/
    
## 🤲 팀원 소개 
|팀원|연락|
|------|---|
|김송현|[G.mail](zpaladin1213@gmail.com) │ [Velog](https://velog.io/@zbooster)|
|김해솔|[G.mail](lunchtime99@gmail.com) │ [Velog](https://velog.io/@kim_haesol)|
