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

## 모델링
Resnet50을 이용한 전이학습을 계획함.


## 성능 개선
(작성중...)

