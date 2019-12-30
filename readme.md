## Pavement Crack Segmentation

Created by glee1228@naver.com

Please leave your questions at donghoon.rhie@gmail.com I'll spend as much time as I can to give you a friendly answer.

## Data

* data 디렉토리 아래 
  * crack 디렉토리와 pot 디렉토리에 train, val, test로 데이터셋 구성

* 256 x 256 단위 패치
* 크랙이 없는 부분은 제거

## Requirements
```
python==3.6
Pillow
matplotlib
numpy==1.14.5
opencv-python
pandas
torch==0.3.1
torchvision==0.2.1
scipy
tqdm
scikit-image
torchtext==0.2.3
```

## Usage


학습 데이터 다운로드

### 공유한 구글 드라이브 링크에서 data-MUHAN.zip 다운로드

코드 다운로드

```
git clone https://github.com/glee1228/segmentation
```

Requirements.txt 설치
```
pip3 install -r requirements.txt
```


학습 데이터 위치 설정

```
$ mkdir data; cd data
$ mv 원래 data-MUHAN.zip경로 .
$ unzip data-MUHAN.zip
$ rm data-MUHAN.zip
$ cd ..
```

학습 초기화를 위해 pretrained VGG16 모델 다운로드
```
$ cd model/
$ wget https://download.pytorch.org/models/vgg16-397923af.pth
$ mv vgg16-397923af.pth vgg16.pth
$ cd ..
```
### Train

```
python train.py
```

path 설정 및 에폭 설정 후 실행



### Inference

```
python infer.py
```

path 설정 및 에폭 설정 후 실행


학습 중 validation 데이터는 train 폴더 안에 생성 -> 확인하면서 학습데이터 구성 조절

## Directory configuration
```
├── segmentation
   ├── train.py
   ├── trainer.py
   ├── util.py
   ├── dataproc.py
   ├── model.py
   ├── infer.py
   ├── crop.py
   ├── mask_proc.py
   ├── measure.py
   ├── merge.py
   ├── ransac.py
   ├── severity.py
   ├── model
       ├── vgg16.pth
   ├── train
       ├── images0
       └── HED0.pth
   ├── output
       └── pred
   └── data
       ├── crack
           ├── train
                ├── croppedimg
                └── croppedgt
           ├── val
                ├── croppedimg
                └── croppedgt
           └── test
                ├── croppedimg
                └── croppedgt
       └── pot
           ├── train
                ├── croppedimg
                └── croppedgt
           ├── val
                ├── croppedimg
                └── croppedgt
           └── test
                ├── croppedimg
                └── croppedgt
```





## HED Network Description

![holistically-nested edge detection](Image/HED_1.png)

구조는 위의 이미지와 같다.

네트워크의 아이디어는 scale pyramid를 통해 좋은 하나의 결과를 만들자이다.

여러개의 receptive field size로 convolution을 하는데, 병렬적으로 하지 않고, 직렬로 한다.

따라서, 첫번째에서 뽑은 feature를 토대로 두 번째, 세 번째 차례로 점점 더 중요하게 여길 부분을 공략하게 만든다.

Receptive field가 점점 커지기 때문에, 레이어마다 출력 feature 크기는 작지만, 선형 보간을 이용해 입력 이미지와 동일한 크기로 키워서 차곡차곡 겹치게 한 후, convolution을 시행하여 최종 결과물을 만든다.



각 레이어의 feature마다 sigmoid함수 출력 값(0~1)을 합하여 back-prop한다.



![holistically-nested edge detection](Image/HED_2.png)

HED의 성능 평가 결과는 해당 논문에서 제시한 표에서 확인할 수 있다.

