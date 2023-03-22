# smart_factory_car_paint
자동차의 스크래치를 마스크 처리하여 보여주는 어플
## 개요
![현대차_도색_결함_2021_04_05](https://user-images.githubusercontent.com/125535111/226703137-a52f9981-ffd4-4f60-8439-079974f9d810.png)
### 기획 의도
- 완성차의 소비자가격은 보통 수천을 호가한다. 도색불량, 차체불량으로 인해 소비자가 무상수리나 환불을 청구한다면 그 피해액은 모두 회사에서 부담해야한다. 
소비자들은 현기차(현대/기아차)의 도색불량에 대해 꾸준히 문제를 제기해 오고 있다.
- 자동차의 공정은 프레스 >> 차체조립 >> 도장 >> 의장 >> 검수 순이다. 도색불량은 도장중, 도장후 출고전, 출고 후 소비자 배송전까지의 과정에서 생겨날 수 있다. 도색불량을 도장과정 직후 바로 잡아낼 수 있다면 도색에 대한 무상수리비용은 기하급수적으로 줄일 수 있을 것이다. 
- 또한, 도장 품질 검수 프로그램은 잦은 도색불량으로 인해 하락된 소비자들의 기업 이미지를 복구할 수 있으며 공장의 스마트팩토리화의 초석이 될 것이다.
### 타겟 선정
- 국내외의 완성차 업체들
  - 현대/기아차, 도요타, 테슬라, 독일3사(벤츠, BMW, Audi) 등
### 시장 분석
- 도장검사지 딥러닝 스캐닝 인식 기술은 자동차 도장면 검사 공정에 인공지능 기술을 접목해 빅데이터를 구축하는 것으로, 도장의 품질 수준을 높일 수 있다.
- AIR Lab은 생산산개발본부와 협업하여 고장검사지에 적힌 검사시간, 차종, 이상유형, 이상발생위치 등의 정보를 빠르게 추출해 데이터베이스로 저장하는 딥러닝 알고리즘을 개발했고, 이를 통해 도장 공정에서 자주 발생하는 문제나 특정 차종에 반복해서 나타나는 문제를 빠르게 파악해 대응책을 마련하고 있다
- 현재 울산공장 생산라인에서 2019년 12월까지 시범적용한 결과 일평균 400장의 도장검사지를 스캔했으며 약 95%의 검출 정확도를 보였다. 데이터 베이스가 충분히 구축되면 결함률이 낮아지며 전반적인 품질향상에 기여할 것이기 때문에에, 향후 다른 공장과 생산 공정에도 딥러닝 스캐닝 인식 기술을 확대 적용할 계획이라고 한다.
## 목표
-  자동차 부품 공장 도장 불량품에 대한 검출율 95% 이상 달성하는 모델 구축하여 국내외 완성차 업체들에게 도장 검수 sw를 제공하는 것
- 품질검사 sw 도입에 대한 현대차 년간 기대수익
  - = 도색에대한 소비자 부담가 x 1년간 자동차 판매량 x 현불량률(예측 1/1000*) x 모델정확도(목표)
  - = (25+40)/2만원 x 3,890,726 x 0.001 x 0.95
  - = 1,201,261,652.5원 이상
## 데이터
### 사용 데이터
- 데이터 출처 : [AIhub](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=578) 중 도장 스크래치 데이터
- 사용 데이터
  - 부품 품질 검사 영상 데이터(자동차) 불량품 5630개
### 데이터 전처리
1. 정상품+불량품 => 정상품/불량품 분류
2. label data => coco data => mask data
3. image data와 mask data에 Min-Max normalization 적용
4. image data는 RGB / mask data를 grayscale로 처리
5. input image size = (128,128)로 resize함
## 모델링
### U-net
![u-net-architecture](https://user-images.githubusercontent.com/125535111/226705327-eca53949-7338-424c-b20f-c254dda8cdab.png)
출처 : <uni-freiburg ( https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/ )>
- U-Net이란 ?
  - U-Net은 생물의학 분야에서 이미지 분할(Image Segmentation)을 목적으로 제안된 End-to-End 방식의 Fully-Convolutional Network 기반 모델입니다. 네트워크 구성의 형태가 'U' 모양으로 U-Net 이름이 붙여졌습니다.
#### 장단점 및 선정 이유
- 장점
  - 적은 양의 학습 데이터로도 Data Augmentation을 활용해 여러 바이오 메디컬 이미지 분할 문제에서 우수한 성능을 보임
  - 컨텍스트 정보를 잘 사용하면서도 정확히 지역화함
    - 맥스풀링 전 데이터를 카피하여 업-콘볼루젼 시 참조하기 때문이다. 
  - End-to-End 구조로 속도가 빠름
  - 속도가 빠른 이유: 검증이 끝난 곳은 건너뛰고 다음 Patch부터 새 검증을 하기 때문.
  - 용어정리
    - Data Augmentation : 이미지를 다양한 각도 형태로 왜곡하여 변형된 이미지를 만든다. 변형시킨 데이터는 학습시 새로운 이미지로 인식한다.
    - End-to-End : 입력에서 출력까지 파이프라인 네트워크없이 한번에 처리하는 것을 말한다. 예를들면 텍스트를 학습시킨다고한다면, 형태소 -> 인코딩 -> ... -> 디코딩 등 여러가지를 거쳐야하지만, End-to-End는 그 과정을 생략하고 바로 텍스트를 바로 출력한다.
- 단점
  - 어두운 이미지의 경우 학습이 어려움
- 선정 이유
  - ANOGAN을 사용하면  학습 데이터의 형태가 일정한면이 아닌 여러방면에서 찍힘
  - Crack과 같은 균열을 찾는데 주로 이용됨
### FCN과 비교
![fcn비교](https://user-images.githubusercontent.com/125535111/226705768-193310e3-5a41-4d75-8414-d1e7867ed4a1.png)
- shortcut
  - layer가 깊어짐에 따라 vanishing gradient 현상이 나타나는데 shortcut을 통해 이전feature를 다시 넣어줌으로 방지함
- upsampling
  - feature map을 생성하여 이미지의 특성을 더 잘 잡아낼 수 있게함
### 모델 학습
- 학습 데이터
  - 총 이미지 데이터 5630개
    - 훈련 데이터 5010개 + 검증 데이터 570개 + 테스트 데이터 50개
  - optimizer : adam
  - loss function : cross entropy
  - batch size : 8
  - epoch : 30
  - training time : 약 3시간
- 개발 스펙
  - tensorflow : 2.11.0
  - Flask : 1.1.2
  - Python : 3.10.9
### 평가 지표
![평가지표](https://user-images.githubusercontent.com/125535111/226706000-6fd19ba2-d58e-431d-8193-f8b661361077.png)
- Train accuracy = 93.1%
- Validation accuracy = 91.9%

![평가지표 손실함수](https://user-images.githubusercontent.com/125535111/226706159-9b110184-a153-4bc7-a9dd-59871d9b1448.png)
- Train loss = 0.09
- Validation loss = 0.22
## 코드
###데이터 준비
#### 양품, 불량품 분리
- AI 허브에서 데이터 받을 시 사용
  - 불량품과 양품이 섞여있어 분리해줘야함
  - 경로설정 해줘야함
  - 코랩 사용시 미리 이동시킬 폴더를 만들어 줘야함
```Ruby 
      import glob
      import json
      import shutil

      def move_good_quality_file_dir( label_cur_path, label_befo_path, data_cur_path, data_befo_path) :
          '''
              사용하기 전 양품 데이터 디렉토리와 라벨 양품 데이터 디렉토리를 만들어주세요
              그리고 밑에서 경로를 설정해주세요
          '''

          files_name = list()

          for file_path in glob.glob(label_cur_path) :
              with open( file_path, mode='r', encoding='UTF-8-sig') as f:
                  data = json.load(f)
                  data_quli = data['annotations'][0]['attributes']['quality']

              if data_quli == '양품' : 
                  files_name.append(data['images'][0]['file_name'])
                  shutil.move(file_path , label_befo_path)           

          for file_name in files_name :
              print('file_name : ' ,file_name)
              shutil.move(data_cur_path+file_name , data_befo_path)


      label_cur_path  = './' # json label 경로 설정  ex) ./car_check/data/Training/label/door/scratch/*.json
      label_befo_path = './' # 이동 전 라벨 경로     ex) ./car_check/data/Training/label/door/scratch_pass
      data_cur_path   = './' # 이동 전 데이터 경로   ex) ./car_check/data/Training/data/door/scratch/
      data_befo_path  = './' # 이동 후 데이터 경로   ex) ./car_check/data/Training/data/door/scratch_pass

      move_good_quality_file_dir( label_cur_path, label_befo_path, data_cur_path, data_befo_path)

  ```
#### 압축파일 업로드 및 압축해제
- 전체데이터를 사용하고 싶다면 구글드라이브에 업로드 후 경로설정하고 사용해주세요
- 실습은 코랩에서 사용가능
# 알집 선택
```
from google.colab import files
files.upload()
```
#### 코코 라벨 준비
- 기존 label을 coco label 형식으로 변경
```
import glob
import json

def conv_coco_label(path, save_path, label_file_name='coco_label.json') :
  files = glob.glob(path)

  image_list = list()
  annotations_list = list()
  id = 0

  for idx, file in enumerate(files) :
    with open(file, mode='r', encoding='utf-8-sig') as f :
      file_json = json.load(f)
      file_images = file_json['images'][0]
      file_annotations = file_json['annotations']
      
      image_list.append({
            "id":idx,
            "license":1,
            "file_name":file_images['file_name'],
            "height":file_images['height'],
            "width":file_images['width'],
            "date_captured":file_images['date_captured']
      })

      for file_annotation in file_annotations :
        annotations_list.append({
            "id":id,
            "image_id":idx,
            "category_id":1,
            "bbox":file_annotation['bbox'],
            "area":file_annotation['area'],
            "segmentation": [file_annotation['segmentation']],
            "iscrowd":0
            })

        id += 1

  form = make_form(image_list, annotations_list)
  json_save(form, save_path, label_file_name)


def make_form(image_list, annotations_list) :
  return {
      "info":{
          "year":"2023",
          "version":"1",
          "description":"Exported from roboflow.ai",
          "contributor":"",
          "url":"https://public.roboflow.ai/object-detection/undefined",
          "date_created":"2023-03-13T01:51:04+00:00"
      },
      "licenses":[{
          "id":1,
          "url":"https://creativecommons.org/publicdomain/zero/1.0/",
          "name":"Public Domain"
      }],
      "categories":[{
          "id":0,
          "name":"car-scratch",
          "supercategory":"none"
      },
      {
          "id":1,
          "name":"car-scratch",
          "supercategory":"car-scratch"
      }],
      "images"     : image_list,
      "annotations": annotations_list
    }

def json_save(form, save_path,label_file_name) :
  print(save_path + label_file_name)
  with open( save_path + label_file_name, 'w', encoding='utf-8') as f:
    json.dump(form, f)
```
```
%cd car_check
```
```
path_train = './scratch/training/label/*.json'
path_valid = './scratch/Validation/label/*.json'
path_test  = './scratch/test/label/*.json'

save_path_train = './scratch/training/coco_label/'
save_path_valid = './scratch/validation/coco_label/'
save_path_test  = './scratch/test/coco_label/'

conv_coco_label(path_train, save_path_train )
conv_coco_label(path_valid, save_path_valid )
conv_coco_label(path_test , save_path_test  )
```
#### 기타 등등 준비
```
%cd cocoapi
!2to3 . -w
%cd PythonAPI
!python3 setup.py install -y
!pip install -q tensorflow
!pip install -q keras
!pip install -q utils
%cd /content/
```
#### 마스크 준비
- 마스크 생성 및 리사이즈까지 클래스로 구현
```
%cd car_check
```
```
from pycocotools import coco, cocoeval, _mask
from pycocotools import mask as maskUtils 
from PIL import Image 
from tqdm import tqdm
import array
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import glob
import time
import cv2
import pylab
import os
from random import shuffle
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
%matplotlib inline

class COCO :
  def __init__(self, label) :
    CATEGORY_NAMES=['car']
    ANNOTATION_FILE = label
    
    self.coco_data = coco.COCO(ANNOTATION_FILE)
    self.catIds  = self.coco_data.getCatIds(catNms=CATEGORY_NAMES);
    self.imgIds  = self.coco_data.getImgIds(catIds=self.catIds);
    self.imgDict = self.coco_data.loadImgs(self.imgIds)
    print('hi',self.coco_data, self.catIds, self.imgIds, self.imgDict)
    
  def shuffle_data( self, img_cnt=6000 ) :
    shuffle(self.imgIds)
    self.imgIds = self.imgIds[0:img_cnt]
    print(self.imgIds)

  def make_mask_img( self, file_path ) :
    print(self.imgIds)
    for ID in tqdm(self.imgIds) :
      sampleImgIds  = self.coco_data.getImgIds( imgIds=[ID] )
      sampleImgDict = self.coco_data.loadImgs(sampleImgIds[np.random.randint(0,len(sampleImgIds))])[0]
      file_name = sampleImgDict['file_name']
      annIds = self.coco_data.getAnnIds(imgIds=sampleImgDict['id'], catIds=self.catIds, iscrowd=0)
      anns   = self.coco_data.loadAnns(annIds)
      mask   = self.coco_data.annToMask(anns[0])

      for i in range(len(anns)):
          mask = mask | self.coco_data.annToMask(anns[i])
      
      mask = Image.fromarray(mask * 255 , mode = "L")

      
      # 저장시 같은 이름 존재하면 건너뛰기
      if os.path.exists( file_path + file_name ) :
        print(f'{file_name} 같은 이름 파일이 존재합니다.')
        continue 

      
      mask.save( file_path + file_name )

def make_img_resize(path, save_dir_path, mode='data') : 
  '''
    path          : 이미지가 있는 경로를 설정해주세요. ex) /content/*
    save_dir_path : resize img 저장하고 싶은 경로를 설정해주세요. ex) /content/save_dir
    mode          : ['data':RGB], ['mask':grayscale]
  '''
  files = glob.glob(path) # 저장할 이미지 경로

  # 저장할 경로 없으면 생성
  if not os.path.exists(save_dir_path):
      os.mkdir(save_dir_path)
      print('저장 경로 생성')

  if mode=='mask' : 
    mode='L'
  elif mode=='data' : 
    mode='RGB'
  else :
    print('모드를 다시 선택해주세요')
    return
  for flie in tqdm(files) :
    file_name = flie.split('/')[-1]                     # 파일이름 추출
    save_path = os.path.join(save_dir_path, file_name ) # 저장경로 설정

    # 저장시 같은 이름 존재하면 건너뛰기
    if os.path.exists(save_path) :
      print(f'{file_name} 같은 이름 파일이 존재합니다.')
      time.sleep(0.1)
      continue 
  
    # 이미지 오픈
    with Image.open(flie) as im :
      im = im.resize((128, 128)) # 이미지 resize
      im = im.convert(mode)     # RGB로 변경
      im.save( save_path )       # resize img 저장
      time.sleep(0.3)
    time.sleep(0.2)    
```
```
save_path_train = './scratch/training/coco_label/coco_label.json'
save_path_valid = './scratch/validation/coco_label/coco_label.json'
save_path_test  = './scratch/test/coco_label/coco_label.json'

save_mask_path_train = './scratch/training/mask'
save_mask_path_valid = './scratch/validation/mask'
save_mask_path_test  = './scratch/test/mask'

train_img_path = './scratch/training/image/*'
valid_img_path = './scratch/validation/image/*'
test_img_path  = './scratch/test/image/*'

save_train_rimg_path = './scratch/training/resize_image'
save_valid_rimg_path = './scratch/validation/resize_image'
save_test_rimg_path  = './scratch/test/resize_image'

save_train_rmask_path = './scratch/training/resize_mask'
save_valid_rmask_path = './scratch/validation/resize_mask'
save_test_rmask_path  = './scratch/test/resize_mask'

coco_train = COCO( save_path_train )
coco_train.shuffle_data()
coco_train.make_mask_img( save_mask_path_train )
make_img_resize(train_img_path, save_train_rimg_path)
make_img_resize( save_mask_path_train +'/*', save_train_rmask_path, 'mask' )

coco_valid = COCO( save_path_valid )
coco_valid.shuffle_data()
coco_valid.make_mask_img( save_mask_path_valid )
make_img_resize(valid_img_path, save_valid_rimg_path)
make_img_resize( save_mask_path_valid +'/*', save_valid_rmask_path, 'mask' )

coco_test  = COCO( save_path_test )
coco_test.shuffle_data()
coco_test.make_mask_img( save_mask_path_test )
make_img_resize(test_img_path, save_test_rimg_path)
make_img_resize( save_mask_path_test +'/*', save_test_rmask_path, 'mask' )
```
### Data Generator
- U-Net 학습시키기 전 데이터를 전처리한다
  - 배치사이즈, 에폭
  - 이미지 크기, 채널
```
from keras.api._v2.keras import Model
import os
import sys
import random

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *

seed = 2019

random.seed = seed
np.random.seed = seed 

class DataGen(tf.keras.utils.Sequence):
  def __init__(self , path_input , path_mask , batch_size = 8 , image_size = 128):
    
    self.ids = os.listdir(path_input)
    self.path_input = path_input
    self.path_mask = path_mask
    self.batch_size = batch_size
    self.image_size = image_size
    self.on_epoch_end()
  
  def __load__(self , id_name):
    image_path = os.path.join(self.path_input , id_name)
    mask_path = os.path.join(self.path_mask , id_name) 
    
    image = cv2.imread(image_path , 1) # 1 specifies RGB format
    mask = cv2.imread(mask_path , -1)
    mask = mask.reshape((self.image_size , self.image_size , 1))
      
    #normalize image
    image = image / 255.0
    mask = mask / 255.0

    return image , mask
  
  def __getitem__(self , index):
    if (index + 1)*self.batch_size > len(self.ids):
      self.batch_size = len(self.ids) - index * self.batch_size
        
    file_batch = self.ids[index * self.batch_size : (index + 1) * self.batch_size]
    
    images = []
    masks = []
    
    for id_name in file_batch : 
      
      _img , _mask = self.__load__(id_name)
      images.append(_img)
      masks.append(_mask)
    
    
    images = np.array(images)
    masks = np.array(masks)

    return images , masks
  
  
  def on_epoch_end(self):
    pass
  
  
  def __len__(self):
    _len = int(np.ceil(len(self.ids) / float(self.batch_size)))
    return _len
```
### UNet
```
def down_block(
    input_tensor,
    no_filters,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="same",
    kernel_initializer="he_normal",
    max_pool_window=(2, 2),
    max_pool_stride=(2, 2)
):
    conv = Conv2D(
        filters=no_filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=None,
        padding=padding,
        kernel_initializer=kernel_initializer
    )(input_tensor)

    conv = BatchNormalization(scale=True)(conv)

    conv = Activation("relu")(conv)

    conv = Conv2D(
        filters=no_filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=None,
        padding=padding,
        kernel_initializer=kernel_initializer
    )(conv)

    conv = BatchNormalization(scale=True)(conv)

    # conv for skip connection
    conv = Activation("relu")(conv)
    
    pool = MaxPooling2D(pool_size=max_pool_window, strides=max_pool_stride)(conv)

    return conv, pool

def bottle_neck(
    input_tensor,
    no_filters,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="same",
    kernel_initializer="he_normal"
):
    conv = Conv2D(
        filters=no_filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=None,
        padding=padding,
        kernel_initializer=kernel_initializer
    )(input_tensor)

    conv = BatchNormalization(scale=True)(conv)

    conv = Activation("relu")(conv)

    conv = Conv2D(
        filters=no_filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=None,
        padding=padding,
        kernel_initializer=kernel_initializer
    )(conv)

    conv = BatchNormalization(scale=True)(conv)

    conv = Activation("relu")(conv)

    return conv

def up_block(    
    input_tensor,
    no_filters,
    skip_connection, 
    kernel_size=(3, 3),
    strides=(1, 1),
    upsampling_factor = (2,2),
    max_pool_window = (2,2),
    padding="same",
    kernel_initializer="he_normal"):
    
    
    conv = Conv2D(
        filters = no_filters,
        kernel_size= max_pool_window,
        strides = strides,
        activation = None,
        padding = padding,
        kernel_initializer=kernel_initializer
    )(UpSampling2D(size = upsampling_factor)(input_tensor))
    
    conv = BatchNormalization(scale=True)(conv)

    conv = Activation("relu")(conv) 
    
    conv = concatenate( [skip_connection , conv]  , axis = -1)
    
    
    conv = Conv2D(
        filters=no_filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=None,
        padding=padding,
        kernel_initializer=kernel_initializer
    )(conv)

    conv = BatchNormalization(scale=True)(conv)
    conv = Activation("relu")(conv)
    conv = Conv2D(
        filters=no_filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=None,
        padding=padding,
        kernel_initializer=kernel_initializer
    )(conv)

    conv = BatchNormalization(scale=True)(conv)

    conv = Activation("relu")(conv)
    
    return conv



def output_block(input_tensor,
    padding="same",
    kernel_initializer="he_normal"
):
    
    conv = Conv2D(
        filters=2,
        kernel_size=(3,3),
        strides=(1,1),
        activation="relu",
        padding=padding,
        kernel_initializer=kernel_initializer
    )(input_tensor)
    
    
    conv = Conv2D(
        filters=1,
        kernel_size=(1,1),
        strides=(1,1),
        activation="sigmoid",
        padding=padding,
        kernel_initializer=kernel_initializer
    )(conv)
    
    
    return conv
    

def UNet(input_shape = (128,128,3)):
    
    filter_size = [64,128,256,512,1024]
    
    inputs = Input(shape = input_shape)
    
    d1 , p1 = down_block(input_tensor= inputs,
                         no_filters=filter_size[0],
                         kernel_size = (3,3),
                         strides=(1,1),
                         padding="same",
                         kernel_initializer="he_normal",
                         max_pool_window=(2,2),
                         max_pool_stride=(2,2))
    
    
    d2 , p2 = down_block(input_tensor= p1,
                         no_filters=filter_size[1],
                         kernel_size = (3,3),
                         strides=(1,1),
                         padding="same",
                         kernel_initializer="he_normal",
                         max_pool_window=(2,2),
                         max_pool_stride=(2,2))
    
    
    
    d3 , p3 = down_block(input_tensor= p2,
                         no_filters=filter_size[2],
                         kernel_size = (3,3),
                         strides=(1,1),
                         padding="same",
                         kernel_initializer="he_normal",
                         max_pool_window=(2,2),
                         max_pool_stride=(2,2))
    
    
    
    d4 , p4 = down_block(input_tensor= p3,
                         no_filters=filter_size[3],
                         kernel_size = (3,3),
                         strides=(1,1),
                         padding="same",
                         kernel_initializer="he_normal",
                         max_pool_window=(2,2),
                         max_pool_stride=(2,2))
    
    
    b = bottle_neck(input_tensor= p4,
                         no_filters=filter_size[4],
                         kernel_size = (3,3),
                         strides=(1,1),
                         padding="same",
                         kernel_initializer="he_normal")
    
    
    
    u4 = up_block(input_tensor = b,
                  no_filters = filter_size[3],
                  skip_connection = d4,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  upsampling_factor = (2,2),
                  max_pool_window = (2,2),
                  padding="same",
                  kernel_initializer="he_normal")
    
    u3 = up_block(input_tensor = u4,
                  no_filters = filter_size[2],
                  skip_connection = d3,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  upsampling_factor = (2,2),
                  max_pool_window = (2,2),
                  padding="same",
                  kernel_initializer="he_normal")
    
    
    u2 = up_block(input_tensor = u3,
                  no_filters = filter_size[1],
                  skip_connection = d2,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  upsampling_factor = (2,2),
                  max_pool_window = (2,2),
                  padding="same",
                  kernel_initializer="he_normal")
    
    
    u1 = up_block(input_tensor = u2,
                  no_filters = filter_size[0],
                  skip_connection = d1,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  upsampling_factor = (2,2),
                  max_pool_window = (2,2),
                  padding="same",
                  kernel_initializer="he_normal")
    
    
    
    output = output_block(input_tensor=u1 , 
                         padding = "same",
                         kernel_initializer= "he_normal")
    
    model = Model(inputs = inputs , outputs = output)
    
    
    return model
```

### 학습준비
#### 모델준비 및 컴파일준비
```
model = UNet(input_shape = (128,128,3))
model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
```

#### 데이터 전처리 및 학습
```
image_size = 128 
epochs = 10
batch_size = 8

train_img_path      = './scratch/training/resize_image'
valid_img_path      = './scratch/validation/resize_mask'
train_img_mask_path = './scratch/training/resize_image'
valid_img_mask_path = './scratch/validation/resize_mask'

train_gen = DataGen(path_input = train_img_path , path_mask = train_img_mask_path , batch_size = batch_size , image_size = image_size)
val_gen   = DataGen(path_input = valid_img_path , path_mask = valid_img_mask_path , batch_size = batch_size , image_size = image_size)

train_steps =  len(os.listdir( train_img_path ))/batch_size

hist = model.fit_generator(train_gen , validation_data = val_gen , steps_per_epoch = train_steps , epochs=epochs)
```

### 모델 성능 평가

```
#HIST
import matplotlib.pyplot as plt
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy graph')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'])
plt.grid()
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss graph')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'])
plt.grid()
plt.show()
```
