# 프로젝트 설명
- 자동차의 스크래치를 마스크 처리하여 보여주는 어플
- 코랩학습 -> 플라스크 모델 서빙 -> 안드로이드 서비스 배포
## 기간
- 1개월
## 팀구성
- 윤예찬 외 2명

## 내가 맡은 역할
- 데이터 분석
- 데이터 전처리
- 모델 학습
- flask를 이용한 백엔드 구축
- 파이어베이스 연동
- 안드로이드 설계 및 구축
  
## 기여한 부분
- 딥러닝 모델 구축
- 백엔드 서버 구축
- 안드로이드 앱 구축

## 아키텍쳐
![image](https://github.com/yunyechan9893/SKRookiesProject3-SmartFactoryAndroid/assets/125535111/18034276-e824-486a-8866-40980563326d)

## 달성한 결과/성과
![평가지표](https://user-images.githubusercontent.com/125535111/226706000-6fd19ba2-d58e-431d-8193-f8b661361077.png)
#### 스크래치 검출 정확도 90% 이상 달성
- Train accuracy = 93.1%
- Validation accuracy = 91.9%


## 결과
![제목을-입력해주세요_-001](https://github.com/yunyechan9893/sk_rookies_project3/assets/125535111/1cd62fe0-1926-49f0-9c82-bffc316ff881)

<br />

---------------------------------------------------------
<br />


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
