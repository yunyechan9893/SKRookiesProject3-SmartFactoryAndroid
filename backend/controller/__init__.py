import numpy as np
from flask import request, Blueprint
import time
from utils.datagen import DataGen
import cv2
from PIL import Image 
from uuid import uuid4
import os
import glob
import tensorflow as tf
from firebase_admin import storage

api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

# 모델을 불러온다
model = tf.keras.models.load_model('resource/car_scratch_model.h5', compile=False)

@api_bp.route('/', methods=['GET'])
def test():
    return 'hi'

# 버킷은 바이너리 객체의 상위 컨테이너. Storage에서 데이터를 보관하는 기본 컨테이너
# bucket = storage.bucket() # 기본 버킷 사용

# 안드로이드 앱에서 /comm 주소로 접근한다. 방식은 post
# http://172.29.24.92:3333/comm
@api_bp.route('/mask-file', methods=['post'])
def hello( ):
    # 이미지 저장 경로 설정 => 임시저장, 작업이 끝나면 자동삭제
    save_img_path    = r'.\image'
    # 앱으로 부터 사진을 받는다
    req_img          = request.files['file']
    # 파일이름 저장 => time.time()을 사용하여 중복이름이 없도록 만들었다
    unique_file_name = f'file_{ str(np.round(time.time(), 5)).replace(".","_") }.jpg'
    # 이미지 파일 저장
    req_img.save(f'{save_img_path}/{unique_file_name}')

    # 위에서 저장한 이미지로 예측을 수행한다 => 마스크 생성 
    # DataGen을 통해 학습하기에 알맞은 형태로 만들어준다 
    y_predict = model.predict(DataGen(save_img_path, batch_size=1))
    
    # 마스크 차원을 (128,128,1) -> (128,128)로 변경한다
    # uint8로 형식을 바꿔주기 위해 => cv2로 이미지 저장하려면 uint8로 바꿔줘야한다
    rsize_mask     = np.reshape(y_predict[0], (128,128))
    # 형식변환
    rsize_mask     = Image.fromarray((rsize_mask * 255).astype(np.uint8))
    # 마스크 저장경로 설정
    save_mask_path = './pred_image/'
    # 마스크 차원 다시 변경 (128,128) -> (128,128,1)
    rsize_mask     =  np.reshape(rsize_mask, (128,128,1))
    
    # 마스크 저장
    cv2.imwrite( save_mask_path + unique_file_name , rsize_mask )
    # 파이어베이스 업로드 => 이미지, 마스크
    # fileUpload(  save_mask_path , unique_file_name )
    # fileUpload(  save_img_path  + '/' , unique_file_name, 'image/' )

    # 앱으로 파일 이름을 전달한다 => 추후에 앱에서 파이어베이스 사진을 받아오기 위한 작업
    return unique_file_name


# def fileUpload(file_path, file_name, fb_save_path ='mask/'):
#     # 저장한 사진을 파이어베이스 storage의 mask/ or image/ 라는 이름의 디렉토리에 저장
#     blob = bucket.blob(fb_save_path + file_name) 
#     # new token and metadata 설정 => 이게 왜 필요한지 나도 공부해야함
#     new_token = uuid4()
#     #access token이 필요하다.
#     metadata = {"firebaseStorageDownloadTokens": new_token} 
#     blob.metadata = metadata
#     # vscode에 저장된 파일 주소와 이미지 형식
#     blob.upload_from_filename(filename=file_path+file_name, content_type='image/jpg') 
#     print(blob.public_url)
#     # vscode에 저장된 이미지와 마스크 삭제 
#     for file in glob.glob(f'{file_path}*') : os.remove(file)