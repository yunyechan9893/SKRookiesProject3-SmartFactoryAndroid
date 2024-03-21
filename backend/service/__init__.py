
import numpy as np
import cv2
from PIL import Image 
from uuid import uuid4
import os
import base64
import os

def get_mask_file(scretch_img):
    IMG_DIR = 'resource/image'
    BASIC_IMG_DIR = 'basic'
    MASK_IMG_DIR = 'mask'

    random_file_name = create_random_file_name()
    basic_path = os.path.join(IMG_DIR, BASIC_IMG_DIR)
    basic_file_path = os.path.join(basic_path, random_file_name)

    scretch_img.save(basic_file_path)
    rsize_mask = make_pretreatment(basic_file_path)
    
    # 마스크 저장
    mask_file_path = os.path.join(IMG_DIR, MASK_IMG_DIR, random_file_name)
    cv2.imwrite( mask_file_path , rsize_mask )

    mask_image = cv2.imread(mask_file_path)  
    _, buffer = cv2.imencode('.jpg', mask_image)
    base64_image = base64.b64encode(buffer).decode('utf-8')

    return base64_image

def make_pretreatment(basic_file_path):
    # 저장한 이미지로 예측을 수행한다 => 마스크 생성 
    image = cv2.imread(basic_file_path , cv2.IMREAD_COLOR) # 1 specifies RGB format
    image = cv2.resize(image, dsize=( 128, 128 ))
    image = image / 255.0
   
    from util.scretch_model import Model
    y_predict = Model().get().predict(np.array([image]))[0]
    
    # 마스크 차원을 (128,128,1) -> (128,128)로 변경한다
    # uint8로 형식을 바꿔주기 위해 => cv2로 이미지 저장하려면 uint8로 바꿔줘야한다
    rsize_mask     = np.reshape(y_predict, (128,128))

    # 형식변환
    rsize_mask     = Image.fromarray((rsize_mask * 255).astype(np.uint8))

    # 마스크 차원 다시 변경 (128,128) -> (128,128,1)
    rsize_mask     =  np.reshape(rsize_mask, (128,128,1))

    return rsize_mask

def create_random_file_name() -> str:
    return str(uuid4()) + '.jpg'
