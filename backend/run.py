from flask import Flask
from controller import api_bp
import os
from requests import get
import tensorflow as tf


def create_app():
    app = Flask(__name__)
    load_env(app)
    setting()
    load_model()
    register_blueprint(app)

    return app

def load_env(app):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 경고와 오류 메시지만 표시
    tf.get_logger().setLevel('ERROR')  # 경고 무시
    app.config['SECRET_KEY'] = 'higaluctc4841'

def load_model():
    url  = 'https://firebasestorage.googleapis.com/v0/b/carscratch.appspot.com/o/Model%2Fbeefprince_model.h5?alt=media&token=23767bf7-35ab-4a55-8bec-f6564bcb097a'
    path = 'resource/model/car_scratch_model.h5'

     # 딥러닝 파일 저장하기, 최초 한번만 실행
    if not os.path.isfile(path) :
        with open(path, 'wb') as f :
            ai_file = get(url)
            f.write(ai_file.content)

def setting():
    RESOURCE_DIR = 'resource'
    MODEL_DIR = 'model'
    IMG_DIR = 'image'
    BASIC_IMG_DIR = 'basic'
    MASK_IMG_DIR = 'mask'

    # 기본 이미지와 마스크 이미지의 경로를 설정
    basic_image_path = os.path.join(RESOURCE_DIR, IMG_DIR, BASIC_IMG_DIR)
    mask_image_path = os.path.join(RESOURCE_DIR, IMG_DIR, MASK_IMG_DIR)
    model_path = os.path.join(RESOURCE_DIR, MODEL_DIR)

    from util.scretch_model import Model
    Model()

    # 디렉토리가 없으면 생성합니다. exist_ok=True는 디렉토리가 이미 존재하면 무시
    os.makedirs(basic_image_path, exist_ok=True)
    os.makedirs(mask_image_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)


def register_blueprint(app):
    app.register_blueprint(api_bp)

if __name__ == '__main__':
    soketio = create_app()
    soketio.run( host='0.0.0.0', port=3333, debug=True )
