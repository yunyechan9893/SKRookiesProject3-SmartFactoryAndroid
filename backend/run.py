from flask import Flask
# from flask_socketio import SocketIO
import firebase_admin
from firebase_admin import credentials
from controller import api_bp
import os
from requests import get
import tensorflow as tf

def create_app():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 경고와 오류 메시지만 표시
    tf.get_logger().setLevel('ERROR')  # 경고 무시
    #my project id
    PROJECT_ID = "carscratch"


    url  = 'https://firebasestorage.googleapis.com/v0/b/carscratch.appspot.com/o/Model%2Fbeefprince_model.h5?alt=media&token=23767bf7-35ab-4a55-8bec-f6564bcb097a'
    path = 'resource/car_scratch_model.h5'

    # 딥러닝 파일 저장하기, 최초 한번만 실행
    if not os.path.isfile(path) :
        with open(path, 'wb') as f :
            ai_file = get(url)
            f.write(ai_file.content)

    # # 파이어베이스 고유식별파일 => 이게 있어야 파이어베이스 접근허가
    # cred = credentials.Certificate(r'.\res\service-account-file.json')
    # # 초기값 설정
    # default_app = firebase_admin.initialize_app(cred,{'storageBucket':f"{PROJECT_ID}.appspot.com"})

    # 객체생성 후 설정을 해주고 소켓과 연동해준다
    app                      = Flask(__name__) # 객체생성 & 현재 실행중인 모듈 전달
    app.register_blueprint(api_bp)
    app.config['SECRET_KEY'] = 'higaluctc4841'
    # soketio                  = SocketIO(app)

    return app

if __name__ == '__main__':
    soketio = create_app()
    soketio.run( host='0.0.0.0', port=3333, debug=True )
