import tensorflow as tf

class Model:
    _instance = None  # 싱글톤 인스턴스를 저장할 클래스 변수

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Model, cls).__new__(cls)
        return cls._instance  # 싱글톤 인스턴스 반환

    def __init__(self):
        self.__model = tf.keras.models.load_model('resource/model/car_scratch_model.h5', compile=False)

    def get(self):
        return self.__model