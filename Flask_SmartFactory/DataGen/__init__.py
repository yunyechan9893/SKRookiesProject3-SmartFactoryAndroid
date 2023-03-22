import os
import numpy as np
import cv2
import tensorflow as tf

class DataGen(tf.keras.utils.Sequence):
  def __init__(self , path_input , batch_size = 8):
    
    self.ids = os.listdir(path_input)
    self.path_input = path_input
    self.batch_size = batch_size
    self.image_size = 128
    self.on_epoch_end()
  
  def __load__(self , id_name):
    image_path = os.path.join(self.path_input, id_name)
    image_path = image_path.replace('\\','/')

    image = cv2.imread(image_path , cv2.IMREAD_COLOR) # 1 specifies RGB format
    image = cv2.resize(image, dsize=( self.image_size, self.image_size ))
    
    #normalize image
    image = image / 255.0

    return image 
  
  def __getitem__(self , index):
    if (index + 1)*self.batch_size > len(self.ids):
      self.batch_size = len(self.ids) - index * self.batch_size
        
    file_batch = self.ids[index * self.batch_size : (index + 1) * self.batch_size]
    
    images = []
    
    for id_name in file_batch : 
      
      _img = self.__load__(id_name)
      images.append(_img)
    
    
    images = np.array(images)

    return images
  
  
  def on_epoch_end(self):
    pass
  
  
  def __len__(self):
    _len = int(np.ceil(len(self.ids) / float(self.batch_size)))
    return _len