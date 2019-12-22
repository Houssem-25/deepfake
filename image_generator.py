from keras.utils.data_utils import Sequence 
from keras.utils import to_categorical
import os
from sklearn.model_selection import train_test_split
from skimage import io
from skimage.transform import resize
import numpy as np
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
from data_loader import get_data
import cv2

class ImageGenerator(Sequence):
    def __init__(self, base_path, data,width,height,channel,sequence, batch_size=32,shuffle=True):
        self.base_path = base_path
        self.batch_size = batch_size
        self.data = data
        self.shuffle = shuffle
        self.on_epoch_end()
        self.width = width
        self.height = height
        self.channel = channel
        self.sequence = sequence 
    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.data.index.values[k] for k in indexes]
        X = self.__data_generation(list_IDs_temp)
        y = [self.data['label'][k] for k in indexes]
        lb = LabelEncoder()
        y = lb.fit_transform(y)
        return (X, to_categorical(y))

    def on_epoch_end(self):
        pass
                 
    def read_images(self,video):
        print(video)
        cap = cv2.VideoCapture(video)
        ret = np.empty( (self.sequence,self.width, self.height, 3))
        for i in range(self.sequence):
            suc , im = cap.read()
            ret[i]=cv2.resize(im, (self.width, self.height),interpolation = cv2.INTER_AREA) 
        return ret
    
    def __data_generation(self, file):
        shape =  (self.batch_size, self.sequence, self.width,self.height,self.channel)
        X = np.empty(shape,dtype=np.float32)
        for i, img in enumerate(file):
            X[i] = self.read_images(self.base_path+img)        
        return X 
           


