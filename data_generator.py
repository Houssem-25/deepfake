
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
import librosa


from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence


class MultipleInputGenerator(Sequence):
    def __init__(self, base_path, data , width, height, channel, sequence,batch_size=2):
        self.audio_generator = AudioGenerator(base_path,data,batch_size=batch_size)
        self.image_generator = ImageGenerator(base_path,data,width,height,channel,sequence,batch_size=batch_size)
        
    def __len__(self):
        """It is mandatory to implement it on Keras Sequence"""
        return self.image_generator.__len__()

    def __getitem__(self, index):
        """Getting items from the 2 generators and packing them"""
        X2_batch, Y_batch = self.audio_generator[index]
        X1_batch, Y_batch = self.image_generator[index]

        X_batch = [X1_batch, X2_batch]

        return X_batch, Y_batch



class AudioGenerator(Sequence):
    def __init__(self,base_path,data,batch_size=10):
        self.base_path = base_path
        self.batch_size = batch_size
        self.data = data
        self.on_epoch_end()
         
    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.data.index.values[k] for k in indexes]
        X = self.__data_generation(list_IDs_temp)
        y = [self.data['label'][k] for k in indexes]
        print("Audio : ")
        print(y)
        print("\n")
        lb = LabelEncoder()
        y = lb.fit_transform(y)
        return (X, to_categorical(y))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))

    
    def __data_generation(self, files):
        print(files)
        x =[0]*self.batch_size
        for i,file in enumerate(files) : 
            aud,y = librosa.load(self.base_path + file)
            x[i]  = librosa.feature.mfcc(aud,y)
        return x 

class ImageGenerator(Sequence):
    def __init__(self, base_path, data,width,height,channel,sequence, batch_size=32):
        self.base_path = base_path
        self.batch_size = batch_size
        self.data = data
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
        print("Image : ")
        print(y)
        print("\n")
        lb = LabelEncoder()
        y = lb.fit_transform(y)
        return (X, to_categorical(y))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
                 
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
