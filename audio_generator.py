from keras.utils.data_utils import Sequence 
from keras.utils import to_categorical
import os
from sklearn.model_selection import train_test_split
import numpy as np
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
from data_loader import get_data
import librosa

class AudioGenerator(Sequence):
    def __init__(self,base_path,data,batch_size=10,shuffle=True):
        self.base_path = base_path
        self.batch_size = batch_size
        self.data = data
        self.shuffle = shuffle
        self.on_epoch_end()
         
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
    
    def __data_generation(self, files):
        x =[0]*self.batch_size
        for i,file in enumerate(files) : 
            aud,y = librosa.load(self.base_path + file)
            x[i]  = librosa.feature.mfcc(aud,y)
        return x 
               