from data_loader import get_data
from data_generator import MultipleInputGenerator
from model import Model
import os

train,test = get_data("/data/train_sample_videos/")
train_gen = MultipleInputGenerator(os.getcwd()+"/data/train_sample_videos/",train,200,200,3,300,batch_size=2)
test_gen = MultipleInputGenerator(os.getcwd()+"/data/train_sample_videos/",test,200,200,3,300,batch_size=2)

model = Model(train_gen,test_gen,[])
model.create_model()
model.summery()
model.fit()