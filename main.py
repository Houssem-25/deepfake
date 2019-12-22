from data_loader import get_data
from audio_generator import AudioGenerator
from image_generator import ImageGenerator
import os

train,test = get_data("/data/train_sample_videos/")
im = ImageGenerator(os.getcwd()+"/data/train_sample_videos/",train,222,222,3,300,1)
aud = AudioGenerator(os.getcwd()+"/data/train_sample_videos/",train)
