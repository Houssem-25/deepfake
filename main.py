from data_loader import get_data

train,test = get_data("/data/train_sample_videos/")
print(train["label"].value_counts())