import os
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

def get_data(path,test_split=0.2):
    data_path = os.getcwd() + path
    df = pd.read_json(data_path + "metadata.json").T
    df = _downsampling(df)
    train , test , _, __= _split_data(df,test_split)
    return train,test

def _downsampling(df):
    # Separate majority and minority classes
    df_majority = df[df["label"]=="REAL"]
    df_minority = df[df["label"]=="FAKE"]
    if (len(df_majority) < len(df_minority)):
        df_majority , df_minority = df_minority , df_majority

    # Downsample majority class
    df_majority_downsampled = resample(df_majority, 
                                    replace=False,    # sample without replacement
                                    n_samples=len(df_minority),     # to match minority class
                                    random_state=123) # reproducible results
    
    # Combine minority class with downsampled majority class
    return pd.concat([df_majority_downsampled, df_minority])
def _split_data(df,test_size_=0.2):
    return train_test_split(df,df['label'], test_size = test_size_, random_state=42)
    

