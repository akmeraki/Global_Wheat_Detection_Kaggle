import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import re 



def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r

def preprocess_csv_to_df(train_dir, train_csv_file):
    """
    This function takes in a images directory and corresponding csv file to 
    to create a dataframe and preprocess it to a suitable format 

    """
    df = pd.read_csv(train_csv_file)

    df['x'] = -1
    df['y'] = -1
    df['w'] = -1
    df['h'] = -1

    df[['x', 'y', 'w', 'h']] = np.stack(df['bbox'].apply(lambda x: expand_bbox(x)))
    df.drop(columns=['bbox'], inplace=True)
    df['x'] = df['x'].astype(np.float)
    df['y'] = df['y'].astype(np.float)
    df['w'] = df['w'].astype(np.float)
    df['h'] = df['h'].astype(np.float)

    return df   


if __name__ == '__main__':
    
    # Directories 
    data_directory = './Data_directory/'
    train_dir = './Data_directory/train/' 
    test_dir = './Data_directory/test/'
    train_csv_path = './Data_directory/train.csv'

    df = preprocess_csv_to_df(train_dir, train_csv_path)
    
    # Take unique images since multiple bounding boxes 
    unique_ids = pd.unique(df['image_id'])
    print('The Unique Ids are :', len(unique_ids))
    
    # The training to testing ratio  
    train_ratio = 0.8

    # Splitting the training and validation as different images so there is no possible data leakage  
    train_df_ids = unique_ids[:int(len(unique_ids)*0.8)]
    valid_df_ids = unique_ids[int(len(unique_ids)*0.8):]

    # Final Training and Testing dataframes  
    train_df = df.loc[df['image_id'].isin(train_df_ids)]
    valid_df = df.loc[df['image_id'].isin(valid_df_ids)]
    
    # Saving the dataframes as csv for later access in Preprocessing
    train_df.to_csv('./Preprocessing/train_df.csv', index=False)
    valid_df.to_csv('./Preprocessing/valid_df.csv', index=False)





































