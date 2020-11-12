import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 


def preprocess_file(train_dir, train_csv_file):
    """
    This function takes in a images directory and corresponding csv file to 
    to create a dataframe 

    """
    csv_file = pd.read_csv(train_csv_file)
    print(csv_file)


    return df 







if __name__ == '__main__':
    data_directory = './Data_directory/train/'
    train_dir = './Data_directory/train/' 
    test_dir = './Data_directory/test'
    train_csv_path = './Data_directory/train.csv'

    preprocess_file(train_dir, train_csv_path)
































