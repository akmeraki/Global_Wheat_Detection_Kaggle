"""
Aneurysm Dataset Class 

returns : Image, target, Image_id 

"""
import cv2
import numpy as np
import pandas as pd 
import os 
import torch 
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

# collate_fn (callable, optional): merges a list of samples to form a  
# mini-batch of Tensor(s).  Used when using batched loading from a map-style dataset.  
def collate_fn(batch):
    return tuple(zip(*batch))

# Data loader 
class Wheat_Detection_Class(Dataset):
    
    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()
        
        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms 
        
    def __getitem__(self, index):

        # the image id that correspods to the index 
        image_id = self.image_ids[index]
        
        # All the bounding boxes that correspond to the image_id 
        records = self.df[self.df['image_id'] == image_id]
        
        # Reading the image 
        image = cv2.imread(os.path.join(self.image_dir,image_id)+'.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
 
        # Normalizing the image 
        image /= 255.0

        # Get the bounding boxes from the records dataframe 
        boxes = records[['x','y','w','h']].values
        boxes[:,2] = boxes[:,0] + boxes[:,2]
        boxes[:,3] = boxes[:,1] + boxes[:,3]

        
        # Get the area of the bounding boxes 
        area = (boxes[:,2] - boxes[:,0])*(boxes[:,3] - boxes[:,1])
        area = torch.as_tensor(area, dtype = torch.float32)
    
        # Labels -> all correspond to the same class so all ones 
        labels = torch.ones((records.shape[0],), dtype= torch.int64) 
        
        # target is a dictionary of all the bounding boxes, area, labels, image_id information associated to a image_id 
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels 
        target['image_id'] = torch.as_tensor([index])
        target['area'] = area
    
        return image, target, image_id
    
    def __len__(self):
        return self.image_ids.shape[0]
        

if __name__ == "__main__":

    # Directory 
    train_df_path = './Preprocessing/train_df.csv'
    train_dir = './Data_directory/train'

    train_df = pd.read_csv(train_df_path)

    # Testing the Dataloader Class 
    transform = transforms.Compose([
                transforms.ToTensor(),])

    train_dataset =  Wheat_Detection_Class(train_df, train_dir, transforms=transform)
    train_data_loader = DataLoader(dataset= train_dataset, batch_size=2, shuffle = True, collate_fn =collate_fn)

    # Checking the batch 
    examples = enumerate(train_data_loader)

    batch_idx, (example_data, target, image_ids) = next(examples)
    print(example_data)
    print(target)