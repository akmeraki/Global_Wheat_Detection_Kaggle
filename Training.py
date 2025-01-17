import pandas as pd 
import numpy as np
import os  
import matplotlib.pyplot as plt 

# Pytorch 
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Custom
import Utils 
from models.models import pytorch_FASTER_RCNN
from Utils.data_loader_class import Wheat_Detection_Class,collate_fn



# Training for One epoch 
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = torch.u.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
        
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

# Evaluate the model 
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    return coco_evaluator


def run_training(df_path,data_dir,training_dict):

    train_df = pd.read_csv(os.path.join(df_path,'train_df.csv'))
    valid_df = pd.read_csv(os.path.join(df_path,'valid_df.csv'))

    # Creating the Train Data loader and Valid Data loader
    transform = transforms.Compose([
                transforms.ToTensor(),])

    train_dataset =  Wheat_Detection_Class(train_df, os.path.join(data_dir,'train'), transforms=transform)
    train_data_loader = DataLoader(dataset= train_dataset, batch_size=2, shuffle = True, collate_fn =collate_fn)
    
    valid_dataset =  Wheat_Detection_Class(valid_df, os.path.join(data_dir,'test'), transforms=transform)
    valid_data_loader = DataLoader(dataset= train_dataset, batch_size=2, shuffle = True, collate_fn =collate_fn)

    model = pytorch_FASTER_RCNN(backbone='Resnet50_fpn', pre_trained=True)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # A learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for # num of epochs 
    num_epochs = training_dict['epochs']
    device = training_dict['device']

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, valid_data_loader, device=device)


if __name__ == "__main__":

    # Directories  
    df_path = './Preprocessing/'
    Data_directory = './Data_directory/'
    epochs = 10 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    training_dict = {'epochs':epochs, 'device':device}


    run_training(df_path, Data_directory,training_dict)
    
    




