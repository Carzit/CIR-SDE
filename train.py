import os
import datetime
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from tqdm import tqdm
from utils.save_and_load import save, load
from dataset import TradeData, TradeDataset
from net import *

def train(
        epoches:int, 
        optimizer:torch.optim.Optimizer,
        model:torch.nn.Module, 
        loss_fn:torch.nn.Module, 
        train_generator:DataLoader, 
        *,
        lr_scheduler:torch.optim.lr_scheduler.LRScheduler=None,
        log_dir:str = r"log",
        print_per_epoch:int=1,
        save_per_epoch:int=1,
        save_dir:str=os.curdir,
        save_name:str="model",
        save_format:str="pt",
        device:torch.device=torch.device('cpu'))->torch.nn.Module:
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    writer = SummaryWriter(os.path.join(log_dir, "TRAIN"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    model = model.to(device=device)

    # Train
    for epoch in range(epoches):
        # Train one Epoch (Trace means different marginal condition (start time))
        model.train()
        for trace in tqdm(range(len(train_generator)-2)):
            # Train one Trace
            loss_steps = []
            
            for step_, (date, swap_rate, vols, vega, epsilon) in enumerate(train_generator):
                if step_ in range(trace+1): continue
                elif step_ == trace+1:
                    date_ = date
                    swap_rate_ = swap_rate
                    vols_ = vols
                    vega_ = vega
                    epsilon_ = epsilon
                    continue

                r = swap_rate_
                dt = date - date_
                sigma = vols_
                e = epsilon_

                r_predict, c = model(r, dt, sigma, epsilon=e)

                loss_step = loss_fn(r_predict, swap_rate, c)
                loss_steps.append(loss_step)
                
                # Record Trace Loss Scalar
                writer.add_scalar(f"Epoch{epoch}_Trace{trace}", loss_step.item(), step_)

                date_ = date
                swap_rate_ = swap_rate
                vols_ = vols
                vega_ = vega
                epsilon_ = epsilon

            loss_trace = torch.stack(loss_steps).mean()

            optimizer.zero_grad()
            loss_trace.backward()
            optimizer.step()
         
            # Record Trace Loss Scalar
            writer.add_scalar("Train Loss", loss_trace.item(), epoch)
            writer.add_scalar("k", model.k.item(), epoch)
            writer.add_scalar("theta", model.theta.item(), epoch)
        
        # If learning rate scheduler exisit, update learning rate per epoch.
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)
        if lr_scheduler:
            lr_scheduler.step()
        
        # Flushes the event file to disk
        writer.flush()

        # Specify print_per_epoch = 0 to unable print training information.
        if print_per_epoch:
            if (epoch+1) % print_per_epoch == 0:
                print('Epoch [{}/{}], Train Loss: {:.6f}, Model State:{}'.format(epoch+1, epoches, loss_trace.item(), model.state_dict()))
        
        # Specify save_per_epoch = 0 to unable save model. Only the final model will be saved.
        if save_per_epoch:
            if (epoch+1) % save_per_epoch == 0:
                model_name = f"{save_name}_epoch{epoch}"
                model_path = os.path.join(save_dir, model_name)
                print(model_path)
                save(model, model_path, save_format)
        
    writer.close()
    model_name = f"{save_name}_final"
    model_path = os.path.join(save_dir, model_name)
    save(model, model_path, save_format)
    
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Train. Use gradient descent to fit k and theta parameters.")
    parser.add_argument("--trade", type=str, required=True, help="Trade info name. e.g. dummyTrade1")
    parser.add_argument("--load", type=str, default=None, help="(Optional) Load `.pt` TradeDataset path.")
    parser.add_argument("--save", type=str, required=True, help="Model Weights `.pt` File name")
    parser.add_argument("--epoch", type=int, default=20, help="Max training epoch")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device `cpu` or `cuda`")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"loading {args.trade} data from {args.load}...")
    trade_data = TradeData(args.trade)
    trade_data.process()

    train_set = trade_data.to_dataset(device=torch.device(args.device))
    if args.load:
        train_set.load(args.load, device=torch.device(args.device))
    train_generator = DataLoader(train_set, batch_size=None, shuffle=False, batch_sampler=None)
    
    model = CIR()
    loss_fn = StepLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train(epoches=args.epoch,
          optimizer=optimizer,
          model=model,
          loss_fn=loss_fn,
          train_generator=train_generator,
          save_dir="weights",
          save_name=args.save,
          device=torch.device(args.device))
    
    #Example: python train.py --trade dummyTrade1 --load data\dataset.pt --save trade1.pt --epoch 20 --lr 0.001 --device cpu

    

