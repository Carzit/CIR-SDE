import os
import datetime
import argparse
import logging

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from tqdm import tqdm
from utils import save_and_load, training_board
from dataset import TradeData, TradeDataset
from net import *



def train(
        epoches:int, 
        optimizer:torch.optim.Optimizer,
        model:CIRNet, 
        loss_fn:TraceLoss, 
        train_generator:DataLoader, 
        val_generator:DataLoader,
        *,
        lr_scheduler:torch.optim.lr_scheduler.LRScheduler=None,
        hparams:dict=None,
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
        
        # Train one Epoch
        model.train()
        train_loss_epoch = 0
        for batch, trace_data in enumerate(tqdm(train_generator)):
            trace_data = trace_data.to(device=device)
            r_predicts, regs, dts = model(trace_data)
            weights = dts.cumsum(dim=0).reciprocal()
            train_loss = loss_fn(r_predicts, regs, trace_data, weights)  

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            with torch.no_grad():
                weight_sum = model.cir_cell.epsilon_linear_layer.weight.data.sum()
                model.cir_cell.epsilon_linear_layer.weight.data /= weight_sum

            train_loss_epoch += train_loss.item()

        # Record Train Loss Scalar
        writer.add_scalar("Train Loss", train_loss_epoch / (batch+1), epoch)

        # If hyper parameters passed, record it in hparams(no val).
        if hparams and not val_generator:
            writer.add_hparams(hparam_dict=hparams, metric_dict={"hparam/TrainLoss":train_loss_epoch / (batch+1)})
        
        # If validation datasets exisit, calculate val loss without recording grad.
        if val_generator:
            val_loss_epoch = 0
            model.eval() # set eval mode to frozen layers like dropout
            with torch.no_grad(): 
                for batch, trace_data in enumerate(tqdm(val_generator)):
                    trace_data = trace_data.to(device=device)
                    r_predicts, regs, dts = model(trace_data)
                    weights = dts.cumsum(dim=0).reciprocal()
                    val_loss = loss_fn(r_predicts, regs, trace_data, weights)  
                    val_loss_epoch += val_loss.item()
                
                writer.add_scalar("Validation Loss", val_loss_epoch / (batch+1), epoch)
                writer.add_scalars("Train-Val Loss", {"Train Loss": train_loss_epoch / (batch+1), "Validation Loss": val_loss_epoch / (batch+1)}, epoch)

        # If hyper parameters passed, record it in hparams.
        if hparams and val_generator:
            writer.add_hparams(hparam_dict=hparams, 
                               metric_dict={"hparam/TrainLoss":train_loss_epoch / (batch+1), 
                                            "hparam/ValLoss":val_loss_epoch / (batch+1)})

        # If learning rate scheduler exisit, update learning rate per epoch.
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)
        if lr_scheduler:
            lr_scheduler.step()
        
        # Flushes the event file to disk
        writer.flush()

        # Specify print_per_epoch = 0 to unable print training information.
        if print_per_epoch:
            if (epoch+1) % print_per_epoch == 0:
                print('Epoch [{}/{}], Train Loss: {:.6f}, Validation Loss: {:.6f}'.format(epoch+1, epoches, train_loss_epoch / (batch+1), val_loss_epoch / (batch+1)))
        
        # Specify save_per_epoch = 0 to unable save model. Only the final model will be saved.
        if save_per_epoch:
            if (epoch+1) % save_per_epoch == 0:
                model_name = f"{save_name}_epoch{epoch}"
                model_path = os.path.join(save_dir, model_name)
                print(model_path)
                save_and_load.save(model, model_path, save_format)
        
        
    writer.close()
    model_name = f"{save_name}_final"
    model_path = os.path.join(save_dir, model_name)
    save_and_load.save(model, model_path, save_format)
    
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Train. Use gradient descent to fit k and theta parameters.")
    parser.add_argument("--load_folder", type=str, required=True, help="Load from folder where`.pt` Dataset file is.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Load Checkpoint Path.")
    parser.add_argument("--epsilon_num", type=int, default=1, help="Number of random samplings of Gaussian noise")
    parser.add_argument("--epoch", type=int, default=20, help="Max training epoch")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--save_folder", type=str, default=None, help="Folder path to save to")
    parser.add_argument("--save_name", type=str, default=None, help="Model name")
    parser.add_argument("--device", type=str, default="cpu", help="Device `cpu` or `cuda`")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s")

    logging.info(f"Loading Dataset from {args.load_folder}")
    dataset = torch.load(os.path.join(args.load_folder, 'dataset_splited.pt'), map_location=torch.device(args.device))
    train_set = dataset["train_set"]
    val_set = dataset["val_set"]
    train_generator = DataLoader(train_set, batch_size=None, shuffle=True, batch_sampler=None)
    val_generator = DataLoader(val_set, batch_size=None, shuffle=True, batch_sampler=None)

    model = CIRNet(sigma_num=5, epsilon_num=args.epsilon_num)
    if args.checkpoint is not None:
        save_and_load.load(model, path=args.checkpoint, device=args.device)
    loss_fn = TraceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    configs = {
        "epsilon_num": args.epsilon_num,
        "sigma_num": 5,
        "epoch": args.epoch,
        "lr": args.lr,
        "device": args.device
    }
    logging.info(f"Training Config:{configs}")

    logging.info("Start Training!")
    train(epoches=args.epoch,
          optimizer=optimizer,
          model=model,
          loss_fn=loss_fn,
          train_generator=train_generator,
          val_generator=val_generator,
          save_dir=args.save_folder,
          save_name=args.save_name,
          device=torch.device(args.device))
    
    # Example: python train.py --load_folder "data\dummyTrade1" --epsilon_num 5  --save_folder "save" --save_name "M1" --epoch 20 --lr 0.001 --device cpu

    

