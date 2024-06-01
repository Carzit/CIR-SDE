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


@torch.no_grad()
def infer(model:torch.nn.Module, 
          model_path:str,
          train_generator:DataLoader, *, 
          log_dir:str = r"log", 
          save_dir:str=os.curdir,
          save_name:str="result.pt",
          device:torch.device=torch.device('cpu'))->torch.nn.Module:

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    writer = SummaryWriter(os.path.join(log_dir, "TEST"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    model = model.to(device=device)
    load(model=model, path=model_path)

    model.eval()
    traces = []
    traces_tensor = []
    for trace in tqdm(range(len(train_generator)-2)):
        # Train one Trace
        steps = []
        steps_tensor = []
            
        for step_, (date, swap_rate, vols, vega, epsilon) in enumerate(train_generator):
            if step_ in range(trace+1): 
                continue
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

            r_predict, c = model(r, dt, sigma, sample=True)
            steps_tensor.append(r_predict)
            steps.append(r_predict.item())
                
            # Record Trace Loss Scalar
            writer.add_scalar(f"Trace{trace}", r_predict.item(), step_)
            writer.flush()

            date_ = date
            swap_rate_ = swap_rate
            vols_ = vols
            vega_ = vega
            epsilon_ = epsilon

        traces_tensor.append(torch.cat(steps_tensor))
        traces.append(steps)

    writer.close()
    torch.save({"traces_tensor": traces_tensor, "traces": traces}, os.path.join(save_dir, save_name))

def add_lists_aligned(lst:list, mean:bool=False):
    max_length = max(len(sublist) for sublist in lst)
    result = [0] * max_length
    for i in range(1, max_length+1):
        for sublist in lst:
            if i <= len(sublist):
                result[-i] += sublist[-i]
        if mean:
            result[-i] /= max_length+1-i
    return result

def parse_args():
    parser = argparse.ArgumentParser(description="Inference.")
    parser.add_argument("--trade", type=str, required=True, help="Trade info name. e.g. dummyTrade1")
    parser.add_argument("--dataset", type=str, default=None, help="TradeDataset path.")
    parser.add_argument("--model", type=str, default=None, help="Model weights path.")
    parser.add_argument("--save", type=str, required=True, help="Model Weights `.pt` File name")
    parser.add_argument("--device", type=str, default="cpu", help="Device `cpu` or `cuda`")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"loading {args.trade} data from {args.dataset}...")
    trade_data = TradeData(args.trade)
    trade_data.process()

    train_set = trade_data.to_dataset(device=torch.device(args.device))
    train_set.load(args.dataset, device=torch.device(args.device))
    train_generator = DataLoader(train_set, batch_size=None, shuffle=False, batch_sampler=None)
    
    model = CIR()
    load(model=model, path=args.model)

    infer(model=model, 
          model_path=args.model, 
          train_generator=train_generator,
          save_dir="result",
          save_name=args.save,
          device=torch.device(args.device))
    
    print(f"Inference result saved to {args.save}")
    
    upper_bound, lower_bound = trade_data.upper_bound, trade_data.lower_bound
    print(f"upper_bound: {upper_bound}, lower_bound: {lower_bound}")

    traces = torch.load(os.path.join("result", args.save))["traces"]

    r_predict = add_lists_aligned(traces, True)
    r_true = trade_data.sub_MDSR["Swap Rate"].tolist()

    r_predict_total = 0
    r_true_total = 0
    for r in r_predict:
        if lower_bound <= r <= upper_bound:
            r_predict_total += 1
    for r in r_true:
        if lower_bound <= r <= upper_bound:
            r_true_total += 1
    print(r_predict_total, r_true_total)


    #Example: python infer.py --trade dummyTrade1 --dataset data\dataset.pt --model weights\model_epoch10.pt  --save result.pt --device cpu