import os
import datetime
import argparse
import logging

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from tqdm import tqdm
from utils import save_and_load, training_board, parser_utils
from dataset import TradeData, TradeDataset
from net import *

@torch.no_grad()
def test(
        model:CIRNet, 
        test_generator:DataLoader, 
        *,
        log_dir:str = r"log",
        device:torch.device=torch.device('cpu'))->torch.nn.Module:
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(os.path.join(log_dir, "TEST"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    model = model.to(device=device)

    # Test
    with torch.no_grad(): 

        diff_n1_sum = 0
        diff_n2_sum = 0
        diff_n1_w_sum = 0
        diff_n2_w_sum = 0

        model.eval()
        for batch, trace_data in enumerate(tqdm(test_generator)):
            trace_data = trace_data.to(device=device)
            r_predicts, regs, dts = model(trace_data)
            weights = dts.cumsum(dim=0).reciprocal()
            r_trues = trace_data.transpose(0,1)[1][1:]
            r_predicts = r_predicts
            diff_n1 = torch.abs(r_predicts - r_trues)
            diff_n2 = torch.square(r_predicts - r_trues)
            diff_n1_w = diff_n1 * weights
            diff_n2_w = diff_n2 * weights

            diff_n1_sum += diff_n1.sum().item()
            diff_n2_sum += diff_n2.sum().item()
            diff_n1_w_sum += diff_n1_w.sum().item()
            diff_n2_w_sum += diff_n2_w.sum().item()

            for step in range(len(r_predicts)):
                writer.add_scalars(f"Trace_{batch} Pred&Truth", 
                                   {"Pred": r_predicts.tolist()[step], "Truth": r_trues.tolist()[step]}, 
                                   step)
                writer.add_scalars(f"Trace_{batch} Diffs", 
                                   {"diff_n1":diff_n1.tolist()[step], 
                                    "diff_n2":diff_n2.tolist()[step], 
                                    "diff_n1_w":diff_n1_w.tolist()[step], 
                                    "diff_n2_w":diff_n2_w.tolist()[step], }, 
                                   step)    
        diffs = {
            "diff_n1": diff_n1_sum / (batch + 1),
            "diff_n2": diff_n2_sum / (batch + 1),
            "diff_n1_w": diff_n1_w_sum / (batch + 1),
            "diff_n2_w": diff_n2_w_sum / (batch + 1),
        }
        print(diffs)
        
        # Flushes the event file to disk
        writer.flush()
    writer.close()
    return diffs

def parse_args():
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument("--dataset_path", type=str, required=True, help="Load from folder where`.pt` Dataset file is.")
    parser.add_argument("--model_path", type=str, default=None, help="Load Checkpoint Path.")
    parser.add_argument("--sample_eps", type=parser_utils.str2bool, default=False)
    parser.add_argument("--device", type=str, default="cpu", help="Device `cpu` or `cuda`")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s")

    logging.info(f"Loading Dataset from {args.dataset_path}")
    dataset = torch.load(args.dataset_path, map_location=torch.device(args.device))
    test_set = dataset["test_set"]
    test_generator = DataLoader(test_set, batch_size=None, shuffle=True, batch_sampler=None)

    if args.sample_eps:
        model = CIRNet(sigma_num=5, epsilon_num=1)
        params = torch.load(args.model_path, map_location=args.device)
        model.cir_cell.k.data = params["cir_cell.k"]
        model.cir_cell.theta.data = params["cir_cell.theta"]
        model.cir_cell.sigma_linear_layer.weight.data = params["cir_cell.sigma_linear_layer.weight"]
        model.cir_cell.sigma_linear_layer.bias.data = params["cir_cell.sigma_linear_layer.bias"]
        model.cir_cell.epsilon_linear_layer.weight.data = torch.tensor([[1.]]).to(device=torch.device(args.device))
    else:
        model = CIRNet(sigma_num=5, epsilon_num=1)
        save_and_load.load(model, path=args.model_path, device=args.device)

    logging.info("Start Test!")
    test(model=model,
         test_generator=test_generator,
         log_dir="log",
         device=torch.device(args.device))
    
    # Example: python test.py --dataset_path "data\dummyTrade1\dataset_splited.pt" --model_path "save\M1_final.pt"  --sample_eps False --device cpu
