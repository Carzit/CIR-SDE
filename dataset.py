import os
import argparse
import json
from typing import List, Tuple, Callable, Union, Dict
import logging

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from utils.Info import *
from utils.pd_ops import *

class Trades:
    def __init__(self, folder_path:str=os.curdir, auto=True) -> None:
        self.raw_csv_folder=folder_path

        assert os.path.exists(os.path.join(self.raw_csv_folder, "market-data-swap-rates.csv")), "`market-data-swap-rates.csv` not found"
        assert os.path.exists(os.path.join(self.raw_csv_folder, "market-data-swaption-vols.csv")), "`market-data-swaption-vols.csv` not found"
        assert os.path.exists(os.path.join(self.raw_csv_folder, "trade-information.csv")), "`trade-information.csv` not found"
        assert os.path.exists(os.path.join(self.raw_csv_folder, "trade-price-ir-vegas.csv")), "`trade-price-ir-vegas.csv` not found"

        self.MDSR_path = os.path.join(self.raw_csv_folder, "market-data-swap-rates.csv")
        self.MDSV_path = os.path.join(self.raw_csv_folder, "market-data-swaption-vols.csv")
        self.TI_path = os.path.join(self.raw_csv_folder, "trade-information.csv")
        self.TPIV_path = os.path.join(self.raw_csv_folder, "trade-price-ir-vegas.csv")

        self.raw_csv_infolist:InfoList
        self.trade_tenor_map:Dict[str, str]
        self.trade_expiry_map:Dict[str, str]
        self.trade_ub_map:Dict[str, float]
        self.trade_lb_map:Dict[str, float]

        if auto:
            self.get_infolist()
            self.get_trade_map()

    def get_infolist(self):
        self.raw_csv_infolist = InfoList([
            Info("market-data-swap-rates", {"MDSR", "Market-Data-Swap-Rates", "market_data_swap_rates"}, df=pd.read_csv(self.MDSR_path)),
            Info("market-data-swaption-vols", {"MDSV", "Market-Data-Swaption-Vols", "market_data_swaption_vols"}, df=pd.read_csv(self.MDSV_path)),
            Info("trade-information", {"TI", "Trade-Information", "trade_information"}, df=pd.read_csv(self.TI_path)),
            Info("trade-price-ir-vegas", {"TPIV", "Trade=Price-Ir-Vegas", "trade_price_ir_vegas"}, df=pd.read_csv(self.TPIV_path))
            ])
        return self.raw_csv_infolist
    
    def get_trade_map(self):
        self.trade_tenor_map = map_dict(self.raw_csv_infolist["trade-information"].df, "trade name", "underlying", lambda v: v[v.rfind(":")+1:].lower())
        self.trade_expiry_map = map_dict(self.raw_csv_infolist["trade-information"].df, "trade name", "maturity", lambda v: v.lower())
        self.trade_ub_map = map_dict(self.raw_csv_infolist["trade-information"].df, "trade name", "upper_bound", lambda v: v)
        self.trade_lb_map = map_dict(self.raw_csv_infolist["trade-information"].df, "trade name", "lower_bound", lambda v: v)

class TradeData:

    def __init__(self, trade_name:str, trades:Trades=None, load_dir:str=None) -> None:

        self.trade_name:str = trade_name # name of the trade, e.g. "dummyTrade1"

        self.tenor:str # tenor matching identifier, `underlying` information, e.g. "2Y"
        self.expiry:str # expiry matching identifier, `maturity` information, e.g. "5Y"

        self.upper_bound:int # upper bound, e.g. 0.0379
        self.lower_bound:int # lower bound, e.g. 0.0042

        self.sub_MDSR: pd.DataFrame # sub DataFrame derived from `market-data-swap-rates.csv`
        self.sub_MDSV: pd.DataFrame # sub DataFrame derived from `market-data-swaption-vols.csv`
        self.sub_TPIV: pd.DataFrame # sub DataFrame derived from `trade-information.csv`
        self.joint_df: pd.DataFrame # sub DataFrame derived from `trade-price-ir-vegas.csv`

        if load_dir is not None and trades is None:
            self.load(load_dir) 
        elif load_dir is None and trades is not None:
            assert trade_name in trades.trade_tenor_map, "trade name bot found"

            self.tenor = trades.trade_tenor_map[self.trade_name]
            self.expiry = trades.trade_expiry_map[self.trade_name]

            self.upper_bound = trades.trade_ub_map[self.trade_name]
            self.lower_bound = trades.trade_lb_map[self.trade_name]

            self.sub_MDSR = reset_copy(filter_dataframe(df=trades.raw_csv_infolist["MDSR"].df, pattern=[("Tenor", self.tenor)]))
            self.sub_MDSV = reset_copy(filter_dataframe(df=trades.raw_csv_infolist["MDSV"].df, pattern=[("Tenor", self.tenor),("Expiry", self.expiry)]))
            self.sub_TPIV = reset_copy(filter_dataframe(df=trades.raw_csv_infolist["TPIV"].df, pattern=[("Tenor Bucket", self.tenor), ("Expiry Bucket", self.expiry), ("Trade Name", self.trade_name)]))
        else:
            raise ValueError("`trades` and `load_dir` must be specified either, but not both.")     

    def process_sub_MDSR(self, sr_strategy:str="mean"):
        if sr_strategy == "mean":
            # calculate mean swap rate per date range from different start day 
            self.sub_MDSR = self.sub_MDSR.groupby("Date")["Swap Rate"].mean().reset_index()
        if sr_strategy == "all":
            # preserve all data
            pass

    def process_sub_MDSV(self, vols_strategy:str="mean"):
        if vols_strategy == "atm_only":
            # use "atm" only
            self.sub_MDSV = reset_copy(filter_dataframe(df=self.sub_MDSV, pattern=[("Strike", "atm")]))
        elif vols_strategy == "all":
            # preserve all data
            self.sub_MDSV = pivot_labels_to_columns(df=self.sub_MDSV, date_col="Date", label_col="Strike", value_col="Vols")
        elif vols_strategy == "mean":
            # calculate mean vols per date range from different strikes and expiry  
            self.sub_MDSV = self.sub_MDSV.groupby('Date')['Vols'].mean().reset_index()
        
    def process_sub_TPIV(self, vega_strategy:str="mean"):
        if vega_strategy == "zero_only":
            # use Zero Rate Shock == 0 only
            self.sub_TPIV = reset_copy(filter_dataframe(df=self.sub_TPIV, pattern=[("Zero Rate Shock", 0)]))
        elif vega_strategy == "extreme_low":
            # use Zero Rate Shock == -100 only
            self.sub_TPIV = reset_copy(filter_dataframe(df=self.sub_TPIV, pattern=[("Zero Rate Shock", -100)]))
        elif vega_strategy == "random":
            self.sub_TPIV = reset_copy(self.sub_TPIV.groupby('Value Date')['Vega'].apply(group_sample).reset_index())
        elif vega_strategy == "all":
            # preserve all data
            self.sub_TPIV = pivot_labels_to_columns(df=self.sub_TPIV, date_col="Value Date", label_col="Zero Rate Shock", value_col="Vega")
        elif vega_strategy == "mean":
            # calculate mean Vega per date  
            self.sub_TPIV = self.sub_TPIV.groupby('Value Date')['Vega'].mean().reset_index()
        elif vega_strategy == "drop":
            self.sub_TPIV = self.sub_TPIV["Value Date"].reset_index(drop=True)

    def process(self, sr_strategy:str="mean", vols_strategy:str="mean", vega_strategy:str="mean"):
        self.process_sub_MDSR(sr_strategy)
        self.process_sub_MDSV(vols_strategy)
        self.process_sub_TPIV(vega_strategy)

        if vega_strategy == "drop":
            self.sub_MDSR, self.sub_MDSV = filter_common([(self.sub_MDSR, "Date"), (self.sub_MDSV, "Date")])
            self.joint_df = join_common([(self.sub_MDSR, "Date"), (self.sub_MDSV, "Date")])
        else:
            self.sub_MDSR, self.sub_MDSV, self.sub_TPIV = filter_common([(self.sub_MDSR, "Date"), (self.sub_MDSV, "Date"), (self.sub_TPIV, "Value Date")])
            self.sub_TPIV = self.sub_TPIV.rename(columns={'Value Date': 'Date'})
            self.joint_df = join_common([(self.sub_MDSR, "Date"), (self.sub_MDSV, "Date"), (self.sub_TPIV, "Date")])

    def save(self, save_dir:str=os.curdir):
        save_dir = os.path.join(save_dir, self.trade_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(os.path.join(save_dir, "info.json"), "w") as f:
            json.dump({"name": self.trade_name,
                       "tenor": self.tenor,
                       "expiry": self.expiry,
                       "upper bound": self.upper_bound,
                       "lower bound": self.lower_bound}, f)

        self.sub_MDSR.to_csv(os.path.join(save_dir, "MDSR.csv"))
        self.sub_MDSV.to_csv(os.path.join(save_dir, "MDSV.csv"))
        self.sub_TPIV.to_csv(os.path.join(save_dir, "TPIV.csv"))
        self.joint_df.to_csv(os.path.join(save_dir, "Joint.csv"))
    
    def load(self, load_dir:str):
        if not load_dir.endswith(self.trade_name):
            load_dir = os.path.join(load_dir, self.trade_name)

        with open(os.path.join(load_dir, "info.json"), "w") as f:
            info = json.load(f)
        
        assert self.trade_name == info["name"], "Unmatched trade name"

        self.tenor = info["tenor"]
        self.expiry = info["expiry"]
        self.upper_bound = info["upper bound"]
        self.lower_bound = info["lower bound"]
        
        self.sub_MDSR = pd.read_csv(os.path.join(load_dir, "MDSR.csv"))
        self.sub_MDSV = pd.read_csv(os.path.join(load_dir, "MDSV.csv"))
        self.sub_TPIV = pd.read_csv(os.path.join(load_dir, "TPIV.csv"))
        self.joint_df = pd.read_csv(os.path.join(load_dir, "Joint.csv"))

    def to_tensor(self, dtype=torch.float, device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")):
        return torch.tensor(convert_date_to_float(self.joint_df, col_name="Date").to_numpy(), 
                            dtype=dtype, 
                            device=device)
    
    def to_dataset(self, epsilon_num:int=1, mode="trace", dtype=torch.float, device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")):
        return TradeDataset(trade_data=self, 
                            epsilon_num=epsilon_num, 
                            mode=mode, 
                            dtype=dtype, 
                            device=device)
    

class TradeDataset(Dataset):
    def __init__(self, trade_data:TradeData=None, epsilon_num:int=1, mode="trace", dtype=torch.float, device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")) -> None:
        super().__init__()

        self.trade_data_:TradeData
        self.trade_tensor:torch.Tensor

        self.epsilon_num:int = epsilon_num
        self.epsilon:torch.Tensor 

        self.mode:str = mode

        if trade_data is not None:
            self.trade_data_ = trade_data
            self.trade_tensor = self.trade_data_.to_tensor(dtype=dtype, device=device)
            self.epsilon = torch.randn(len(self.trade_tensor), self.epsilon_num).to(dtype=self.trade_tensor.dtype, device=self.trade_tensor.device)
            self.trade_tensor = torch.cat((self.trade_tensor, self.epsilon), dim=1)
               
    def __len__(self):
        if self.mode == "trace":
            return len(self.trade_tensor)-1
        elif self.mode == "step":
            return len(self.trade_tensor)
        else:
            raise NotImplementedError("mode support `trace` and `step` only")
    
    def __getitem__(self, index):
        if self.mode == "trace":
            return self.trade_tensor[index:]
        elif self.mode == "step":
            return self.trade_tensor[index]
        else:
            raise NotImplementedError("mode support `trace` and `step` only")
    
    def load(self, path:str, device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")):
        if path.endswith(".pt"): # load from pt file
            self = torch.load(path, map_location=device)
        else:
            raise ValueError("Must load .pt file")

    def save(self, path):
        torch.save(self, path)

    def split_dataset(self, lengths:list, generator:torch.Generator=None):
        total_length = len(self)
        split_lengths = list(map(lambda x:round(x / sum(lengths) * total_length), lengths))
        split_lengths[0] = total_length - sum(split_lengths[1:])
        return random_split(self, split_lengths, generator=generator)
    

def split_dataset(dataset:Dataset, lengths:list, generator:torch.Generator=None):
    total_length = len(dataset)
    split_lengths = list(map(lambda x:round(x / sum(lengths) * total_length), lengths))
    split_lengths[0] = total_length - sum(split_lengths[1:])
    return random_split(dataset, split_lengths, generator=generator)

def parse_args():
    parser = argparse.ArgumentParser(description="Data Preprocess.")
    parser.add_argument("--trade_name", type=str, required=True, help="Trade info name. e.g. dummyTrade1")
    parser.add_argument("--data_folder", type=str, default=None, help="(Optional) Load raw trade data path.")
    parser.add_argument("--load_folder", type=str, default=None, help="(Optional) Load preprocessed trade data path.")
    parser.add_argument("--epsilon_num", type=int, default=1, help="Number of random samplings of Gaussian noise")
    parser.add_argument("--split_ratio", type=int, nargs=3, default=[0.7, 0.2, 0.1], help="Train-Val-Test Split Rate")
    parser.add_argument("--save_folder", type=str, required=True, help="Dataset `.pt` File path")
    parser.add_argument("--device", type=str, default="cpu", help="Device `cpu` or `cuda`")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args() 

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s")
    
    if args.load_folder is not None and args.data_folder is None:
        logging.info(f"Loading {args.trade_name} Data from {args.load_folder}")
        trade_data = TradeData(args.trade_name, load_dir=args.load_folder)
    elif args.load_folder is None and args.data_folder is not None:
        logging.info(f"Loading {args.trade_name} Data from {args.data_folder}")
        trades = Trades(args.data_folder)
        trade_data = TradeData(args.trade_name, trades=trades)
        trade_data.process(vols_strategy="all", vega_strategy="drop")
    else:
        raise ValueError("`data_folder` and `load_folder` must be specified either, but not both.")  
    
    trade_data.save(args.save_folder)
    logging.info(f"Processed Data files saved into `{os.path.join(args.save_folder,args.trade_name)}`")

    logging.info(f"Constructing Dataset with {args.epsilon_num} epsilons sampled and use device {args.device}")
    dataset = trade_data.to_dataset(epsilon_num=args.epsilon_num, device=torch.device(args.device))
    dataset.save(os.path.join(args.save_folder, args.trade_name, 'dataset.pt'))
    logging.info(f"Dataset saved to `{os.path.join(args.save_folder, args.trade_name, 'dataset.pt')}`")

    logging.info(f"Spliting Dataset into `train_set`, `val_set` and `test_set` in proportion {args.split_ratio}")
    train_set, val_set, test_set = split_dataset(dataset, args.split_ratio)
    save_sets = {"train_set":train_set,
                 "val_set":val_set,
                 "test_set":test_set}
    torch.save(save_sets, os.path.join(args.save_folder, args.trade_name, 'dataset_splited.pt'))
    logging.info(f"Splited Dataset saved to `{os.path.join(args.save_folder, args.trade_name, 'dataset_splited.pt')}`")
    #Example: python dataset.py --trade_name "dummyTrade1" --data_folder "data" --save_folder "data" --device cpu
