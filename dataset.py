import os
import argparse
from typing import List, Tuple, Callable, Union

import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.Info import *
from utils.pd_ops import *

# Put raw csv file at the same path
raw_csv_infolist = InfoList([
    Info("market-data-swap-rates", {"MDSR", "Market-Data-Swap-Rates", "market_data_swap_rates"}, df=pd.read_csv("market-data-swap-rates.csv")),
    Info("market-data-swaption-vols", {"MDSV", "Market-Data-Swaption-Vols", "market_data_swaption_vols"}, df=pd.read_csv("market-data-swaption-vols.csv")),
    Info("trade-information", {"TI", "Trade-Information", "trade_information"}, df=pd.read_csv("trade-information.csv")),
    Info("trade-price-ir-vegas", {"TPIV", "Trade=Price-Ir-Vegas", "trade_price_ir_vegas"}, df=pd.read_csv("trade-price-ir-vegas.csv"))
])


class TradeData:

    trade_tenor_map = map_dict(raw_csv_infolist["trade-information"].df, "trade name", "underlying", lambda v: v[v.rfind(":")+1:].lower())
    trade_expiry_map = map_dict(raw_csv_infolist["trade-information"].df, "trade name", "maturity", lambda v: v.lower())
    trade_ub_map = map_dict(raw_csv_infolist["trade-information"].df, "trade name", "upper_bound", lambda v: v)
    trade_lb_map = map_dict(raw_csv_infolist["trade-information"].df, "trade name", "lower_bound", lambda v: v)

    def __init__(self, trade_name:str) -> None:
        assert trade_name in self.trade_tenor_map, "trade name bot found"
        self.trade_name = trade_name

        self.tenor = self.trade_tenor_map[self.trade_name]
        self.expiry = self.trade_expiry_map[self.trade_name]

        self.upper_bound:int = self.trade_ub_map[self.trade_name]
        self.lower_bound:int = self.trade_lb_map[self.trade_name]

        self.sub_MDSR: pd.DataFrame
        self.sub_MDSV: pd.DataFrame
        self.sub_TPIV: pd.DataFrame
        self.joint_df: pd.DataFrame

    def get_sub_MDSR(self, sr_strategy:str="mean"):
        self.sub_MDSR = reset_copy(filter_dataframe(df=raw_csv_infolist["MDSR"].df, pattern=[("Tenor", self.tenor)]))
        if sr_strategy == "mean":
            # calculate mean swap rate per date range from different start day 
            self.sub_MDSR = self.sub_MDSR.groupby("Date")["Swap Rate"].mean().reset_index()
        if sr_strategy == "all":
            # preserve all data
            pass

    def get_sub_MDSV(self, vols_strategy:str="mean"):
        self.sub_MDSV = reset_copy(filter_dataframe(df=raw_csv_infolist["MDSV"].df, pattern=[("Tenor", self.tenor)]))
        if vols_strategy == "atm_only":
            # use "atm" only
            self.sub_MDSV = reset_copy(filter_dataframe(df=self.sub_MDSV, pattern=[("Strike", "atm")]))
        elif vols_strategy == "all":
            # preserve all data
            pass
        elif vols_strategy == "mean":
            # calculate mean vols per date range from different strikes and expiry  
            self.sub_MDSV = self.sub_MDSV.groupby('Date')['Vols'].mean().reset_index()
        
    def get_sub_TPIV(self, vega_strategy:str="mean"):
        self.sub_TPIV = reset_copy(filter_dataframe(df=raw_csv_infolist["TPIV"].df, pattern=[("Tenor Bucket", self.tenor), ("Expiry Bucket", self.expiry), ("Trade Name", self.trade_name)]))
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
            pass
        elif vega_strategy == "mean":
            # calculate mean Vega per date  
            self.sub_TPIV = self.sub_TPIV.groupby('Value Date')['Vega'].mean().reset_index()

    def process(self, sr_strategy:str="mean", vols_strategy:str="mean", vega_strategy:str="mean"):
        self.get_sub_MDSR(sr_strategy)
        self.get_sub_MDSV(vols_strategy)
        self.get_sub_TPIV(vega_strategy)
        self.sub_MDSR, self.sub_MDSV, self.sub_TPIV = filter_common([(self.sub_MDSR, "Date"), (self.sub_MDSV, "Date"), (self.sub_TPIV, "Value Date")])
        self.sub_TPIV = self.sub_TPIV.rename(columns={'Value Date': 'Date'})
        self.joint_df = join_common([(self.sub_MDSR, "Date"), (self.sub_MDSV, "Date"), (self.sub_TPIV, "Date")])

    def save(self, save_dir:str=os.curdir):
        save_dir = os.path.join(save_dir, self.trade_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.sub_MDSR.to_csv(os.path.join(save_dir, "MDSR.csv"))
        self.sub_MDSV.to_csv(os.path.join(save_dir, "MDSV.csv"))
        self.sub_TPIV.to_csv(os.path.join(save_dir, "TPIV.csv"))
        self.joint_df.to_csv(os.path.join(save_dir, "Joint.csv"))
    
    def load(self, load_dir:str):
        if not load_dir.endswith(self.trade_name):
            load_dir = os.path.join(load_dir, self.trade_name)
        self.sub_MDSR = pd.read_csv(os.path.join(load_dir, "MDSR.csv"))
        self.sub_MDSV = pd.read_csv(os.path.join(load_dir, "MDSV.csv"))
        self.sub_TPIV = pd.read_csv(os.path.join(load_dir, "TPIV.csv"))
        self.joint_df = pd.read_csv(os.path.join(load_dir, "Joint.csv"))

    def to_tensor(self, dtype=torch.float, device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")):
        return torch.tensor(convert_date_to_float(self.joint_df, col_name="Date").to_numpy(), dtype=dtype, device=device)
    
    def to_dataset(self, dtype=torch.float, device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")):
        return TradeDataset(self, dtype=dtype, device=device)
    

class TradeDataset(Dataset):
    def __init__(self, trade_data:TradeData, epsilon_num:int=1, dtype=torch.float, device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")) -> None:
        super().__init__()

        self.trade_data_ = trade_data
        self.trade_tensor = self.trade_data_.to_tensor(dtype=dtype, device=device)

        self.epsilon = torch.randn(len(self), epsilon_num).to(dtype=self.trade_tensor.dtype, device=self.trade_tensor.device)
        self.epsilon_num = epsilon_num
        self.trade_tensor = torch.cat((self.trade_tensor, self.epsilon), dim=1)
    
    def __len__(self):
        return len(self.trade_tensor)
    
    def __getitem__(self, index):
        date, swap_rate, vols, vega, *epsilon = self.trade_tensor[index]
        if self.epsilon_num == 1:
            return date, swap_rate, vols, vega, epsilon[0]
        return date, swap_rate, vols, vega, epsilon
    
    def load(self, path:str, dtype=torch.float, device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")):
        if path.endswith(".csv"):
            df = pd.read_csv(path)
            self.trade_tensor = torch.tensor(convert_date_to_float(df, col_name="Date").to_numpy(), dtype=dtype, device=device)
            self.trade_tensor = torch.cat((self.trade_data_, self.epsilon), dim=1)
        elif path.endswith(".pt"):
            self = torch.load(path, map_location=device)
        else:
            self.trade_data_ = self.trade_data_.load(path)
            self.trade_tensor = self.trade_data_.to_tensor(dtype=dtype, device=device)
            self.trade_tensor = torch.cat((self.trade_data_, self.epsilon), dim=1)

    def save(self, path):
        torch.save(self, path)

def parse_args():
    parser = argparse.ArgumentParser(description="Data Preprocess.")
    parser.add_argument("--trade", type=str, required=True, help="Trade info name. e.g. dummyTrade1")
    parser.add_argument("--load", type=str, default=None, help="(Optional) Load preprocessed trade data path.")
    parser.add_argument("--save", type=str, required=True, help="Dataset `.pt` File path")
    parser.add_argument("--device", type=str, default="cpu", help="Device `cpu` or `cuda`")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    trade_data = TradeData(args.trade)

    if args.load:
        trade_data.load(args.load)
    else:
        trade_data.process()

    train_set = trade_data.to_dataset(device=torch.device(args.device))
    train_set.save(args.save)

    #Example: python dataset.py --trade dummyTrade1 --save data\dataset.pt --device cpu 
