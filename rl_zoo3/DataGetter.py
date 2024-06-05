import os
from datetime import datetime, timezone, timedelta
from enum import Enum
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append('/Users/mahsaraeisinezhad/code/power-edge/stella-sandbox/src/')
from sbx.defaults import HISTORIC_CRYPTO

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Ticker(Enum):
    AVAX_USDT = "AVAX-USDT"
    BTC_USDT = "BTC-USDT"


class Granularity(Enum):
    s86400 = 86400


def get_path(ticker: Ticker, granularity: Granularity):
    return os.path.join(
        HISTORIC_CRYPTO, f"ticker={ticker.value}", f"granularity={granularity.value}"
    )


def read_parquet(ticker: Ticker, granularity: Granularity):
    df = pd.read_parquet(get_path(ticker, granularity))
    df["ticker"] = ticker.value
    df["granularity"] = timedelta(seconds=granularity.value)
    df["time"] = df["time"].apply(datetime.fromtimestamp)
    return df.set_index("time")


def load_data():
    tick = Ticker.AVAX_USDT
    dfs = {g: read_parquet(tick, g) for g in Granularity}
    df = dfs[Granularity.s86400]
    return df


class DataGetter:
    """
    The class for getting data for assets.
    """

    def __init__(
        self,
        asset="AVAX-USD",
        start_date=None,
        end_date=None,
        freq="1d",
        timeframes=[1, 2, 5, 10, 20, 40],
    ):

        self.asset = asset
        self.sd = start_date
        self.ed = end_date
        self.freq = freq

        self.timeframes = timeframes
        self.getData()

        self.scaler = StandardScaler()
        self.scaler.fit(self.data[:, 1:])

    def getData(self):
        asset = self.asset

        df = load_data()
        df_spy = df
        df.drop(columns=["ticker", "granularity"], inplace=True)

        # Reward - Not included in Observation Space.
        df["rf"] = df["close"].pct_change().shift(-1)

        # Returns and Trading Volume Changes
        for i in self.timeframes:
            df_spy[f"spy_ret-{i}"] = df_spy["close"].pct_change(i)
            df_spy[f"spy_v-{i}"] = df_spy["volume"].pct_change(i)

            df[f"r-{i}"] = df["close"].pct_change(i)
            df[f"v-{i}"] = df["volume"].pct_change(i)

        # Volatility
        for i in [5, 10, 20, 40]:
            df[f"sig-{i}"] = np.log(1 + df["r-1"]).rolling(i).std()

        # Moving Average Convergence Divergence (MACD)
        df["macd_lmw"] = df["r-1"].ewm(span=26, adjust=False).mean()
        df["macd_smw"] = df["r-1"].ewm(span=12, adjust=False).mean()
        df["macd_bl"] = df["r-1"].ewm(span=9, adjust=False).mean()
        df["macd"] = df["macd_smw"] - df["macd_lmw"]

        # Relative Strength Indicator (RSI)
        rsi_lb = 5
        pos_gain = df["r-1"].where(df["r-1"] > 0, 0).ewm(rsi_lb).mean()
        neg_gain = df["r-1"].where(df["r-1"] < 0, 0).ewm(rsi_lb).mean()
        rs = np.abs(pos_gain / neg_gain)
        df["rsi"] = 100 * rs / (1 + rs)

        # Bollinger Bands
        bollinger_lback = 10
        df["bollinger"] = df["r-1"].ewm(bollinger_lback).mean()
        df["low_bollinger"] = (
            df["bollinger"] - 2 * df["r-1"].rolling(bollinger_lback).std()
        )
        df["high_bollinger"] = (
            df["bollinger"] + 2 * df["r-1"].rolling(bollinger_lback).std()
        )

        # Check if columns exist in df_spy before merging
        spy_columns = [f"spy_ret-{i}" for i in self.timeframes] + [
            f"spy_sig-{i}" for i in [5, 10, 20, 40]
        ]
        if all(col in df_spy.columns for col in spy_columns):
            # Merge SP500 data into the main dataframe
            df = df.merge(
                df_spy[spy_columns], how="left", right_index=True, left_index=True
            )
        else:
            print("One or more required columns not found in df_spy.")

        # Filtering and Interpolation
        numerical_columns = df.select_dtypes(include=np.number).columns
        df[numerical_columns] = df[numerical_columns].interpolate(
            "linear", limit_direction="both"
        )
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        self.frame = df
        self.data = np.array(df.iloc[:, 6:])
        self.column_names = df.columns.tolist()
        df.to_csv("df_saved.csv", index=False)
        return

    def scaleData(self):
        self.scaled_data = self.scaler.fit_transform(self.data[:, 1:])
        # Concatenate scaled data with non-numeric columns and restore column names
        self.scaled_df = pd.DataFrame(self.scaled_data, columns=self.column_names[1:])
        self.scaled_df.insert(0, self.column_names[0], self.data[:, 0])
        print(self.scaled_df.shape)

        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, col_idx=None):
        if col_idx is None:
            return self.data[idx]
        elif col_idx < len(list(self.data.columns)):
            return self.data[idx][col_idx]
        else:
            raise IndexError
