from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import pprint

### plt設定
frame_lw = 1.0
plt.rcParams['xtick.direction'] = 'in' # 目盛内向き
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.minor.visible'] = True # 小目盛表示
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['xtick.major.size'] =  20.0 # 目盛の長さ
plt.rcParams['ytick.major.size'] =  20.0
plt.rcParams['xtick.minor.size'] =  10.0
plt.rcParams['ytick.minor.size'] =  10.0
plt.rcParams['xtick.major.width'] =  frame_lw # 目盛の太さ
plt.rcParams['ytick.major.width'] =  frame_lw
plt.rcParams['xtick.minor.width'] =  frame_lw
plt.rcParams['ytick.minor.width'] =  frame_lw
plt.rcParams['mathtext.fontset'] =  'cm'
plt.rcParams['axes.linewidth'] =  frame_lw # 外枠の太さ
plt.rcParams['axes.axisbelow'] = True # グリッド線を最背面に
plt.rcParams["legend.fancybox"] = False # 角が四角に
plt.rcParams["legend.framealpha"] = 1 # 透明度
plt.rcParams["legend.edgecolor"] = 'black' # edgeの色
### よく使う色
red = '#E95464'
green = '#00AF55'
blue = '#4179F7'
orange = '#F08300'
yellow = '#FFEC47'
purple = '#7058A3'
ebicha = '#6c2c2f'
momo = '#e198b4'
fuji = '#5a5359'

# --------------------------
# 基本IO
# --------------------------
def read_timeseries_csv(path: str | Path) -> pd.DataFrame:
    """
    CSVは2列: Time, Value を想定
    Timeはパースしてindexに置く
    """
    df = pd.read_csv(path)
    if not {"Time", "Value"}.issubset(df.columns):
        raise ValueError(f"{path} must have columns Time, Value. got={df.columns.tolist()}")
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values("Time").set_index("Time")
    # Valueを数値に（変な文字が混じってもNaNに）
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    return df


def downsample_every_n(df: pd.DataFrame, n: int = 600) -> pd.DataFrame:
    """
    600行刻み（1秒サンプルなら10分）で間引き。
    ※等間隔じゃない場合もあるので、まずは単純に行間引き。
    """
    if len(df) == 0:
        return df
    return df.iloc[::n].copy()


def align_on_common_time(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    4系列をTime indexでouter join→必要なら欠損を残す（まずは事実を尊重）
    """
    out = None
    for k, df in dfs.items():
        s = df["Value"].rename(k)
        out = s.to_frame() if out is None else out.join(s, how="outer")
    return out.sort_index()


# --------------------------
# ループ解析
# --------------------------
@dataclass
class LoopFiles:
    pv: str
    sv: str
    mv: str
    mode: str


def load_loop(loop: LoopFiles, base_dir: str | Path = ".",
              downsample_n: int = 600) -> pd.DataFrame:
    base_dir = Path(base_dir)

    dfs = {
        "PV": downsample_every_n(read_timeseries_csv(base_dir / loop.pv), downsample_n),
        "SV": downsample_every_n(read_timeseries_csv(base_dir / loop.sv), downsample_n),
        "MV": downsample_every_n(read_timeseries_csv(base_dir / loop.mv), downsample_n),
        "MODE": downsample_every_n(read_timeseries_csv(base_dir / loop.mode), downsample_n),
    }

    data = align_on_common_time(dfs)

    # MODEは本来カテゴリか0/1/数値のはず。ここでは数値化しておく
    # 文字列だった場合はNaNになるけど、その場合は先にCSV側を確認してね
    data["MODE"] = pd.to_numeric(data["MODE"], errors="coerce")

    # ★追加：偏差 e = SV - PV（両方あるところだけ）
    data["E"] = data["SV"] - data["PV"]
    
    return data


def plot_loop(data: pd.DataFrame, title: str = "", max_points: Optional[int] = None) -> None:
    """
    PV/SV/MV/MODEを同じ時間軸で表示
    """
    if max_points is not None and len(data) > max_points:
        data = data.iloc[:max_points]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(data.index.to_numpy(), data["PV"].to_numpy(), label="PV", c=red)
    ax1.plot(data.index.to_numpy(), data["SV"].to_numpy(), label="SV", c=green, ls='dashed')
    ax1.set_ylabel("PV / SV")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(data.index.to_numpy(), data["MV"].to_numpy(), label="MV", c=blue, alpha=0.8)
    ax2.set_ylabel("MV")
    ax2.legend(loc="upper right")
    ax1.grid(color='k', linestyle='dotted', linewidth=frame_lw, axis='both')
    ax1.tick_params(labelsize=15)
    ax1.xaxis.set_ticks_position('both')
    plt.title(title or "Control Loop (PV/SV/MV)")
    plt.tight_layout()
    plt.show()

    # ★3) 偏差 e = SV - PV
    fig, ax = plt.subplots(figsize=(12, 3.5))
    # 欠損があると線が途切れるけど、それが「どこで揃ってないか」の情報にもなる
    ax.plot(data.index.to_numpy(), data["E"].to_numpy(), label="E = SV - PV", c=orange)
    ax.axhline(0.0, linestyle="--", linewidth=1, c='k')
    ax.set_ylabel("Error (SV - PV)")
    ax.set_title((title + "  ") if title else "" + "Error (SV - PV)")
    plt.tight_layout()
    ax.tick_params(labelsize=15)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.grid(color='k', linestyle='dotted', linewidth=frame_lw, axis='both')
    plt.show()

    # MODEを別図で
    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.plot(data.index.to_numpy(), data["MODE"].to_numpy(), label="MODE",c=red)
    ax.set_ylabel("MODE")
    ax.set_title((title + "  ") if title else "" + "MODE")
    ax.tick_params(labelsize=15)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.grid(color='k', linestyle='dotted', linewidth=frame_lw, axis='both')
    plt.tight_layout()
    plt.show()


def lag_correlation(x: pd.Series, y: pd.Series, max_lag_steps: int = 200) -> Tuple[int, float, pd.DataFrame]:
    """
    ラグ相関:
      corr(lag) = corr( x(t), y(t+lag) )
    戻り: (best_lag, best_corr, all_corr_df)
    """
    # 同じindexに揃える
    z = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(z) < 10:
        raise ValueError("Not enough overlap after dropna()")

    xs = z["x"].values
    ys = z["y"].values

    rows = []
    for lag in range(-max_lag_steps, max_lag_steps + 1):
        if lag < 0:
            a = xs[:lag]      # shorter
            b = ys[-lag:]
        elif lag > 0:
            a = xs[lag:]
            b = ys[:-lag]
        else:
            a = xs
            b = ys

        if len(a) < 10:
            continue

        c = np.corrcoef(a, b)[0, 1]
        rows.append((lag, c, len(a)))

    corr_df = pd.DataFrame(rows, columns=["lag_steps", "corr", "n"])
    # 絶対値最大のラグを採用（符号も見たいのでcorrも返す）
    best = corr_df.iloc[corr_df["corr"].abs().argmax()]
    return int(best["lag_steps"]), float(best["corr"]), corr_df


def analyze_loop(loop: LoopFiles, base_dir: str | Path = ".",
                 downsample_n: int = 600,
                 max_lag_steps: int = 200,
                 title: str = "") -> None:
    """
    1) 読み込み＆ダウンサンプル
    2) プロット
    3) MV→PV の遅れ推定（ラグ相関）
    """
    data = load_loop(loop, base_dir=base_dir, downsample_n=downsample_n)
    plot_loop(data, title=title)

    # MVがPVに効くなら「MVが先、PVが後」なので、MV(t) と PV(t+lag) を見る
    best_lag, best_corr, corr_df = lag_correlation(data["MV"], data["PV"], max_lag_steps=max_lag_steps)

    # downsampleが600行刻み（=10分）前提なら lag_steps*10分 がざっくり遅れ
    approx_minutes = best_lag * 10

    print("=== Lag correlation (MV -> PV) ===")
    print(f"best_lag_steps = {best_lag}, best_corr = {best_corr:.4f}")
    print(f"approx delay = {approx_minutes} minutes (assuming 1s sampling and downsample_n=600)")

    # 相関カーブも見せる
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(corr_df["lag_steps"].to_numpy(), corr_df["corr"].to_numpy(), c=red)
    ax.axvline(best_lag, linestyle="--",c='green')
    ax.set_title((title + "  ") if title else "" + "Lag Corr: corr(MV(t), PV(t+lag))")
    ax.set_xlabel("lag_steps")
    ax.set_ylabel("corr")
    fig.tight_layout()
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.grid(color='k', linestyle='dotted', linewidth=frame_lw, axis='both')
    plt.show()

def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    s = s.dropna()
    if len(s) < 3:
        return s
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return s * 0.0
    return (s - mu) / sd


def linear_fit_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """
    y ≈ a*x + b の最小二乗フィットと R^2 を返す
    """
    a, b = np.polyfit(x, y, 1)
    yhat = a * x + b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(a), float(b), float(r2)


def compare_series_similarity(
    data: pd.DataFrame,
    col_x: str,
    col_y: str,
    max_lag_steps: int = 200,
    title: str = "",
    standardize: bool = True,
) -> dict:
    """
    col_x と col_y の「似てる度」を定量化：
      - 0遅れ相関
      - ラグ相関（最適ラグ）
      - 線形フィット R^2
      - 時系列重ね描き（標準化あり）
      - 散布図
    """
    xy = data[[col_x, col_y]].copy()
    xy[col_x] = pd.to_numeric(xy[col_x], errors="coerce")
    xy[col_y] = pd.to_numeric(xy[col_y], errors="coerce")
    xy = xy.dropna()
    if len(xy) < 20:
        raise ValueError(f"Not enough overlap points for {col_x} vs {col_y}: n={len(xy)}")

    # 標準化（形だけ比較したい時）
    if standardize:
        x_s = zscore(xy[col_x])
        y_s = zscore(xy[col_y])
        # zscoreでdropnaしたので再アライン
        z = pd.concat([x_s.rename("x"), y_s.rename("y")], axis=1).dropna()
        x = z["x"]
        y = z["y"]
    else:
        x = xy[col_x]
        y = xy[col_y]

    # 0遅れ相関
    corr0 = float(np.corrcoef(x.values, y.values)[0, 1])

    # ラグ相関（x(t) と y(t+lag)）
    best_lag, best_corr, corr_df = lag_correlation(x, y, max_lag_steps=max_lag_steps)

    # 線形フィット（同時刻での比例っぽさ）
    a, b, r2 = linear_fit_r2(x.values, y.values)

    # 時系列重ね描き（形の比較）
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x.index.to_numpy(), x.values, label=col_x + (" (z)" if standardize else ""),c=blue)
    ax.plot(y.index.to_numpy(), y.values, label=col_y + (" (z)" if standardize else ""),c=green)
    ax.set_title(title or f"Overlay: {col_x} vs {col_y}" + (" [standardized]" if standardize else ""))
    ax.legend()
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.grid(color='k', linestyle='dotted', linewidth=frame_lw, axis='both')
    plt.show()

    # 散布図
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(x.values, y.values, s=6, c=red)
    ax.set_title(title or f"Scatter: {col_x} vs {col_y}")
    ax.set_xlabel(col_x + (" (z)" if standardize else ""))
    ax.set_ylabel(col_y + (" (z)" if standardize else ""))
    fig.tight_layout()
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.grid(color='k', linestyle='dotted', linewidth=frame_lw, axis='both')
    plt.show()

    # ラグ相関カーブ
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(corr_df["lag_steps"], corr_df["corr"], c=red)
    ax.axvline(best_lag, linestyle="--",c=green)
    ax.set_title(title or f"Lag Corr: corr({col_x}(t), {col_y}(t+lag))")
    ax.set_xlabel("lag_steps")
    ax.set_ylabel("corr")
    fig.tight_layout()
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.grid(color='k', linestyle='dotted', linewidth=frame_lw, axis='both')
    plt.show()

    # 結果
    result = {
        "col_x": col_x,
        "col_y": col_y,
        "standardize": standardize,
        "corr_at_0_lag": corr0,
        "best_lag_steps": best_lag,
        "best_corr": float(best_corr),
        "linear_fit_a": a,
        "linear_fit_b": b,
        "linear_fit_r2": r2,
        "n_points": int(len(x)),
    }

    print("=== Similarity ===")
    pprint.pprint(result)
    return result

def compare_two_files(
    file_x: str,
    file_y: str,
    base_dir: str | Path = ".",
    downsample_n: int = 600,
    max_lag_steps: int = 200,
    title: str = "",
    ):
    base_dir = Path(base_dir)
    x = downsample_every_n(read_timeseries_csv(base_dir / file_x), downsample_n)["Value"].rename("X")
    y = downsample_every_n(read_timeseries_csv(base_dir / file_y), downsample_n)["Value"].rename("Y")
    data = pd.concat([x, y], axis=1).sort_index()

    return compare_series_similarity(
        data=data,
        col_x="X",
        col_y="Y",
        max_lag_steps=max_lag_steps,
        title=title or f"{file_x} vs {file_y}",
        standardize=True,
    )


# --------------------------
# 使い方例（最初の1本）
# --------------------------
if __name__ == "__main__":
    BASE_DIR = "./加熱炉データ/データ/"
    loop = LoopFiles(
        pv="加熱帯上部炉温制御：PV（℃）_1.csv",
        sv="加熱帯上部炉温制御：SV（℃）_1.csv",
        mv="加熱帯上部炉温制御：MV（%）.csv",
        mode="加熱帯上部炉温制御：MODE.csv",
    )
    analyze_loop(loop, base_dir=BASE_DIR, downsample_n=600, max_lag_steps=200, title="加熱帯上部炉温制御")