import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np

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

### df
df = pd.read_csv('merged_downsampled.csv')
# df = df[pd.to_datetime(df["Time"]) >= pd.Timestamp("2026-01-31")]
time = (pd.to_datetime(df['Time'])).to_numpy()

### よく使う関数
def corr_report(
    col_x: str,
    col_y: str,
    time_col: str = "Time",
    lags: int = 30,
) -> dict:
    """
    df内の2列について、相関・差分相関・誤差指標・ラグ相関などをまとめた辞書を返す。
    time_colでソートしてから計算（datetime想定）。
    """
    # --- prep ---
    cols = [time_col, col_x, col_y]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")

    d = df[cols].copy()
    d = d.sort_values(time_col)
    d = d.dropna(subset=[col_x, col_y])  # 相関は欠損に弱いので揃える

    n = int(len(d))
    if n < 2:
        return {
            "cols": (col_x, col_y),
            "n": n,
            "warning": "not enough rows after dropna",
        }

    x = d[col_x].astype(float)
    y = d[col_y].astype(float)

    # --- base correlations ---
    pearson = float(x.corr(y, method="pearson"))
    spearman = float(x.corr(y, method="spearman"))

    # --- difference / returns correlations (time series trap mitigation) ---
    dx = x.diff()
    dy = y.diff()
    diff_n = int(pd.concat([dx, dy], axis=1).dropna().shape[0])
    diff_pearson = float(dx.corr(dy, method="pearson")) if diff_n >= 2 else np.nan
    diff_spearman = float(dx.corr(dy, method="spearman")) if diff_n >= 2 else np.nan

    rx = x.pct_change().replace([np.inf, -np.inf], np.nan)
    ry = y.pct_change().replace([np.inf, -np.inf], np.nan)
    ret_n = int(pd.concat([rx, ry], axis=1).dropna().shape[0])
    ret_pearson = float(rx.corr(ry, method="pearson")) if ret_n >= 2 else np.nan
    ret_spearman = float(rx.corr(ry, method="spearman")) if ret_n >= 2 else np.nan

    # --- error metrics (how close in value) ---
    err = x - y
    mae = float(err.abs().mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    bias = float(err.mean())
    std_err = float(err.std(ddof=1)) if n >= 2 else np.nan

    # --- linear fit (y ≈ alpha + beta*x) + R^2 ---
    # numpy polyfit (deg=1) gives slope, intercept
    beta, alpha = np.polyfit(x.to_numpy(), y.to_numpy(), 1)
    y_hat = alpha + beta * x
    resid = y - y_hat
    ss_res = float(np.sum((resid.to_numpy()) ** 2))
    ss_tot = float(np.sum((y.to_numpy() - y.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else np.nan

    # --- lag correlations (Pearson & Spearman) ---
    # k > 0 means y shifted forward (y(t+k)) compared to x(t)
    def _lag_corr_series(x_s: pd.Series, y_s: pd.Series, method: str) -> pd.Series:
        vals = {}
        for k in range(-lags, lags + 1):
            vals[k] = x_s.corr(y_s.shift(k), method=method)
        return pd.Series(vals, dtype="float64")

    lag_pearson = _lag_corr_series(x, y, "pearson")
    lag_spearman = _lag_corr_series(x, y, "spearman")

    # best lags (ignore all-nan / ties)
    best_lag_p = int(lag_pearson.idxmax()) if lag_pearson.notna().any() else None
    best_lag_s = int(lag_spearman.idxmax()) if lag_spearman.notna().any() else None

    # --- package ---
    return {
        "cols": (col_x, col_y),
        "time_col": time_col,
        "n": n,
        "corr": {
            "pearson": pearson,
            "spearman": spearman,
        },
        "diff_corr": {
            "n": diff_n,
            "pearson": diff_pearson,
            "spearman": diff_spearman,
        },
        "return_corr": {
            "n": ret_n,
            "pearson": ret_pearson,
            "spearman": ret_spearman,
        },
        "error": {
            "mae": mae,
            "rmse": rmse,
            "bias_x_minus_y": bias,
            "std": std_err,
        },
        "linear_fit_y_on_x": {
            "alpha_intercept": float(alpha),
            "beta_slope": float(beta),
            "r2": r2,
        },
        "lag_corr": {
            "lags": lags,
            "pearson": {
                "best_lag": best_lag_p,
                "best_corr": float(lag_pearson.max()) if lag_pearson.notna().any() else np.nan,
                "series": lag_pearson,  # pd.Seriesで返す（必要なら.to_dict()に）
            },
            "spearman": {
                "best_lag": best_lag_s,
                "best_corr": float(lag_spearman.max()) if lag_spearman.notna().any() else np.nan,
                "series": lag_spearman,
            },
        },
    }

def plot_control_loop(pv=None, sv=None, mv=None, title=None, pv_unit=None, mv_unit=None, mv_min=None, mv_max=None):
    fig = plt.figure(figsize=(16, 9),tight_layout=True)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    if pv is not None:
        ax1.plot(time, df[pv].to_numpy(), label='PV', lw=2, c=red)
    if sv is not None:
        ax1.step(time, df[sv].to_numpy(), label='SV', linestyle='--', lw=2,c=green, where='post')
    if mv is not None:
        ax2.plot(time, df[mv].to_numpy(), label='MV', lw=2, c=blue, alpha=0.7)
        if mv_min is not None and mv_max is not None:
            ax2.set_ylim(mv_min, mv_max)    
    ax1.set_xlabel('時刻', fontsize=15)
    ax1.set_ylabel(f'PV / SV ({pv_unit})', fontsize=15)
    ax2.set_ylabel(f'MV ({mv_unit})', fontsize=15)
    ax1.grid(color='k', linestyle='dotted', linewidth=frame_lw, axis='both')
    ax1.set_title(f'{title}', fontsize='xx-large')
    ax1.tick_params(labelsize=15)
    ax2.tick_params(labelsize=15)
    ax1.xaxis.set_ticks_position('both')
    legend2 = ax2.legend(loc='upper right', fontsize=15)
    h1, l1 = ax1.get_legend_handles_labels()
    ax2.legend(h1, l1, loc='upper left', fontsize=15)
    ax2.add_artist(legend2)
    plt.show()

    fig = plt.figure(figsize=(16, 9),tight_layout=True)
    ax1 = fig.add_subplot(111)
    e = df[sv].to_numpy()-df[pv].to_numpy()
    bias = np.nanmean(e)
    mae  = np.nanmean(np.abs(e))
    rmse = np.sqrt(np.nanmean(e**2))
    ax1.plot(time, e, label=f'偏差', lw=2, c=red)
    ax1.axhline(0, color='k', linestyle='--', lw=frame_lw)
    ax1.axhline(bias, color=orange, linestyle='--', lw=1.5, label=f'Bias={bias:.2f} {pv_unit}')
    # ax1.axhline(mae, color=blue, linestyle=':', lw=1.5, label=f'MAE={mae:.2f} {pv_unit}')
    # ax1.axhline(-mae, color=blue, linestyle=':', lw=1.5)
    # ax1.axhline(rmse, color=green, linestyle=':', lw=1.5, label=f'RMSE={rmse:.2f} {pv_unit}')
    # ax1.axhline(-rmse, color=green, linestyle=':', lw=1.5)
    ax1.set_xlabel('時刻', fontsize=15)
    ax1.set_ylabel(f'偏差 ({pv_unit})', fontsize=15)
    ax1.grid(color='k', linestyle='dotted', linewidth=frame_lw, axis='both')
    ax1.set_title(f'{title}の偏差', fontsize='xx-large')
    ax1.tick_params(labelsize=15)
    ax1.xaxis.set_ticks_position('both')
    ax1.legend(loc='upper right', fontsize=15)
    plt.show()


def plot_compare(col1,col2,ubound=None,lbound=None):
    fig = plt.figure(figsize=(16, 9),tight_layout=True)
    ax1 = fig.add_subplot(111)

    ax1.plot(time, df[col1].to_numpy(), label=f'{col1}', lw=2, c=red)
    ax1.plot(time, df[col2].to_numpy(), label=f'{col2}', linestyle='--', lw=2,c=green)

    ax1.set_xlabel('時刻', fontsize=15)
    ax1.set_ylabel(f'Value', fontsize=15)
    if ubound is not None:
        ax1.plot(time, df[ubound].to_numpy(), label=f'{ubound}', linestyle=':', lw=1.5, c=orange)
    if lbound is not None:
        ax1.plot(time, df[lbound].to_numpy(), label=f'{lbound}', linestyle=':', lw=1.5, c=blue)
    ax1.grid(color='k', linestyle='dotted', linewidth=frame_lw, axis='both')
    ax1.set_title(f'{col1} vs {col2}', fontsize='xx-large')
    ax1.tick_params(labelsize=15)
    ax1.xaxis.set_ticks_position('both')
    ax1.legend(loc='upper right', fontsize=15)
    plt.show()

def plot_compare_twin_ax(col1, col2,ubound=None,lbound=None,min=None, max=None):
    rep = corr_report(col1, col2)
    best_lag = rep["lag_corr"]["pearson"]["best_lag"]
    best_corr = rep["lag_corr"]["pearson"]["best_corr"]
    diff_corr = rep["diff_corr"]["pearson"]
    fig = plt.figure(figsize=(16, 9),tight_layout=True)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.plot(time, df[col1].to_numpy(), label=f'{col1}', lw=2, c=red)
    ax2.plot(time, df[col2].to_numpy(), label=f'{col2}', linestyle='--', lw=2, c=green)
    if ubound is not None:
        ax2.plot(time, df[ubound].to_numpy(), label=f'{ubound}', linestyle=':', lw=1.5, c=orange)
    if lbound is not None:
        ax2.plot(time, df[lbound].to_numpy(), label=f'{lbound}', linestyle=':', lw=1.5, c=blue)
    ax1.set_xlabel('時刻', fontsize=15)
    ax1.set_ylabel(f'{col1}', fontsize=15)
    ax2.set_ylabel(f'{col2}', fontsize=15)
    ax1.grid(color='k', linestyle='dotted', linewidth=frame_lw, axis='both')
    ax1.set_title(f'{col1} vs {col2}\n lag={best_lag}, corr={best_corr:.2f}, diff_corr={diff_corr:.2f}', fontsize='xx-large')
    ax1.tick_params(labelsize=15)
    ax2.tick_params(labelsize=15)
    ax1.set_ylim(min, max)
    ax2.set_ylim(min, max)
    ax1.xaxis.set_ticks_position('both')
    legend2 = ax2.legend(loc='upper right', fontsize=15)
    h1, l1 = ax1.get_legend_handles_labels()
    ax2.legend(h1, l1, loc='upper left', fontsize=15)
    ax2.add_artist(legend2)
    plt.show()

def plot_linear_comb(a,x,b,y):
    fig = plt.figure(figsize=(16, 9),tight_layout=True)
    ax1 = fig.add_subplot(111)
    ax1.plot(time, a*df[x].to_numpy()+b*df[y].to_numpy(), label=f'{a}×「{x}」 + {b}×「{y}」', lw=2, c=red)
    ax1.set_xlabel('時刻', fontsize=15)
    ax1.set_ylabel(f'Value', fontsize=15)
    ax1.grid(color='k', linestyle='dotted', linewidth=frame_lw, axis='both')
    ax1.set_title(f'{a}×「{x}」 + {b}×「{y}」', fontsize='xx-large')
    ax1.tick_params(labelsize=15)
    ax1.xaxis.set_ticks_position('both')
    ax1.legend(loc='upper right', fontsize=15)
    plt.show()

def plot_diff(col1, col2):
    fig = plt.figure(figsize=(16, 9),tight_layout=True)
    ax1 = fig.add_subplot(111)
    e = df[col1].to_numpy()-df[col2].to_numpy()
    bias = np.nanmean(e)
    mae  = np.nanmean(np.abs(e))
    rmse = np.sqrt(np.nanmean(e**2))
    ax1.plot(time, e, label=f'{col1} - {col2}', lw=2, c=red)
    ax1.axhline(0, color='k', linestyle='--', lw=frame_lw)
    ax1.axhline(bias, color=orange, linestyle='--', lw=1.5, label=f'Bias={bias:.2f}')
    ax1.axhline(mae, color=blue, linestyle=':', lw=1.5, label=f'MAE={mae:.2f}')
    ax1.axhline(-mae, color=blue, linestyle=':', lw=1.5)
    ax1.axhline(rmse, color=green, linestyle=':', lw=1.5, label=f'RMSE={rmse:.2f}')
    ax1.axhline(-rmse, color=green, linestyle=':', lw=1.5)
    ax1.set_xlabel('時刻', fontsize=15)
    ax1.set_ylabel(f'差', fontsize=15)
    ax1.grid(color='k', linestyle='dotted', linewidth=frame_lw, axis='both')
    ax1.set_title(f'{col1} - {col2}', fontsize='xx-large')
    ax1.tick_params(labelsize=15)
    ax1.xaxis.set_ticks_position('both')
    ax1.legend(loc='upper right', fontsize=15)
    plt.show()

# col1と、col2*col3を比較する関数
def plot_compare_scaled(col1, col2, col3):
    rep = corr_report(col1, col2*col3)
    best_lag = rep["lag_corr"]["pearson"]["best_lag"]
    best_corr = rep["lag_corr"]["pearson"]["best_corr"]
    diff_corr = rep["diff_corr"]["pearson"]
    fig = plt.figure(figsize=(16, 9),tight_layout=True)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.plot(time, df[col1].to_numpy(), label=f'{col1}', lw=2, c=red)
    ax2.plot(time, (df[col2]*df[col3]).to_numpy(), label=f'{col2}×{col3}', linestyle='--', lw=2, c=green)
    ax1.set_xlabel('時刻', fontsize=15)
    ax1.set_ylabel(f'{col1}', fontsize=15)
    ax2.set_ylabel(f'{col2}×{col3}', fontsize=15)
    ax1.grid(color='k', linestyle='dotted', linewidth=frame_lw, axis='both')
    ax1.set_title(f'{col1} vs {col2}×{col3}\n lag={best_lag}, corr={best_corr:.2f}, diff_corr={diff_corr:.2f}', fontsize='xx-large')
    ax1.tick_params(labelsize=15)
    ax2.tick_params(labelsize=15)
    ax1.xaxis.set_ticks_position('both')
    legend2 = ax2.legend(loc='upper right', fontsize=15)
    h1, l1 = ax1.get_legend_handles_labels()
    ax2.legend(h1, l1, loc='upper left', fontsize=15)
    ax2.add_artist(legend2)
    plt.show()

# col1/col2を比較する関数
def plot_compare_divided(col1, col2, col3=None, scale=1.0):
    fig = plt.figure(figsize=(16, 9),tight_layout=True)
    ax1 = fig.add_subplot(111)
    e = df[col1].to_numpy()/df[col2].to_numpy()
    median = np.nanmedian(e)
    ax1.plot(time, e, label=f'{col1} / {col2}', lw=2, c=red)
    ax1.axhline(0, color='k', linestyle='--', lw=frame_lw)
    # ax1.axhline(bias, color=orange, linestyle='--', lw=1.5, label=f'Bias={bias:.2f}')
    # ax1.axhline(mae, color=blue, linestyle=':', lw=1.5, label=f'MAE={mae:.2f}')
    # ax1.axhline(-mae, color=blue, linestyle=':', lw=1.5)
    # ax1.axhline(rmse, color=green, linestyle=':', lw=1.5, label=f'RMSE={rmse:.2f}')
    # ax1.axhline(-rmse, color=green, linestyle=':', lw=1.5)
    ax1.axhline(median, color=orange, linestyle='--', lw=1.5, label=f'Median={median:.2f}')
    if col3 is not None:
        ax1.plot(time, df[col3].to_numpy()*scale, color=blue, linestyle=':', lw=1.5, label=f'{col3}×{scale}')
    ax1.set_xlabel('時刻', fontsize=15)
    ax1.set_ylabel(f'比', fontsize=15)
    ax1.set_ylim(median*0.5, median*1.5)
    ax1.grid(color='k', linestyle='dotted', linewidth=frame_lw, axis='both')
    ax1.set_title(f'{col1} / {col2}', fontsize='xx-large')
    ax1.tick_params(labelsize=15)
    ax1.xaxis.set_ticks_position('both')
    ax1.legend(loc='upper right', fontsize=15)
    plt.show()

def plot_air(lng,air,air_ratio,air_ratio_correction,coeff=0.01):
    fig = plt.figure(figsize=(16, 9),tight_layout=True)
    ax1 = fig.add_subplot(111)
    air_theoretical = df[lng].to_numpy()*10.7*df[air_ratio].to_numpy()*(1+coeff*df[air_ratio_correction].to_numpy())
    ax1.plot(time, df[air].to_numpy(), label=f'燃焼空気流量', lw=2, c=red)
    ax1.plot(time, air_theoretical, label=f'LNG流量×10.7×空気比×(1+空気比補正×{coeff:.2f})', lw=2, c=green, linestyle='--')
    ax1.set_xlabel('時刻', fontsize=15)
    ax1.set_ylabel(f'流量 [Nm3/h]', fontsize=15)
    ax1.grid(color='k', linestyle='dotted', linewidth=frame_lw, axis='both')
    ax1.set_title(f'LNG流量と燃焼空気流量の比較', fontsize='xx-large')
    ax1.tick_params(labelsize=15)
    ax1.xaxis.set_ticks_position('both')
    ax1.legend(loc='upper right', fontsize=15)
    plt.show()

    fig = plt.figure(figsize=(16, 9),tight_layout=True)
    ax1 = fig.add_subplot(111)
    e = df[air].to_numpy()-air_theoretical
    bias = np.nanmean(e)
    mae  = np.nanmean(np.abs(e))
    rmse = np.sqrt(np.nanmean(e**2))
    ax1.plot(time, e, label=f'空気流量の理論値との差', lw=2, c=red)
    ax1.axhline(0, color='k', linestyle='--', lw=frame_lw)
    ax1.axhline(bias, color=orange, linestyle='--', lw=1.5, label=f'Bias={bias:.2f}')
    ax1.axhline(mae, color=blue, linestyle=':', lw=1.5, label=f'MAE={mae:.2f}')
    ax1.axhline(-mae, color=blue, linestyle=':', lw=1.5)
    ax1.axhline(rmse, color=green, linestyle=':', lw=1.5, label=f'RMSE={rmse:.2f}')
    ax1.axhline(-rmse, color=green, linestyle=':', lw=1.5)
    ax1.set_xlabel('時刻', fontsize=15)
    ax1.set_ylabel(f'差', fontsize=15)
    ax1.grid(color='k', linestyle='dotted', linewidth=frame_lw, axis='both')
    ax1.set_title(f'空気流量の理論値との差', fontsize='xx-large')
    ax1.tick_params(labelsize=15)
    ax1.xaxis.set_ticks_position('both')
    ax1.legend(loc='upper right', fontsize=15)
    plt.show()


if __name__ == "__main__":
    if False:
        # 1. 加熱帯上部炉温制御ループ
        ## 1.1 まずpv,sv,mvと偏差を見てみる
        ## 1250℃程度が目標値だが、目標値は頻繁に変更されている。偏差のbias=0℃、MAE=8.0℃、RMSE=12.6℃
        plot_control_loop(
            pv="加熱帯上部炉温制御：PV（℃）_1",
            sv="加熱帯上部炉温制御：SV（℃）_1",
            mv="加熱帯上部炉温制御：MV（%）",
            title="加熱帯上部炉温制御",
            pv_unit="℃",
            mv_unit="%",
        )
        # plot_compare_twin_ax(
        #     col1="加熱帯上部炉温制御：SV（℃）_1",
        #     col2="加熱帯上部炉温制御：MV（%）",
        # )

        ## 1.2 炉温MVと1Z燃焼量指令が一致していることを見る
        ## 結果：ほぼ完全に一致している（bestlag=0, corr=1.0, 差分のmae=0.02%）。
        plot_compare(
            col1="加熱帯上部炉温制御：MV（%）",
            col2="1Z燃焼量指令",
        )
        plot_diff(
            col1="加熱帯上部炉温制御：MV（%）",
            col2="1Z燃焼量指令",
        )

        ## 1.3 炉温MVとLNG流量制御のSVが似た動きをしているか（lagも見てみる）
        ## 結果：かなり一致している（bestlag=0, corr=0.98, diff_corr=0.90）
        plot_compare_twin_ax(
            col1="加熱帯上部炉温制御：MV（%）",
            col2="加熱帯上部LNG流量制御：SV（Nm3_h）_1",
        )
        # plot_compare_divided(
        #     col1="加熱帯上部LNG流量制御：SV（Nm3_h）_1",
        #     col2="加熱帯上部炉温制御：MV（%）",
        # )

        ### 1.3.1 LNG流量制御のSVと1Z燃料設定が（ほぼ）一致していることを見る
        ### 結果：かなり一致している（bestlag=0, corr=0.97, diff_corr=0.98）。ただし、1Z燃料設定の方は上下限に収まるようになっている一方で、SVの方は下限を下回るようなものもある。SVから上下限を破らないように1Z燃料設定が作られているのかもしれないが、不明。
        plot_compare(
            col1="加熱帯上部LNG流量制御：SV（Nm3_h）_1",
            col2="1Z燃料設定",
            # ubound="1Z燃料設定上限",
            # lbound="1Z燃料設定下限",
        )
        # plot_compare_divided(
        #     col1="1Z燃料設定",
        #     col2="1Z燃焼量指令",
        # )

        ### 1.3.2 LNG流量制御のPVと1Z燃料流量が一致するか見たいが、PVのデータなし
        # plot_compare(
        #     col1="加熱帯上部LNG流量制御：PV（Nm3_h）_1",
        #     col2="1Z燃料流量",
        # )

        ### 1.3.3 制御ループを見る
        plot_control_loop(
            pv="1Z燃料流量",
            sv="1Z燃料設定",
            # mv="",
            title="1Z燃料制御",
            pv_unit="Nm3/h",
            mv_unit="%",
        )

        ## 1.4 炉温MVと燃焼空気流量制御のSVが似た動きをしているか（lagも見てみる）
        ## 結果：かなり一致している（bestlag=0, corr=0.92, diff_corr=0.90）
        plot_compare_twin_ax(
            col1="加熱帯上部炉温制御：MV（%）",
            col2="加熱帯上部燃焼空気流量制御：SV（Nm3_h）_1",
        )

        ### 1.4.1 燃焼空気流量制御のSVと1Z空気設定が（ほぼ）一致していることを見るだけ
        ### 結果：まあ一致してるか？（bestlag=0, corr=0.76, diff_corr=0.86）
        plot_compare_twin_ax(
            col1="加熱帯上部燃焼空気流量制御：SV（Nm3_h）_1",
            col2="1Z空気設定",
            ubound="1Z空気設定上限",
            lbound="1Z空気設定下限",
        )

        ### 1.4.2 燃焼空気流量制御のPVと1Z空気流量が一致するか見る
        ### 結果：1Z空気流量の方は10000超えのデータが取得できていないため、ぱっと見似てないように見えるが、多分結構近い観測値となってそう。
        plot_compare(
            col1="加熱帯上部燃焼空気流量制御：PV（Nm3_h）_1",
            col2="1Z空気流量",
        )

        ### 1.4.3 空気比を見る
        ### 結果：空気比はてっきり燃料と空気の比かと思ったが、どうもそうではないらしい。何の比かは分からない。
        # plot_compare_divided(
        #     col1="加熱帯上部燃焼空気流量制御：SV（Nm3_h）_1",
        #     col2="加熱帯上部LNG流量制御：SV（Nm3_h）_1",
        # )
        # plot_compare_divided(
        #     col1="1Z空気設定",
        #     col2="1Z燃料設定",
        #     scale = 10.7,
        #     col3="1Z空気比設定",
        # )
        plot_air(
            lng="1Z燃料設定",
            air="1Z空気設定",
            air_ratio="1Z空気比設定",
            air_ratio_correction="1Z空気比補正",
        )

        ### 1.4.4 制御ループ
        plot_control_loop(
            pv="加熱帯上部燃焼空気流量制御：PV（Nm3_h）_1",
            sv="加熱帯上部燃焼空気流量制御：SV（Nm3_h）_1",
            mv="加熱帯上部燃焼空気流量制御：MV（%）",
            title="加熱帯上部燃焼空気流量制御",
            pv_unit="Nm3/h",
            mv_unit="%",
            mv_min=-20,
            mv_max=120,
        )

        ## 1.5 炉温MVと排ガス流量制御のSVが似た動きをしているか（lagも見てみる）
        ## 結果：かなり一致している（bestlag=0, corr=0.93, diff_corr=0.86）
        plot_compare_twin_ax(
            col1="加熱帯上部炉温制御：MV（%）",
            col2="加熱帯上部排ｶﾞｽ流量制御：SV（Nm3_h）_1",
        )

        ### 1.5.X 制御ループ
        plot_control_loop(
            pv="加熱帯上部排ｶﾞｽ流量制御：PV（Nm3_h）_1",
            sv="加熱帯上部排ｶﾞｽ流量制御：SV（Nm3_h）_1",
            mv="加熱帯上部排ｶﾞｽ流量制御：MV（%）",
            title="加熱帯上部排ガス流量制御",
            pv_unit="Nm3/h",
            mv_unit="%",
            mv_min=-20,
            mv_max=120,
        )

    if False:
        # 1. 加熱帯下部炉温制御ループ
        ## 1.1 まずpv,sv,mvと偏差を見てみる
        ## 1250℃程度が目標値だが、目標値は頻繁に変更されている。偏差のbias=0℃、MAE=8.0℃、RMSE=12.6℃
        plot_control_loop(
            pv="加熱帯下部炉温制御：PV（℃）_1",
            sv="加熱帯下部炉温制御：SV（℃）_1",
            mv="加熱帯下部炉温制御：MV（%）",
            title="加熱帯下部炉温制御",
            pv_unit="℃",
            mv_unit="%",
            mv_max=120,
            mv_min=-20,
        )
        # plot_compare_twin_ax(
        #     col1="加熱帯下部炉温制御：SV（℃）_1",
        #     col2="加熱帯下部炉温制御：MV（%）",
        # )

        ## 1.2 炉温MVと1Z燃焼量指令が一致していることを見る
        ## 結果：ほぼ完全に一致している（bestlag=0, corr=1.0, 差分のmae=0.02%）。
        plot_compare(
            col1="加熱帯下部炉温制御：MV（%）",
            col2="2Z燃焼量指令",
        )
        plot_diff(
            col1="加熱帯下部炉温制御：MV（%）",
            col2="2Z燃焼量指令",
        )

        ## 1.3 炉温MVとLNG流量制御のSVが似た動きをしているか（lagも見てみる）
        ## 結果：かなり一致している（bestlag=0, corr=0.98, diff_corr=0.90）
        plot_compare_twin_ax(
            col1="加熱帯下部炉温制御：MV（%）",
            col2="加熱帯下部LNG流量制御：SV（Nm3_h）_1",
        )
        # plot_compare_divided(
        #     col1="加熱帯下部LNG流量制御：SV（Nm3_h）_1",
        #     col2="加熱帯下部炉温制御：MV（%）",
        # )

        ### 1.3.1 LNG流量制御のSVと2Z燃料設定が（ほぼ）一致していることを見る
        ### 結果：かなり一致している（bestlag=0, corr=0.97, diff_corr=0.98）。ただし、2Z燃料設定の方は上下限に収まるようになっている一方で、SVの方は下限を下回るようなものもある。SVから上下限を破らないように2Z燃料設定が作られているのかもしれないが、不明。
        plot_compare(
            col1="加熱帯下部LNG流量制御：SV（Nm3_h）_1",
            col2="2Z燃料設定",
            # ubound="2Z燃料設定上限",
            # lbound="2Z燃料設定下限",
        )
        # plot_compare_divided(
        #     col1="2Z燃料設定",
        #     col2="2Z燃焼量指令",
        # )

        ### 1.3.2 LNG流量制御のPVと2Z燃料流量が一致するか見たいが、PVのデータなし
        # plot_compare(
        #     col1="加熱帯下部LNG流量制御：PV（Nm3_h）_1",
        #     col2="2Z燃料流量",
        # )

        ### 1.3.3 制御ループを見る
        plot_control_loop(
            pv="加熱帯下部LNG流量制御：PV（Nm3_h）_1",
            sv="加熱帯下部LNG流量制御：SV（Nm3_h）_1",
            mv="加熱帯下部LNG流量制御：MV（%）",
            title="加熱帯下部LNG流量制御",
            pv_unit="Nm3/h",
            mv_unit="%",
            mv_min=-20,
            mv_max=120,
        )

        plot_control_loop(
            pv="2Z燃料流量",
            sv="2Z燃料設定",
            # mv="",
            title="2Z燃料制御",
            pv_unit="Nm3/h",
            mv_unit="%",
        )

        ## 1.4 炉温MVと燃焼空気流量制御のSVが似た動きをしているか（lagも見てみる）
        ## 結果：かなり一致している（bestlag=0, corr=0.92, diff_corr=0.90）
        plot_compare_twin_ax(
            col1="加熱帯下部炉温制御：MV（%）",
            col2="加熱帯下部燃焼空気流量制御：SV（Nm3_h）_1",
        )

        ### 1.4.1 燃焼空気流量制御のSVと2Z空気設定が（ほぼ）一致していることを見るだけ
        ### 結果：まあ一致してるか？（bestlag=0, corr=0.76, diff_corr=0.86）
        plot_compare_twin_ax(
            col1="加熱帯下部燃焼空気流量制御：SV（Nm3_h）_1",
            col2="2Z空気設定",
            # ubound="2Z空気設定上限",
            # lbound="2Z空気設定下限",
        )

        ### 1.4.2 燃焼空気流量制御のPVと2Z空気流量が一致するか見る
        ### 結果：2Z空気流量の方は10000超えのデータが取得できていないため、ぱっと見似てないように見えるが、多分結構近い観測値となってそう。
        plot_compare(
            col1="加熱帯下部燃焼空気流量制御：PV（Nm3_h）_1",
            col2="2Z空気流量",
        )

        ### 1.4.3 空気比を見る
        ### 結果：空気比はてっきり燃料と空気の比かと思ったが、どうもそうではないらしい。何の比かは分からない。
        # plot_compare_divided(
        #     col1="加熱帯上部燃焼空気流量制御：SV（Nm3_h）_1",
        #     col2="加熱帯上部LNG流量制御：SV（Nm3_h）_1",
        # )
        # plot_compare_divided(
        #     col1="2Z空気設定",
        #     col2="1Z燃料設定",
        #     scale = 10.7,
        #     col3="1Z空気比設定",
        # )
        plot_air(
            lng="2Z燃料設定",
            air="2Z空気設定",
            air_ratio="2Z空気比設定",
            air_ratio_correction="2Z空気比補正",
        )

        ### 1.4.4 制御ループ
        plot_control_loop(
            pv="加熱帯下部燃焼空気流量制御：PV（Nm3_h）_1",
            sv="加熱帯下部燃焼空気流量制御：SV（Nm3_h）_1",
            mv="加熱帯下部燃焼空気流量制御：MV（%）",
            title="加熱帯下部燃焼空気流量制御",
            pv_unit="Nm3/h",
            mv_unit="%",
            mv_min=-20,
            mv_max=120,
        )

        ## 1.5 炉温MVと排ガス流量制御のSVが似た動きをしているか（lagも見てみる）
        ## 結果：かなり一致している（bestlag=0, corr=0.93, diff_corr=0.86）
        plot_compare_twin_ax(
            col1="加熱帯下部炉温制御：MV（%）",
            col2="加熱帯下部排ｶﾞｽ流量制御：SV（Nm3_h）_1",
        )

        ### 1.5.X 制御ループ
        plot_control_loop(
            pv="加熱帯下部排ｶﾞｽ流量制御：PV（Nm3_h）_1",
            sv="加熱帯下部排ｶﾞｽ流量制御：SV（Nm3_h）_1",
            mv="加熱帯下部排ｶﾞｽ流量制御：MV（%）",
            title="加熱帯下部排ガス流量制御",
            pv_unit="Nm3/h",
            mv_unit="%",
            mv_min=-20,
            mv_max=120,
        )

    if False:
        # 1. 均熱帯上部炉温制御ループ
        ## 1.1 まずpv,sv,mvと偏差を見てみる
        ## 1250℃程度が目標値だが、目標値は頻繁に変更されている。偏差のbias=0℃、MAE=8.0℃、RMSE=12.6℃
        plot_control_loop(
            pv="均熱帯上部炉温制御：PV（℃）_1",
            sv="均熱帯上部炉温制御：SV（℃）_1",
            mv="均熱帯上部炉温制御：MV（%）",
            title="均熱帯上部炉温制御",
            pv_unit="℃",
            mv_unit="%",
            mv_max=120,
            mv_min=-20,
        )
        # plot_compare_twin_ax(
        #     col1="均熱帯上部炉温制御：SV（℃）_1",
        #     col2="均熱帯上部炉温制御：MV（%）",
        # )

        ## 1.2 炉温MVと1Z燃焼量指令が一致していることを見る
        ## 結果：ほぼ完全に一致している（bestlag=0, corr=1.0, 差分のmae=0.02%）。
        plot_compare(
            col1="均熱帯上部炉温制御：MV（%）",
            col2="3Z燃焼量指令",
        )
        plot_diff(
            col1="均熱帯上部炉温制御：MV（%）",
            col2="3Z燃焼量指令",
        )

        ## 1.3 炉温MVとLNG流量制御のSVが似た動きをしているか（lagも見てみる）
        ## 結果：かなり一致している（bestlag=0, corr=0.98, diff_corr=0.90）
        plot_compare_twin_ax(
            col1="均熱帯上部炉温制御：MV（%）",
            col2="均熱帯上部LNG流量制御：SV（Nm3_h）_1",
        )
        # plot_compare_divided(
        #     col1="均熱帯上部LNG流量制御：SV（Nm3_h）_1",
        #     col2="均熱帯上部炉温制御：MV（%）",
        # )

        ### 1.3.1 LNG流量制御のSVと2Z燃料設定が（ほぼ）一致していることを見る
        ### 結果：かなり一致している（bestlag=0, corr=0.97, diff_corr=0.98）。ただし、2Z燃料設定の方は上下限に収まるようになっている一方で、SVの方は下限を下回るようなものもある。SVから上下限を破らないように2Z燃料設定が作られているのかもしれないが、不明。
        # plot_compare(
        #     col1="均熱帯上部LNG流量制御：SV（Nm3_h）_1",
        #     col2="3Z燃料設定",
        #     # ubound="3Z燃料設定上限",
        #     # lbound="3Z燃料設定下限",
        # )
        # plot_compare_divided(
        #     col1="3Z燃料設定",
        #     col2="3Z燃焼量指令",
        # )

        ### 1.3.2 LNG流量制御のPVと3Z燃料流量が一致するか見たいが、PVのデータなし
        # plot_compare(
        #     col1="均熱帯上部LNG流量制御：PV（Nm3_h）_1",
        #     col2="3Z燃料流量",
        # )

        ### 1.3.3 制御ループを見る
        plot_control_loop(
            pv="均熱帯上部LNG流量制御：PV（Nm3_h）_1",
            sv="均熱帯上部LNG流量制御：SV（Nm3_h）_1",
            mv="均熱帯上部LNG流量制御：MV（%）",
            title="均熱帯上部LNG流量制御",
            pv_unit="Nm3/h",
            mv_unit="%",
            mv_min=-20,
            mv_max=200,
        )

        # plot_control_loop(
        #     pv="3Z燃料流量",
        #     sv="3Z燃料設定",
        #     # mv="",
        #     title="3Z燃料制御",
        #     pv_unit="Nm3/h",
        #     mv_unit="%",
        # )

        ## 1.4 炉温MVと燃焼空気流量制御のSVが似た動きをしているか（lagも見てみる）
        ## 結果：かなり一致している（bestlag=0, corr=0.92, diff_corr=0.90）
        plot_compare_twin_ax(
            col1="均熱帯上部炉温制御：MV（%）",
            col2="均熱帯上部燃焼空気流量制御：SV（Nm3_h）_1",
        )

        ### 1.4.1 燃焼空気流量制御のSVと3Z空気設定が（ほぼ）一致していることを見るだけ
        ### 結果：まあ一致してるか？（bestlag=0, corr=0.76, diff_corr=0.86）
        # plot_compare_twin_ax(
        #     col1="均熱帯上部燃焼空気流量制御：SV（Nm3_h）_1",
        #     col2="3Z空気設定",
        #     # ubound="3Z空気設定上限",
        #     # lbound="3Z空気設定下限",
        # )

        ### 1.4.2 燃焼空気流量制御のPVと3Z空気流量が一致するか見る
        ### 結果：3Z空気流量の方は10000超えのデータが取得できていないため、ぱっと見似てないように見えるが、多分結構近い観測値となってそう。
        plot_compare(
            col1="均熱帯上部燃焼空気流量制御：PV（Nm3_h）_1",
            col2="3Z空気流量",
        )

        ### 1.4.3 空気比を見る
        ### 結果：空気比はてっきり燃料と空気の比かと思ったが、どうもそうではないらしい。何の比かは分からない。
        # plot_compare_divided(
        #     col1="加熱帯上部燃焼空気流量制御：SV（Nm3_h）_1",
        #     col2="加熱帯上部LNG流量制御：SV（Nm3_h）_1",
        # )
        # plot_compare_divided(
        #     col1="3Z空気設定",
        #     col2="1Z燃料設定",
        #     scale = 10.7,
        #     col3="1Z空気比設定",
        # )
        # plot_air(
        #     lng="3Z燃料設定",
        #     air="3Z空気設定",
        #     air_ratio="3Z空気比設定",
        #     air_ratio_correction="3Z空気比補正",
        # )

        ### 1.4.4 制御ループ
        plot_control_loop(
            pv="均熱帯上部燃焼空気流量制御：PV（Nm3_h）_1",
            sv="均熱帯上部燃焼空気流量制御：SV（Nm3_h）_1",
            mv="均熱帯上部燃焼空気流量制御：MV（%）",
            title="均熱帯上部燃焼空気流量制御",
            pv_unit="Nm3/h",
            mv_unit="%",
            mv_min=-20,
            mv_max=120,
        )

        ## 1.5 炉温MVと排ガス流量制御のSVが似た動きをしているか（lagも見てみる）
        ## 結果：かなり一致している（bestlag=0, corr=0.93, diff_corr=0.86）
        plot_compare_twin_ax(
            col1="均熱帯上部炉温制御：MV（%）",
            col2="均熱帯上部排ｶﾞｽ流量制御：SV（Nm3_h）_1",
        )

        ### 1.5.X 制御ループ
        plot_control_loop(
            pv="均熱帯上部排ｶﾞｽ流量制御：PV（Nm3_h）_1",
            sv="均熱帯上部排ｶﾞｽ流量制御：SV（Nm3_h）_1",
            mv="均熱帯上部排ｶﾞｽ流量制御：MV（%）",
            title="均熱帯上部排ガス流量制御",
            pv_unit="Nm3/h",
            mv_unit="%",
            mv_min=-20,
            mv_max=120,
        )

    if True:
        # 1. 均熱帯下部炉温制御ループ
        ## 1.1 まずpv,sv,mvと偏差を見てみる
        ## 1250℃程度が目標値だが、目標値は頻繁に変更されている。偏差のbias=0℃、MAE=8.0℃、RMSE=12.6℃
        plot_control_loop(
            pv="均熱帯下部炉温制御：PV（℃）_1",
            sv="均熱帯下部炉温制御：SV（℃）_1",
            mv="均熱帯下部炉温制御：MV（%）",
            title="均熱帯下部炉温制御",
            pv_unit="℃",
            mv_unit="%",
            mv_max=120,
            mv_min=-20,
        )
        # plot_compare_twin_ax(
        #     col1="均熱帯下部炉温制御：SV（℃）_1",
        #     col2="均熱帯下部炉温制御：MV（%）",
        # )

        ## 1.2 炉温MVと4Z燃焼量指令が一致していることを見る
        ## 結果：ほぼ完全に一致している（bestlag=0, corr=1.0, 差分のmae=0.02%）。
        plot_compare(
            col1="均熱帯下部炉温制御：MV（%）",
            col2="4Z燃焼量指令",
        )
        plot_diff(
            col1="均熱帯下部炉温制御：MV（%）",
            col2="4Z燃焼量指令",
        )

        ## 1.3 炉温MVとLNG流量制御のSVが似た動きをしているか（lagも見てみる）
        ## 結果：かなり一致している（bestlag=0, corr=0.98, diff_corr=0.90）
        # plot_compare_twin_ax(
        #     col1="均熱帯下部炉温制御：MV（%）",
        #     col2="均熱帯下部LNG流量制御：SV（Nm3_h）_1",
        # )
        # plot_compare_divided(
        #     col1="均熱帯下部LNG流量制御：SV（Nm3_h）_1",
        #     col2="均熱帯下部炉温制御：MV（%）",
        # )

        ### 1.3.1 LNG流量制御のSVと2Z燃料設定が（ほぼ）一致していることを見る
        ### 結果：かなり一致している（bestlag=0, corr=0.97, diff_corr=0.98）。ただし、2Z燃料設定の方は上下限に収まるようになっている一方で、SVの方は下限を下回るようなものもある。SVから上下限を破らないように2Z燃料設定が作られているのかもしれないが、不明。
        # plot_compare(
        #     col1="均熱帯下部LNG流量制御：SV（Nm3_h）_1",
        #     col2="4Z燃料設定",
        #     # ubound="4Z燃料設定上限",
        #     # lbound="4Z燃料設定下限",
        # )
        # plot_compare_divided(
        #     col1="4Z燃料設定",
        #     col2="4Z燃焼量指令",
        # )

        ### 1.3.2 LNG流量制御のPVと3Z燃料流量が一致するか見たいが、PVのデータなし
        # plot_compare(
        #     col1="均熱帯下部LNG流量制御：PV（Nm3_h）_1",
        #     col2="3Z燃料流量",
        # )

        ### 1.3.3 制御ループを見る
        # plot_control_loop(
        #     pv="均熱帯下部LNG流量制御：PV（Nm3_h）_1",
        #     sv="均熱帯下部LNG流量制御：SV（Nm3_h）_1",
        #     mv="均熱帯下部LNG流量制御：MV（%）",
        #     title="均熱帯下部LNG流量制御",
        #     pv_unit="Nm3/h",
        #     mv_unit="%",
        #     mv_min=-20,
        #     mv_max=200,
        # )

        plot_control_loop(
            pv="4Z燃料流量",
            sv="4Z燃料設定",
            # mv="",
            title="4Z燃料制御",
            pv_unit="Nm3/h",
            mv_unit="%",
        )

        ## 1.4 炉温MVと燃焼空気流量制御のSVが似た動きをしているか（lagも見てみる）
        ## 結果：かなり一致している（bestlag=0, corr=0.92, diff_corr=0.90）
        plot_compare_twin_ax(
            col1="均熱帯下部炉温制御：MV（%）",
            col2="均熱帯下部燃焼空気流量制御：SV（Nm3_h）_1",
        )

        ### 1.4.1 燃焼空気流量制御のSVと3Z空気設定が（ほぼ）一致していることを見るだけ
        ### 結果：まあ一致してるか？（bestlag=0, corr=0.76, diff_corr=0.86）
        # plot_compare_twin_ax(
        #     col1="均熱帯下部燃焼空気流量制御：SV（Nm3_h）_1",
        #     col2="3Z空気設定",
        #     # ubound="3Z空気設定上限",
        #     # lbound="3Z空気設定下限",
        # )

        ### 1.4.2 燃焼空気流量制御のPVと3Z空気流量が一致するか見る
        ### 結果：3Z空気流量の方は10000超えのデータが取得できていないため、ぱっと見似てないように見えるが、多分結構近い観測値となってそう。
        plot_compare(
            col1="均熱帯下部燃焼空気流量制御：PV（Nm3_h）_1",
            col2="4Z空気流量",
        )

        ### 1.4.3 空気比を見る
        ### 結果：空気比はてっきり燃料と空気の比かと思ったが、どうもそうではないらしい。何の比かは分からない。
        # plot_compare_divided(
        #     col1="加熱帯下部燃焼空気流量制御：SV（Nm3_h）_1",
        #     col2="加熱帯下部LNG流量制御：SV（Nm3_h）_1",
        # )
        # plot_compare_divided(
        #     col1="4Z空気設定",
        #     col2="4Z燃料設定",
        #     scale = 10.7,
        #     col3="4Z空気比設定",
        # )
        # plot_air(
        #     lng="4Z燃料設定",
        #     air="4Z空気設定",
        #     air_ratio="4Z空気比設定",
        #     air_ratio_correction="4Z空気比補正",
        # )

        ### 1.4.4 制御ループ
        plot_control_loop(
            pv="均熱帯下部燃焼空気流量制御：PV（Nm3_h）_1",
            sv="均熱帯下部燃焼空気流量制御：SV（Nm3_h）_1",
            mv="均熱帯下部燃焼空気流量制御：MV（%）",
            title="均熱帯下部燃焼空気流量制御",
            pv_unit="Nm3/h",
            mv_unit="%",
            mv_min=-20,
            mv_max=120,
        )

        ## 1.5 炉温MVと排ガス流量制御のSVが似た動きをしているか（lagも見てみる）
        ## 結果：かなり一致している（bestlag=0, corr=0.93, diff_corr=0.86）
        plot_compare_twin_ax(
            col1="均熱帯下部炉温制御：MV（%）",
            col2="均熱帯下部排ｶﾞｽ流量制御：SV（Nm3_h）_1",
        )

        ### 1.5.X 制御ループ
        plot_control_loop(
            pv="均熱帯下部排ｶﾞｽ流量制御：PV（Nm3_h）_1",
            sv="均熱帯下部排ｶﾞｽ流量制御：SV（Nm3_h）_1",
            mv="均熱帯下部排ｶﾞｽ流量制御：MV（%）",
            title="均熱帯下部排ガス流量制御",
            pv_unit="Nm3/h",
            mv_unit="%",
            mv_min=-20,
            mv_max=120,
        )
