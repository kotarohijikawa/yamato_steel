import toolbox
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "./加熱炉データ/データ/"

def analyze_loop(obj, ctrl, unit, has_underscore=False):
    if has_underscore:
        str0 = '_1'
    else:
        str0 = ''
    loop = toolbox.LoopFiles(
        pv=f"{obj}{ctrl}：PV（{unit}）{str0}.csv",
        sv=f"{obj}{ctrl}：SV（{unit}）{str0}.csv",
        mv=f"{obj}{ctrl}：MV（%）.csv",
    )
    toolbox.analyze_loop(loop, base_dir=BASE_DIR, downsample_n=600, max_lag_steps=200, title=f"【{obj}】{ctrl}")

def compare_two_files(file_x, file_y):
    toolbox.compare_two_files(
    file_x=file_x,
    file_y=file_y,
    base_dir = BASE_DIR,
    downsample_n=600,
    max_lag_steps=300,
    title=f"{os.path.basename(file_x)} vs {os.path.basename(file_y)}",
    )

def simple_plot():
    args = sys.argv
    for arg in args:
        if arg.strip()[-4:]==".csv":
            df = pd.read_csv(arg)
            # print(df.describe().T)

            # 10分=600秒おきにダウンサンプリング
            df = df.iloc[::600].reset_index(drop=True)

            plt.plot((pd.to_datetime(df['Time'])).to_numpy(), df['Value'].to_numpy(),label=f'{os.path.basename(arg).split(".")[0]}')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # analyze_loop("1側", "抽出温度", "%")
    # analyze_loop("加熱上", "炉温度", "%")
    # analyze_loop("加熱上", "燃焼量", "%")

    # analyze_loop("加熱帯上部", "炉温制御", "℃", True)

    # compare_two_files("加熱帯上部炉温制御：SV（℃）_1.csv", "加熱帯ﾊﾟﾀｰﾝ温度.csv")
    compare_two_files("加熱帯上部炉温制御：SV（℃）_1.csv", "加熱帯上部  PC温度設定.csv")
    # simple_plot()

