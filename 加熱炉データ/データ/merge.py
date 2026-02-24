import pandas as pd
from functools import reduce
from pathlib import Path

csv_dir = Path("./")
files = sorted(csv_dir.glob("*.csv"))

dfs = []
for i, f in enumerate(files, start=1):
    print(f"Processing {i}/{len(files)}: {f}")
    df = pd.read_csv(f)
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.rename(columns={"Value": f.stem})
    dfs.append(df)

merged = reduce(lambda l, r: pd.merge(l, r, on="Time", how="outer"), dfs)

# --- ここからダウンサンプリング ---
print("ダウンサンプリング中...")
merged = (
    merged
    .set_index("Time")
    .sort_index()
    .resample("10min")      # "5min", "10s", "1H" とかもOK
    .mean(numeric_only=True)  # 列ごとに平均。欠損は勝手に無視してくれる
    .reset_index()
)

print("ダウンサンプリング完了。出力中...")
merged.to_csv("merged_downsampled.csv", index=False)
