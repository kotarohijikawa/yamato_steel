import glob
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import os
import japanize_matplotlib

################################################################################
# 解析結果めも
# データディレクトリにはCSVファイルだけが存在。
# CSVファイルは全部で638個ある。
# すべてのCSVファイルのヘッダは「Time」と「Value」である。
# ロギング周期は全て1秒

################################################################################

# 準備1：CSVファイルのパスを取得する。またファイル一覧を外部ファイルに出力する。
csv_paths = glob.glob("./加熱炉データ/データ/*.csv")
csv_paths = sorted(csv_paths)
# with open("./csv_file_list.txt", "w", encoding="utf-8") as f:
#     for path in csv_paths:
#         f.write(f"{path}\n")

# 調査1：「データ」ディレクトリにCSV以外が存在するかどうか調べる。
def check_non_csv_files():
    non_csv_files = glob.glob("./加熱炉データ/データ/*")
    non_csv_files = [f for f in non_csv_files if not f.endswith(".csv")]
    if non_csv_files:
        print("CSV以外のファイルが存在します:")
        for f in non_csv_files:
            print(f)
    else:
        print("CSV以外のファイルは存在しません。")

# 調査2：各CSVのヘッダ確認
# おそらく全て二列で、一列目のヘッダ名が「Time」、二列目が「Value」。これに該当しないものだけはじき出す。
def check_csv_headers():
    for i, path in enumerate(csv_paths):
        print(f"{i+1}/{len(csv_paths)}: {path}")
        df = pd.read_csv(path)
        if list(df.columns) != ["Time", "Value"]:
            print(f"ヘッダが異なるCSVファイル: {path}")

# 調査3：各CSVの情報を出力
# ファイル名、行数、開始時刻、終了時刻、時間刻み幅の最頻値をCSVに出力する。
def check_csv_row_counts():
    output_data = []
    for i, path in enumerate(csv_paths):
        print(f"{i+1}/{len(csv_paths)}: {path}")
        df = pd.read_csv(path)
        row_count = len(df)
        start_time = df["Time"].iloc[0] if row_count > 0 else None
        end_time = df["Time"].iloc[-1] if row_count > 0 else None
        # 時間型に変換して差分を取る
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        time_diffs = df["Time"].diff().dropna()
        most_common_time_diff = time_diffs.mode().iloc[0] if not time_diffs.empty else None

        u = df["Value"].dropna().unique()
        all_same = (len(u) == 1)
        if all_same:
            note = f"値は全て {u[0]}"
        else:
            note = ""

        output_data.append({
            "file_name": os.path.basename(path),
            "row_count": row_count,
            "start_time": start_time,
            "end_time": end_time,
            "most_common_time_diff": most_common_time_diff,
            "note": note,
        })
    output_df = pd.DataFrame(output_data)
    output_df.to_csv("./_csv_file_info.csv", index=False, encoding="utf-8-sig")

# 調査4:先ほど作った「csv_file_info.csv」を読み込んで、開始時刻-終了時刻の組み合わせでユニークなものを列挙して、各パターンに当てはまるCSVがどれくらいあったか表示する。
def analyze_csv_file_info():
    df = pd.read_csv("./csv_file_info.csv")
    unique_combinations = df[["start_time", "end_time"]].drop_duplicates()
    for _, row in unique_combinations.iterrows():
        count = ((df["start_time"] == row["start_time"]) & (df["end_time"] == row["end_time"])).sum()
        print(f"開始時刻 {row['start_time']} から終了時刻 {row['end_time']} までのCSVファイルの数: {count}")
    print()

    # unique_start_times = df["start_time"].unique()
    # for start_time in unique_start_times:
    #     count = (df["start_time"] == start_time).sum()
    #     print(f"開始時刻 {start_time} から始まるCSVファイルの数: {count}")
    # print()

    # unique_end_times = df["end_time"].unique()
    # for end_time in unique_end_times:
    #     count = (df["end_time"] == end_time).sum()
    #     print(f"終了時刻 {end_time} で終わるCSVファイルの数: {count}")
    # print()

    unique_time_diffs = df["most_common_time_diff"].unique()
    for time_diff in unique_time_diffs:
        count = (df["most_common_time_diff"] == time_diff).sum()
        print(f"時間刻み幅の最頻値 {time_diff} のCSVファイルの数: {count}")

# 調査

def plot():
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
    # check_non_csv_files()
    # check_csv_headers()
    # check_csv_row_counts()
    # analyze_csv_file_info()
    plot()

    print()