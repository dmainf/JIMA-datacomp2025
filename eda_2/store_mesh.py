import os
import pandas as pd
import chardet
import datetime as dt

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
STORE_FILE = os.path.join(BASE_DIR, "data", "Store.csv")
STORE_CODE_FILE = os.path.join(BASE_DIR, "data", "store_code.txt")
DATA_STORE_DIR = os.path.join(BASE_DIR, "data_store")

def get_population_from_mesh(mesh_code: str):
    try:
        mesh_code = str(mesh_code).strip().replace("　", "")
        if not mesh_code.isdigit() or len(mesh_code) < 4:
            return 0

        prefix = mesh_code[:4]
        # ③ 読み込むファイルを tblT001140S に変更
        file_name = f"tblT001140S{prefix}.csv"
        file_path = os.path.join(DATA_STORE_DIR, file_name)

        if not os.path.exists(file_path):
            return 0

        with open(file_path, "rb") as f:
            raw = f.read(10000)
            enc = chardet.detect(raw)["encoding"] or "utf-8"

        try:
            df_sub = pd.read_csv(file_path, encoding=enc, header=None)
        except UnicodeDecodeError:
            df_sub = pd.read_csv(file_path, encoding="cp932", header=None)

        df_sub[0] = df_sub[0].astype(str).str.strip().str.replace("　", "")
        df_match = df_sub[df_sub[0] == mesh_code]

        if not df_match.empty:
            val = df_match.iloc[0, 4]
            if isinstance(val, str):
                val = val.replace(",", "").strip()
            try:
                return float(val)
            except ValueError:
                return 0
        else:
            return 0

    except Exception as e:
        print(f"Error : mesh_code = {mesh_code} {e}")
        return 0

# --- Storeファイル読み込み ---
try:
    df_store = pd.read_csv(STORE_FILE, encoding="utf-8-sig")
except UnicodeDecodeError:
    df_store = pd.read_csv(STORE_FILE, encoding="cp932")

# --- store_code.txtの補完 ---
try:
    df_codes = pd.read_csv(
        STORE_CODE_FILE,
        sep=r"[\s　]+",
        engine="python",
        encoding="utf-8-sig"
    )
    df_codes.columns = df_codes.columns.str.strip().str.replace("　", "")

    if len(df_store) == len(df_codes):
        for col in ["書店コード", "書店名"]:
            if col in df_store.columns and col in df_codes.columns:
                df_store[col] = df_store[col].fillna(df_codes[col])
                df_store.loc[df_store[col] == "", col] = df_codes[col]
        print("input store_code.txt")
    else:
        print(f"Error : Store.csv={len(df_store)}, store_code.txt={len(df_codes)}")
except Exception as e:
    print(f"Not Found : store_code.txt {e}")

# --- メッシュ人口計算 ---
print("input store_mesh...")
df_store["3次メッシュ人口"] = df_store["店舗メッシュ"].apply(get_population_from_mesh)  # ①

print("input around_mesh...")
周辺カラム = [f"周辺{i}" for i in range(1, 9)]
for col in 周辺カラム:
    new_col = f"{col}_3次人口"  # ②
    df_store[new_col] = df_store[col].apply(get_population_from_mesh)

for col in [f"周辺{i}_3次人口" for i in range(1, 9)]:
    df_store[col] = pd.to_numeric(df_store[col], errors="coerce").fillna(0)

df_store["周辺合計人口"] = df_store[[f"周辺{i}_3次人口" for i in range(1, 9)]].astype(float).sum(axis=1)

# --- 営業時間をdatetime型で計算 ---
time_cols = ["開店時間(平)", "閉店時間(平)", "開店時間(特)", "閉店時間(特)"]

def convert_time(value):
    try:
        value = int(value)
        hour = value // 100
        minute = value % 100
        if hour == 24:
            hour = 0
        return dt.time(hour, minute)
    except Exception:
        return None

for col in time_cols:
    if col in df_store.columns:
        df_store[col] = df_store[col].apply(convert_time)

for mode in ["(平)", "(特)"]:
    open_col = f"開店時間{mode}"
    close_col = f"閉店時間{mode}"
    work_col = f"営業時間{mode}"

    if open_col in df_store.columns and close_col in df_store.columns:
        def calc_duration(row):
            open_t = row[open_col]
            close_t = row[close_col]
            if pd.notnull(open_t) and pd.notnull(close_t):
                start = dt.datetime.combine(dt.date.today(), open_t)
                end = dt.datetime.combine(dt.date.today(), close_t)

                # 24時間営業対応
                if open_t == close_t:
                    return dt.time(23, 59, 59)
                # 翌日閉店対応
                if end <= start:
                    end += dt.timedelta(days=1)

                delta = (end - start).seconds
                hours = delta // 3600
                minutes = (delta % 3600) // 60
                return dt.time(hours % 24, minutes)
            return None

        df_store[work_col] = df_store.apply(calc_duration, axis=1)

print("change datetime")

# --- 昼夜比計算 ---
if all(col in df_store.columns for col in ["市区別_夜間人口", "市区別_昼間人口"]):
    df_store["昼間/夜間割合"] = df_store.apply(
        lambda x: round((x["市区別_昼間人口"] / x["市区別_夜間人口"]) * 100, 1)
        if x["市区別_夜間人口"] not in [0, None, ""] else 0,
        axis=1
    ).astype(float)  # ④ 小数点第1位を保持
    print("calculation day/night")
else:
    print("Not Found : daytime or nighttime")

# --- カラム型チェック＆変換 ---
print("check column...")

expected_dtypes = {
    "書店コード": "int",
    "書店名": "object",
    "住所": "object",
    "駅距離": "int",
    "開店時間(平)": "datetime",
    "閉店時間(平)": "datetime",
    "営業時間(平)": "datetime",
    "開店時間(特)": "datetime",
    "閉店時間(特)": "datetime",
    "営業時間(特)": "datetime",
    "駅構内": "int",
    "複合施設": "int",
    "独立店舗": "int",
    "人口": "int",
    "周辺人口": "int",
    "合算人口": "int",
    "店舗メッシュ": "int",
    "周辺1": "int",
    "周辺2": "int",
    "周辺3": "int",
    "周辺4": "int",
    "周辺5": "int",
    "周辺6": "int",
    "周辺7": "int",
    "周辺8": "int",
    "市区別_夜間人口": "int",
    "市区別_昼間人口": "int",
    "昼間フラグ": "int",
    "昼間/夜間割合": "float",   # ④ float化
    "3次メッシュ人口": "int",   # ① 変更済み
    "周辺1_3次人口": "int",
    "周辺2_3次人口": "int",
    "周辺3_3次人口": "int",
    "周辺4_3次人口": "int",
    "周辺5_3次人口": "int",
    "周辺6_3次人口": "int",
    "周辺7_3次人口": "int",
    "周辺8_3次人口": "int",
    "周辺合計人口": "int"
}

def fix_dtype(series, target_type):
    if target_type == "int":
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    elif target_type == "float":
        return pd.to_numeric(series, errors="coerce").astype(float)
    elif target_type == "datetime":
        if all(isinstance(v, dt.time) or pd.isnull(v) for v in series):
            return series
        return pd.to_datetime(series, errors="coerce").dt.time
    elif target_type == "object":
        return series.astype(str)
    return series

for col, t in expected_dtypes.items():
    if col in df_store.columns:
        before = df_store[col].dtype
        df_store[col] = fix_dtype(df_store[col], t)
        after = df_store[col].dtype
        if str(before) != str(after):
            print(f" {col}: {before} → {after}")
    else:
        print(f"Not Found : column {col}")

print("check complete")

# --- 出力（CSV + Parquet） ---
output_csv = os.path.join(BASE_DIR, "data", "Store_mesh.csv")
output_parquet = os.path.join(BASE_DIR, "data", "Store_mesh.parquet")

df_store.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"CSV: {output_csv}")

df_store.to_parquet(output_parquet, index=False)
print(f"Parquet: {output_parquet}")

print("Complete!")
