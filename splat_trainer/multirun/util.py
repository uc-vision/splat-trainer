import argparse
import json
from pathlib import Path
import os
from typing import Union

from filelock import FileLock
import pandas as pd
import redis


def load_data(file: Union[Path, str]) -> dict:
    data = []
    if os.path.exists(file):
        try:
            with open(file, "r") as f:
                data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while reading file '{file}': {e}")
        
    return data


def dump_data(file: Union[Path, str], data: dict) -> None:
    try:
        with open(file, "w") as f:
            json.dump(data, f, indent=4)

    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while writing to file '{file}': {e}")


def save_to_json(file: str, data: dict) -> None:  
    lockfile = file + ".lock"
    lock = FileLock(lockfile)

    with lock: 
        all_data = load_data(file)
        all_data.append(data)
        dump_data(file, all_data)
        
        
def save_to_csv(df: pd.DataFrame, output_file: str):
    df.to_csv(output_file, index=False)


def compute_average_across_scenes(results_file: Path | str | None) -> tuple[dict, pd.DataFrame]:
    
    if not results_file:
        results_file = get_args().path
        
    result = load_data(results_file)
    
    df = pd.json_normalize(result, sep=':')
    df.rename(columns={"params:test_scene": "test_scene"}, inplace=True, errors="ignore")
    
    df.columns = pd.MultiIndex.from_tuples([col.split(":") for col in df.columns])
    df.columns = pd.MultiIndex.from_tuples([tuple('' if pd.isna(name) else name for name in col) for col in df.columns])
    
    
    param_columns = [("params", col) for col in df.get("params", pd.DataFrame()).columns.tolist()]
    result_columns = [("result", col) for col in df.get("result", pd.DataFrame()).columns.tolist()]
    
    if param_columns:
        df = df.set_index(param_columns)
        result_avg = df.groupby(level=[0, 1])[result_columns].mean()
    else:
        result_avg = df[result_columns].mean().to_frame().T

    result_avg.columns = pd.MultiIndex.from_product([["average_result"], result_avg['result'].columns])
    
    df = df.join(result_avg).reset_index()
    
    columns_to_sort = result_avg.columns.tolist() + param_columns + [('test_scene', '')]
    sort_orders = [False if any('psnr' or 'ssim' in item for item in col) else True for col in result_avg.columns.tolist()] + [True] * (len(param_columns) + 1)
    df.sort_values(columns_to_sort, ascending=sort_orders, inplace=True)
    
    result = df.iloc[0]['average_result'].to_dict()
    
    return result, df




def get_args():
  args = argparse.ArgumentParser(description='Average training results across the scenes.')
  args.add_argument('results_file', type=str, help='/Path/to/the/result/file.json')

  return args.parse_args()




if __name__ == "__main__":
    args = get_args()
    result, df = compute_average_across_scenes(args.results_file)
    output_file = os.path.join(os.path.dirname(args.results_file), 'averaged_results.csv')
    save_to_csv(df, output_file)