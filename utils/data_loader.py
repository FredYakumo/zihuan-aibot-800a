import pandas as pd
import os
import datetime
import pytz
from utils.logging_config import logger

def read_data_from_path(file_path):
    """读取单个文件的数据"""
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".json"):
        return pd.read_json(file_path)
    return None

def collect_all_files(directory):
    """递归收集目录下所有的csv和json文件"""
    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith((".csv", ".json")):
                all_files.append(os.path.join(root, file))
    return all_files

def load_knowledge_from_path(file_path: str) -> pd.DataFrame | None:
    if os.path.isdir(file_path):
        all_files = collect_all_files(file_path)
        logger.info(f"找到 {len(all_files)} 个文件待处理")
        # 合并所有数据框
        dfs = []
        for f in all_files:
            logger.info(f"正在读取文件: {f}")
            df = read_data_from_path(f)
            if df is not None:
                dfs.append(df)
        if not dfs:
            logger.error("未找到有效的数据文件")
            return None
        df = pd.concat(dfs, ignore_index=True)
    elif os.path.isfile(file_path):
        df = read_data_from_path(file_path)
    else:
        logger.error(f"文件路径 {file_path} 无效")
        return None
    return df