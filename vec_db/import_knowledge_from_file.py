import sys
import os
import weaviate
import pandas as pd

# Load model directly
import jieba
import torch
import time
import pytz
import datetime

from utils.config_loader import config
from utils.logging_config import logger
from nn.models import get_device

device = get_device()

print("正在加载embedding模型...")
start_time = time.time()
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("GanymedeNil/text2vec-large-chinese")
model = AutoModel.from_pretrained("GanymedeNil/text2vec-large-chinese")
end_time = time.time()
print(f"加载完成, 耗时: {end_time - start_time:.2f}秒")

# 将模型移动到指定设备
model = model.to(device)

from weaviate.classes.query import Filter


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


# Add this function to convert timestamps
def convert_to_rfc3339(timestamp_str: str):
    if not timestamp_str:
        return ""
    try:
        # Parse the timestamp string
        dt = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S%z")
        # Convert to RFC3339 format
        return dt.astimezone(pytz.UTC).isoformat()
    except ValueError:
        logger.warning(f"Could not parse timestamp {timestamp_str}")
        return ""


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


schema_name = config.knowledge_collection_name

if __name__ == "__main__":
    if len(sys.argv) < 3:
        logger.error("请提供要导入的知识库路径、字典文件和停用词列表文件")
        sys.exit(1)
    file_path = sys.argv[1]
    dict_path = sys.argv[2]
    stop_words_path = sys.argv[3]

    # 收集所有需要处理的文件
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
            sys.exit(1)
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = read_data_from_path(file_path)
        if df is None:
            logger.error("不支持的文件格式，仅支持csv和json")
            sys.exit(1)

    logger.info(f"总数据条数: {len(df)}")
    logger.info(df)

    jieba.load_userdict(dict_path)
    logger.info(jieba.user_word_tag_tab)

    logger.info("正在加载停用词列表")
    with open(stop_words_path, "r", encoding="utf-8") as f:
        stop_words = set(f.read().splitlines())
    logger.info(f"停用词列表加载完成, 数量: {len(stop_words)}")

    logger.info("开始预处理数据...")
    stage_start_time = time.time()
    all_contents = []
    all_class_names = []
    metadata = []
    
    # 首先收集所有数据
    for _, e in df.iterrows():
        content = e["content"] if not pd.isna(e["content"]) else ""
        all_contents.append(content)
        
        class_name_list = []
        if not pd.isna(e["class_name_list"]) and len(e["class_name_list"]) >= 1:
            class_name_list = [i.strip() for i in e["class_name_list"].split("|")]
        all_class_names.append(" ".join(class_name_list))
        
        creator_name = e["creator_name"] if not pd.isna(e["creator_name"]) else ""
        create_time = (datetime.datetime.now(pytz.UTC).isoformat() 
                      if "create_time" not in df.columns
                      else convert_to_rfc3339(e["create_time"]) if not pd.isna(e["create_time"]) 
                      else datetime.datetime.now(pytz.UTC).isoformat())
        
        metadata.append({
            "class_name_list": class_name_list,
            "creator_name": creator_name,
            "create_time": create_time
        })
    
    stage_end_time = time.time()
    logger.info(f"数据收集完成, 耗时: {stage_end_time - stage_start_time:.2f}秒")

    # 批量计算content embeddings
    logger.info("开始批量计算content embeddings...")
    stage_start_time = time.time()
    content_inputs = tokenizer(all_contents, 
                             padding=True, 
                             truncation=True, 
                             return_tensors="pt")
    content_inputs = {k: v.to(device) for k, v in content_inputs.items()}
    
    with torch.no_grad():
        content_outputs = model(**content_inputs)
        content_embeddings = mean_pooling(content_outputs, 
                                        content_inputs["attention_mask"]).cpu().numpy()
    stage_end_time = time.time()
    logger.info(f"content embeddings计算完成, 耗时: {stage_end_time - stage_start_time:.2f}秒")

    # 批量计算class_name embeddings
    logger.info("开始批量计算class_name embeddings...")
    stage_start_time = time.time()
    class_name_inputs = tokenizer(all_class_names, 
                                padding=True, 
                                truncation=True, 
                                return_tensors="pt")
    class_name_inputs = {k: v.to(device) for k, v in class_name_inputs.items()}
    
    with torch.no_grad():
        class_name_outputs = model(**class_name_inputs)
        class_name_embeddings = mean_pooling(class_name_outputs, 
                                           class_name_inputs["attention_mask"]).cpu().numpy()
    stage_end_time = time.time()
    logger.info(f"class_name embeddings计算完成, 耗时: {stage_end_time - stage_start_time:.2f}秒")

    # 合并embeddings
    logger.info("开始合并embeddings...")
    stage_start_time = time.time()
    processed_data = []
    for i in range(len(metadata)):
        if metadata[i]["class_name_list"]:
            class_name_tensor = torch.tensor(class_name_embeddings[i], device=device)
            content_tensor = torch.tensor(content_embeddings[i], device=device)
            combined_embedding = ((3 * class_name_tensor + content_tensor) / 4).cpu().numpy()
        else:
            combined_embedding = content_embeddings[i]

        processed_data.append({
            "properties": {
                "class_name_list": metadata[i]["class_name_list"],
                "content": all_contents[i],
                "create_time": metadata[i]["create_time"],
                "creator_name": metadata[i]["creator_name"],
            },
            "vector": combined_embedding.tolist()
        })
        
        if (i + 1) % 100 == 0:
            logger.info(f"已处理 {i + 1} 条数据")
    
    stage_end_time = time.time()
    logger.info(f"embeddings合并完成, 耗时: {stage_end_time - stage_start_time:.2f}秒")

    # 批量导入数据到Weaviate
    logger.info("开始导入数据到向量数据库...")
    stage_start_time = time.time()
    with weaviate.connect_to_local(port=config.vector_db_port) as client:
        collection = client.collections.get(schema_name)
        logger.info(collection.config.get())

        with collection.batch.fixed_size(10) as batch:
            for item in processed_data:
                batch.add_object(
                    properties=item["properties"],
                    vector=item["vector"]
                )
                if batch.number_errors > 10:
                    logger.error("Batch import stopped due to excessive errors.")
                    break

        failed_objects = collection.batch.failed_objects
        if failed_objects:
            logger.error(f"Number of failed imports: {len(failed_objects)}")
            logger.error(f"First failed object: {failed_objects[0]}")
        else:
            logger.info("导入完成")
    
    stage_end_time = time.time()
    logger.info(f"数据导入完成, 耗时: {stage_end_time - stage_start_time:.2f}秒")
