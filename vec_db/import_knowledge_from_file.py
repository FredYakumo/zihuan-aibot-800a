import sys
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


print("正在加载embedding模型...")
start_time = time.time()
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("GanymedeNil/text2vec-large-chinese")
model = AutoModel.from_pretrained("GanymedeNil/text2vec-large-chinese")
end_time = time.time()
print(f"加载完成, 耗时: {end_time - start_time:.2f}秒")

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


schema_name = config.knowledge_collection_name

if __name__ == "__main__":
    if len(sys.argv) < 3:
        logger.error("请提供要导入的知识库csv和字典文件和停用词列表文件")
        sys.exit(1)
    file_path = sys.argv[1]
    dict_path = sys.argv[2]
    stop_words_path = sys.argv[3]

    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".json"):
        df = pd.read_json(file_path)
    logger.info(df)
    jieba.load_userdict(dict_path)
    logger.info(jieba.user_word_tag_tab)

    logger.info("正在加载停用词列表")
    with open(stop_words_path, "r", encoding="utf-8") as f:
        stop_words = set(f.read().splitlines())
    logger.info(f"停用词列表加载完成, 数量: {len(stop_words)}")

    with weaviate.connect_to_local(port=config.vector_db_port) as client:
        collection = client.collections.get(schema_name)
        logger.info(collection.config.get())

        with collection.batch.fixed_size(10) as batch:
            for _, e in df.iterrows():
                logger.info(e)

                content = e["content"] if not pd.isna(e["content"]) else ""

                class_name_list = []
                if pd.isna(e["class_name_list"]) or len(e["class_name_list"]) > 0:
                    print("No class name")
                else:
                    class_name_list = [
                        i.strip() for i in e["class_name_list"].split("|")
                    ]
                    print(f"Class_name_list: {','.join(class_name_list)}")

                creator_name = (
                    e["creator_name"] if not pd.isna(e["creator_name"]) else ""
                )
                if "create_time" not in df.columns:
                    create_time = datetime.datetime.now(pytz.UTC).isoformat()
                else:
                    create_time = (
                        convert_to_rfc3339(e["create_time"])
                        if not pd.isna(e["create_time"])
                        else datetime.datetime.now(pytz.UTC).isoformat()
                    )

                # Calculate embeddings for class_name_list
                class_name_text = " ".join(class_name_list)
                class_name_inputs = tokenizer(
                    class_name_text, return_tensors="pt", padding=True, truncation=True
                )
                class_name_outputs = model(**class_name_inputs)
                class_name_embedding = (
                    mean_pooling(
                        class_name_outputs, class_name_inputs["attention_mask"]
                    )
                    .detach()
                    .numpy()
                    .tolist()[0]
                )

                # Calculate embeddings for content
                # 先对content进行分词
                words = jieba.cut(content)
                # 去除停用词
                filtered_words = [
                    word for word in words if word not in stop_words and word.strip()
                ]
                # 重新组合成文本
                filtered_content = " ".join(filtered_words)
                # 计算embedding
                content_inputs = tokenizer(
                    filtered_content, return_tensors="pt", padding=True, truncation=True
                )
                content_outputs = model(**content_inputs)
                content_embedding = (
                    mean_pooling(content_outputs, content_inputs["attention_mask"])
                    .detach()
                    .numpy()
                    .tolist()[0]
                )

                # Combine embeddings with weights only if class_name_list is not empty
                if class_name_list:
                    # Combine embeddings with weights (3x for class_name_list, 1x for content)
                    class_name_tensor = torch.tensor(class_name_embedding)
                    content_tensor = torch.tensor(content_embedding)
                    combined_embedding = (
                        (3 * class_name_tensor + content_tensor) / 4
                    ).tolist()
                else:
                    # Use content embedding only if no class names
                    combined_embedding = content_embedding

                logger.info("Adding object.")
                batch.add_object(
                    properties={
                        "class_name_list": class_name_list,
                        "content": content,
                        "create_time": create_time,
                        "creator_name": creator_name,
                    },
                    vector=combined_embedding,
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
