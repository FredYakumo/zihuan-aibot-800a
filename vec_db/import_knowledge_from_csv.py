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
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Add this function to convert timestamps
def convert_to_rfc3339(timestamp_str: str):
    if not timestamp_str:
        return ""
    try:
        # Parse the timestamp string
        dt = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S%z')
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
    
    df = pd.read_csv(file_path)
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
                
                keyword_list = []
                if not pd.isna(e["keyword"]):
                    keyword_list = [i.strip() for i in e["keyword"].split(",")]
                    print(f"Use file keywords: {','.join(keyword_list)}")
                else:
                    keyword_list = list(jieba.cut(content))
                    print(f"Use jieba cut keywords: {','.join(keyword_list)}")

                creator_name = e["creator_name"] if not pd.isna(e["creator_name"]) else ""
                create_time = convert_to_rfc3339(e["create_time"]) if not pd.isna(e["create_time"]) else ""
                
                # Calculate embeddings for keywords
                keyword_text = " ".join(keyword_list)
                keyword_inputs = tokenizer(keyword_text, return_tensors="pt", padding=True, truncation=True)
                keyword_outputs = model(**keyword_inputs)
                keyword_embedding = mean_pooling(keyword_outputs, keyword_inputs["attention_mask"]).detach().numpy().tolist()[0]
                
                # logger.info("Delete old same content...")
                # collection.data.delete_many(Filter.by_property("content").equal(content))
                
                logger.info("Adding object.")
                batch.add_object(properties={
                    "keyword": keyword_list,
                    "content": content,
                    "create_time": create_time,
                    "creator_name": creator_name
                }, vector=keyword_embedding)
                if batch.number_errors > 10:
                    logger.error("Batch import stopped due to excessive errors.")
                    break

    failed_objects = collection.batch.failed_objects
    if failed_objects:
        logger.error(f"Number of failed imports: {len(failed_objects)}")
        logger.error(f"First failed object: {failed_objects[0]}")
    else:
        logger.info("导入完成")