import datetime
import time
from typing import List, cast
from utils.logging_config import logger
from nn.models import get_device, TextEmbedder

logger.info("正在加载embedding模型...")
start_time = time.time()
from transformers import AutoTokenizer, AutoModel
import torch

device = get_device()

text_embeder = TextEmbedder(device=device, mean_pooling=True)

from fastapi import FastAPI, Header
from pydantic import BaseModel
import weaviate
from weaviate.classes.query import Filter
from utils.config_loader import config
from vec_db.schema_collection import VecDBKnowledge


app = FastAPI()

g_vec_db_client = weaviate.connect_to_local(port=config.vector_db_port)
g_vec_db_collection = g_vec_db_client.collections.get(config.knowledge_collection_name)

class QueryKnowledgeRequest(BaseModel):
    query: str

def query_knowledge(query: str) -> List[VecDBKnowledge]:
    logger.info(f"从向量数据库中查询: {query}")
    start_time = time.time()
    vector = text_embeder.embed(query)
    res = g_vec_db_collection.query.near_vector(
        near_vector=vector,
        limit=6,
        certainty=0.75,
        return_metadata=["certainty"],
        return_properties=["key", "value", "create_time", "creator_name"],
    )
    end_time = time.time()
    knowledge_result = []
    
    for e in res.objects:
        logger.info(f"{e.properties.get('key')}: {e.properties.get('value')} - 创建者: {e.properties.get('creator_name')} - 时间: {e.properties.get('create_time')}, 置信度: {e.metadata.certainty}")
        create_time_val = e.properties.get("create_time")
        knowledge_result.append(VecDBKnowledge(
            key=str(e.properties.get("key")),
            value=str(e.properties.get("value")),
            create_time=cast(datetime.datetime, create_time_val) if create_time_val else None,
            creator_name=str(e.properties.get("creator_name")),
            certainty=e.metadata.certainty or 0.0
        ))
    logger.info(f"查询耗时: {end_time - start_time:.2f}秒")
    return knowledge_result


@app.post("/query_knowledge")
def query_data(request: QueryKnowledgeRequest):
    return query_knowledge(request.query)

@app.post("/find_class_name_match")
def find_class_name_match(request: QueryKnowledgeRequest):
    logger.info(f"从向量数据库中精确查找类别: {request.query}")
    start_time = time.time()
    res = g_vec_db_collection.query.fetch_objects(
        filters=Filter.by_property("class_name_list").contains_any([request.query]),
        return_properties=["key", "value", "create_time", "creator_name"]
    )
    end_time = time.time()
    knowledge_result = []
    for e in res.objects:
        logger.info(f"{e.properties.get('key')}: {e.properties.get('value')} - 创建者: {e.properties.get('creator_name')} - 时间: {e.properties.get('create_time')}")
        create_time_val = e.properties.get("create_time")
        knowledge_result.append(VecDBKnowledge(
            key=str(e.properties.get("key")),
            value=str(e.properties.get("value")),
            create_time=cast(datetime.datetime, create_time_val) if create_time_val else None,
            creator_name=str(e.properties.get("creator_name")),
            certainty=1.0
        ))
    logger.info(f"查询耗时: {end_time - start_time:.2f}秒")
    return knowledge_result

def run():
    # Run the FastAPI application using Uvicorn server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=28085)
    
if __name__ == "__main__":
    run()