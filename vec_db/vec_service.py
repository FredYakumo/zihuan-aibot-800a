import time
from typing import List

print("正在加载embedding模型...")
start_time = time.time()
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("GanymedeNil/text2vec-large-chinese")
model = AutoModel.from_pretrained("GanymedeNil/text2vec-large-chinese")
end_time = time.time()
print(f"加载embedding模型完成, 加载耗时: {end_time - start_time:.2f}秒")

from fastapi import FastAPI, Header
from pydantic import BaseModel
import weaviate
from weaviate.classes.query import Filter
from config_loader import config
from logging_config import logger
from model import Knowledge


app = FastAPI()

g_vec_db_client = weaviate.connect_to_local(port=config.vector_db_port)
g_vec_db_collection = g_vec_db_client.collections.get(config.knowledge_collection_name)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embedding(text):
    # Encode the input text
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    
    # Calculate token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Apply mean pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    
    # Normalize the embeddings
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    
    return sentence_embeddings[0].tolist()


class QueryKnowledgeRequest(BaseModel):
    query: str

def query_knowledge(query: str) -> List[Knowledge]:
    logger.info(f"从向量数据库中查询: {query}")
    start_time = time.time()
    vector = get_embedding(query)
    res = g_vec_db_collection.query.near_vector(
        near_vector=vector,
        # limit=5,
        certainty=0.72,
        return_metadata=["certainty"],
        return_properties=["keyword", "content", "create_time", "creator_name"],
    )
    end_time = time.time()
    knowledge_result = []
    for e in res.objects:
        logger.info(f"{e.properties.get('content')} - 创建者: {e.properties.get('creator_name')} - 时间: {e.properties.get("create_time")}, 置信度: {e.metadata.certainty}, keyword: {e.properties.get('keyword')}")
        knowledge_result.append(Knowledge(
            keyword=e.properties.get("keyword"),
            content=e.properties.get("content"),
            create_time=e.properties.get("create_time"),
            creator_name=e.properties.get("creator_name"),
            certainty=e.metadata.certainty
        ))
    logger.info(f"查询耗时: {end_time - start_time:.2f}秒")
    return knowledge_result


@app.post("/query_knowledge")
def query_data(request: QueryKnowledgeRequest):
    return query_knowledge(request.query)

@app.post("/find_keyword_match")
def find_keyword_match(request: QueryKnowledgeRequest):
    logger.info(f"从向量数据库中精确查找关键字: {request.query}")
    start_time = time.time()
    res = g_vec_db_collection.query.fetch_objects(
        filters=Filter.by_property("keyword").contains_any([request.query]),
        return_properties=["keyword", "content", "create_time", "creator_name"]
    )
    end_time = time.time()
    knowledge_result = []
    for e in res.objects:
        logger.info(f"{e.properties.get('content')} - 创建者: {e.properties.get('creator_name')} - 时间: {e.properties.get("create_time")}, 置信度: {e.metadata.certainty}, keyword: {e.properties.get('keyword')}")
        knowledge_result.append(Knowledge(
            keyword=e.properties.get("keyword"),
            content=e.properties.get("content"),
            create_time=e.properties.get("create_time"),
            creator_name=e.properties.get("creator_name"),
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