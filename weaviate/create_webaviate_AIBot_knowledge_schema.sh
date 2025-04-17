VECTOR_DB_URL="http://localhost:28080/v1/schema"

curl -X POST $VECTOR_DB_URL \
    -H "Content-Type: application/json" \
    -d '{
  "class": "AIBot_knowledge_new_schema",
  "description": "AIBot 的知识仓库",
  "vectorizers": [
    {
      "name": "text2vec-ollama",
      "model": "herald/dmeta-embedding-zh",
      "apiEndpoint": "http://host.docker.internal:11434",
      "dimensions": 4096
    }
  ],
  "moduleConfig": {
    "text2vec-ollama": {
                "apiEndpoint": "http://host.docker.internal:11434",
                "model": "herald/dmeta-embedding-zh"
    },
    "generative-ollama": {
                "apiEndpoint": "http://host.docker.internal:11434",
                "model": "DeepSeek-R1:14b"
    }
  },
  "properties": [
    {
      "name": "creator_name",
      "dataType": ["string"],
      "tokenization": "word",
      "description": "创建者名字",
            "text2vec-ollama": {
            "skip": true,
            "vectorizePropertyName": false
        }
    },
    {
      "name": "create_time",
      "dataType": ["date"],
      "description": "创建时间",
        "text2vec-ollama": {
        "skip": true,
        "vectorizePropertyName": false
        }
    },
    {
      "name": "keyword",
      "dataType": ["text"],
      "description": "关键词",
      "vectorize": true,
      "tokenization": "word",
      "vectorizers": [
        {
          "name": "text2vec-ollama",
          "properties": ["content"],
          "skip": false
        }
      ]
    },
    {
      "name": "content",
      "dataType": ["text"],
      "description": "内容",
      "vectorize": true,
      "tokenization": "word",
      "vectorizers": [
        {
          "skip": true
        }
      ]
    }
  ]
}'
