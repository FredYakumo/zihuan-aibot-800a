curl -X POST http://localhost:28080/v1/schema \
    -H "Content-Type: application/json" \
    -d '{
  "class": "Group_message",
  "description": "群聊消息",
  "vectorizers": [
    {
      "name": "text2vec-ollama",
      "model": "herald/dmeta-embedding-zh",
      "apiEndpoint": "http://host.docker.internal:11434",
      "textFields": ["content", "sender_name"]
    }
  ],
  "moduleConfig": {
    "text2vec-ollama": {
                "apiEndpoint": "http://host.docker.internal:11434",
                "model": "herald/dmeta-embedding-zh"
    }
  },
 "properties": [
    {
        "dataType": [
        "text"
        ],
        "description": "群号",
        "indexFilterable": true,
        "indexRangeFilters": false,
        "indexSearchable": true,
        "moduleConfig": {
        "text2vec-ollama": {
            "skip": true,
            "vectorizePropertyName": false
        }
        },
        "name": "group_id",
        "tokenization": "whitespace"
    },
    {
        "dataType": [
        "text"
        ],
        "description": "群名",
        "indexFilterable": true,
        "indexRangeFilters": false,
        "indexSearchable": true,
        "moduleConfig": {
        "text2vec-ollama": {
            "skip": true,
            "vectorizePropertyName": false
        }
        },
        "name": "group_name",
        "tokenization": "word"
    },
    {
        "dataType": [
        "text"
        ],
        "description": "发送者qq号",
        "indexFilterable": true,
        "indexRangeFilters": false,
        "indexSearchable": true,
        "moduleConfig": {
        "text2vec-ollama": {
            "skip": true,
            "vectorizePropertyName": false
        }
        },
        "name": "sender_id",
        "tokenization": "whitespace"
    },
    {
        "dataType": [
        "text"
        ],
        "description": "发送者名称",
        "indexFilterable": true,
        "indexRangeFilters": false,
        "indexSearchable": true,
        "moduleConfig": {
        "text2vec-ollama": {
            "skip": true,
            "vectorizePropertyName": false
        }
        },
        "name": "sender_name",
        "tokenization": "word"
    },
    {
        "dataType": [
        "date"
        ],
        "description": "消息发送时间",
        "indexFilterable": true,
        "indexRangeFilters": true,
        "indexSearchable": false,
        "moduleConfig": {
        "text2vec-ollama": {
            "skip": true,
            "vectorizePropertyName": false
        }
        },
        "name": "send_time"
    },
    {
        "dataType": [
        "text"
        ],
        "description": "消息内容",
        "indexFilterable": true,
        "indexRangeFilters": false,
        "indexSearchable": true,
        "moduleConfig": {
        "text2vec-ollama": {
            "skip": false,
            "vectorizePropertyName": false
        }
        },
        "name": "content",
        "tokenization": "word"
    }
    ]
}'
