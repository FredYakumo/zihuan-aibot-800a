from config_loader import config
import weaviate
import pandas as pd

if __name__ == "__main__":
    with weaviate.connect_to_local(port=config.vector_db_port) as client:
        old_schema = client.collections.get(config.knowledge_collection_name)
        old_objs = []
        for e in old_schema.query.fetch_objects().objects:
            old_objs.append({
                "content": e.properties.get("content", None),
                "create_time": e.properties.get("create_time", None),
                "creator_name": e.properties.get("creator_name", None)
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(old_objs)
        
        # Export to CSV
        output_file = "export_knowledge.csv"
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Data exported to {output_file}")