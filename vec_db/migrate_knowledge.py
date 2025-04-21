import weaviate

if __name__ == "__main__":
    with weaviate.connect_to_local(port="28080") as client:
        old_schema = client.collections.get("AIBot_knowledge")
        old_objs = []
        for e in old_schema.query.fetch_objects().objects:
            old_objs.append({
                "content": e.properties["content"],
                "create_time": e.properties["create_time"],
                "creator_name": e.properties["creator_name"],
            })
        for e in old_objs:
            print(e)