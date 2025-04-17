import weaviate

if __name__ == "__main__":
    with weaviate.connect_to_local(port="28080") as client:
        old_schema = client.collections["AIBot_knowledge"]