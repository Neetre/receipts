from qdrant_client import QdrantClient

def connect_to_qdrant(host='localhost', port=6333):
    """
    Establish a connection to the Qdrant database.
    
    :param host: Qdrant server host (default: localhost)
    :param port: Qdrant server port (default: 6333)
    :return: QdrantClient instance
    """
    try:
        client = QdrantClient(host=host, port=port)
        print("Successfully connected to Qdrant database")
        return client
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        return None

def read_vectors(client, collection_name, limit=10):
    """
    Read vectors from a specified Qdrant collection.
    
    :param client: QdrantClient instance
    :param collection_name: Name of the collection to read from
    :param limit: Maximum number of records to retrieve (default: 10)
    :param filter_conditions: Optional filter to apply to the query
    :return: List of retrieved records
    """
    try:
        search_params = {
            "limit": limit
        }

        results = client.scroll(
            collection_name=collection_name,
            **search_params
        )
        
        return results
    except Exception as e:
        print(f"Error reading from collection {collection_name}: {e}")
        return None

def main():
    client = connect_to_qdrant()
    collection_info = client.get_collection("receipts")
    total_points = collection_info.points_count
    print(total_points)
    
    if not client:
        return
    collection_name = "receipts"

    print("Reading all vectors:")
    all_vectors = read_vectors(client, collection_name)
    if all_vectors:
        for vector in all_vectors[0]:
            print(vector)

if __name__ == "__main__":
    main()