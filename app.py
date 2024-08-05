from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient

class APP():


    model_name = "BAAI/bge-large-en"
    model_kwargs = {
        'device': 'cpu'
    }
    encode_kwargs = {
        'normalize_embeddings': False
    }

    embeddings = HuggingFaceBgeEmbeddings(
        model_kwargs=model_kwargs,
        model_name=model_name,
        encode_kwargs=encode_kwargs
    )

    url = "http://localhost:6333"
    collection_name = "gpt_db"

    client = QdrantClient(
        url=url,
        prefer_grpc=False
    )

    print(client)
    print("###############")
    db = Qdrant(
        client=client,
        embeddings=embeddings,
        collection_name=collection_name
    )
    print(db)
    print("###############")

    query = "What are some of the limitations of GPT-4?"

    docs = db.similarity_search_with_score(
        query=query,
        k=5
    )

    for i in docs:
        doc, score = i
        print({
            "score": score,
            "content": doc.page_content,
            "metadata": doc.metadata
        })