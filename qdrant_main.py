import uuid
import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct


openai.api_key = "sk-proj-HJDZRZXAzIsjRsX36xhRT3BlbkFJPBY0fIauMJh2JirMPmEV"

record = 0
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

qdrant_client = QdrantClient(
    url="https://309e7711-786b-4c2c-aa32-6fff465f6bf9.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="481cCVm0Ks14GPiWtcjv9zWFsNreetkCjdBcUg3X19_IaxiG80756w",
)

qdrant_client.recreate_collection(
    collection_name="Gyber_Collection",
    vectors_config=models.VectorParams(size= 1536, distance=models.Distance.COSINE),
)

print("Create collection response:", qdrant_client)

info = qdrant_client.get_collection(collection_name="Gyber_Collection")
print("Collection info: ", info)
for get_info in info:
    print(get_info)


def read_data_from_pdf():
    loader = PyPDFLoader("data.pdf")
    documents = loader.load()

    return documents

def get_text_chunks(documents):

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_documents(documents)
    return chunks


def get_embedding(text_chunks, model_id="text-embedding-ada-002"):
    points = []
    for idx, chunk in enumerate(text_chunks):
        response = openai.Embedding.create(
            input=chunk,
            model=model_id
        )
        embeddings = response['data'][0]['embedding']
        point_id = str(uuid.uuid4())

        points.append(PointStruct(id=point_id, vector=embeddings, payload={"text": chunk}))

    return points


def insert_data(get_points):

    operation_info = qdrant_client.upsert(
    collection_name="Gyber_Collection",
    wait=True,
    points=get_points
)
    return operation_info


def create_answer_with_context(query):
    response = openai.Embedding.create(
        input=query,

        model="text-embedding-ada-002"
    )
    embeddings = response['data'][0]['embedding']

    search_result = qdrant_client.search(
        collection_name="Gyber_Collection",
        query_vector=embeddings,
        limit=1
    )

    prompt = "Context:\n"
    for result in search_result:
        prompt += result.payload['text'] + "\n---\n"
    prompt += "Question:" + query + "\n---\n" + "Answer:"

    print("----PROMPT START----")
    print(":", prompt)
    print("----PROMPT END----")

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
        )

    return completion.choices[0].message.content



def main():
    get_raw_text=read_data_from_pdf()
    chunks=get_text_chunks(get_raw_text)
    vectors=get_embedding(chunks)

    insert_data(vectors)
    question="What are some of the limitations of GPT-4?"
    answer=create_answer_with_context(question)
    print(answer)


if __name__ == '__main__':
    main()