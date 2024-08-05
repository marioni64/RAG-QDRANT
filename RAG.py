import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma, Qdrant
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

loader = PyPDFLoader("data.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size= 500,
    chunk_overlap= 50
)

texts = text_splitter.split_documents(documents)


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


openapi_api_key = os.getenv("sk-proj-HJDZRZXAzIsjRsX36xhRT3BlbkFJPBY0fIauMJh2JirMPmEV")
model_gpt = "gpt-3.5-turbo"

embeddings = OpenAIEmbeddings(
    openapi_api_key = openapi_api_key
)
vector = Chroma.from_documents(documents, embeddings)
llm = ChatOpenAI(openapi_api_key = openapi_api_key)
output_parser = StrOutputParser()
retriever = vector.as_retriever()


instruction_to_system = """
Given a chat history and the latest user question
which might reference context in the chat history, formulate a standalone question 
which can be understood without the chat history. Do NOT answer the question, 
just reformulate it if needed and otherwise return it as is
"""
question_maker_promt = ChatPromptTemplate.from_message(
    [
        ("system", instruction_to_system),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ]
)

question_chain = question_maker_promt | llm | StrOutputParser()

question_chain.invoke({"question": "can you explain more?",
                       "chat_history": [HumanMessage(content="you explained that the moon is round")]})


qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, provide a summary of the context. Do not generate your answer.\

{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def contextualized_question(input: dict):
    if input.get("chat_history"):
        return question_chain
    else:
        return input["question"]

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



retriever_chain = RunnablePassthrough.assign(
        context=contextualized_question | retriever | format_docs
    )

retriever_chain.invoke({
    "chat_history": [HumanMessage(content="you explained that the moon is round")],
    "question": "can you explain more?"
})


rag_chain = (
    retriever_chain | qa_prompt | llm
)

question = "What are some of the limitations of GPT-4?"


chat_history = []

ai_msg = rag_chain.invoke({
    "question": question,
    "chat_history": chat_history
})
chat_history.extend([HumanMessage(content=question), ai_msg])

print(ai_msg.content)

question = "Can you explain more?"
ai_msg = rag_chain.invoke({
    "question": question,
    "chat_history": chat_history
})
chat_history.extend([HumanMessage(content=question), ai_msg])



















"""def __init__(self, api_key ="sk-proj-HJDZRZXAzIsjRsX36xhRT3BlbkFJPBY0fIauMJh2JirMPmEV", retry_delay: float = 1.0) -> None:
    self.messages = {}
    self.chat = ChatOpenAI(open_api_key = api_key)
    self.model = ChatOpenAI(model="gpt-3.5-turbo")
    self.retry_delay = retry_delay


def new_rag(self, username: str, rag_content: str) -> None:
    self.messages[username] = [
        SystemMessage(content="Ты помошник. Твое имя RAG."),
        HumanMessage(content=make_rag(rag_content)),
        AIMessage(content="Я понял. Я отвечу на выши вопросы по этоому документу.")
    ]


def ask(self, username: str, query: str) -> None:
    if username not in self.messages:
        self.reset(username)
    self.messages[username].append(HumanMessage(content=query))
    try:
        res = self.chat(self.messages[username])
    except Exception as e:
        if "429" in str(e):
            time.sleep(self.retry_delay)
            self.retry_delay *= 2
            return self.ask(username, query)
        raise e
    self.messages[username].append(res)
    return res.content


def reset(self, username: str) -> None:
    self.messages[username] = [SystemMessage(content="Ты помощник.")]
    self.retry_delay = 1.0


messages = [
    SystemMessage(content=""),
    HumanMessage(content=""),
    AIMessage(content=""),
    HumanMessage(content="")
]


result = model.invoke(messages)
print(f"Answer from AI:{result.content}")"""
