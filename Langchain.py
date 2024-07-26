%pip install --upgrade --quiet  llmlingua accelerate
# Helper function for printing docs

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )
%pip install langchain-community
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
import os
api_key = os.getenv("OPENAI_API_KEY")
azure_endpoint = os.getenv("OPENAI_ENDPOINT")
api_version = os.getenv("OPENAI_API_VERSION")
model = os.getenv("OPENAI_MODEL")
deployment_name = os.getenv("OPENAI_DEPLOYMENT")
embed_model = os.getenv("OPENAI_EMBEDDING_MODEL")
embedding_deployment_name = os.getenv("OPENAI_EMBEDDING_DEPLOYMENT_NAME")

# Load OpenAI chat model
llm = AzureChatOpenAI(
    model= model,
    azure_deployment= deployment_name,
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
    # system_prompt= prompt,
    temperature=0.2,
)

embedding = AzureOpenAIEmbeddings(
            model= embed_model,
            azure_deployment= embedding_deployment_name,
            api_key= api_key,
            azure_endpoint= azure_endpoint,
            api_version= api_version,
        )

%pip install faiss-cpu
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

documents = TextLoader(
    "state_of_the_union.txt",encoding="UTF-8", autodetect_encoding=True,
).load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

embedding = embedding
retriever = FAISS.from_documents(texts, embedding).as_retriever(search_kwargs={"k": 20})

query = "What did the president say about Ketanji Brown Jackson"
docs = retriever.invoke(query)
pretty_print_docs(docs)
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import LLMLinguaCompressor


llm = llm

compressor = LLMLinguaCompressor(model_name="openai-community/gpt2", device_map="cpu")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke(
    "What did the president say about Ketanji Jackson Brown"
)
pretty_print_docs(compressed_docs)
