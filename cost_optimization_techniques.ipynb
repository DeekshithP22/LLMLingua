from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import os
# Helper function for printing docs
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )
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
    temperature=0.2,
)

embedding = AzureOpenAIEmbeddings(
    model= embed_model,
    azure_deployment= embedding_deployment_name,
    api_key= api_key,
    azure_endpoint= azure_endpoint,
    api_version= api_version,
)
Normal Retriever
import warnings
warnings.filterwarnings("ignore")

documents = TextLoader(
    "state_of_the_union.txt",encoding="UTF-8", autodetect_encoding=True,
).load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
retriever = FAISS.from_documents(texts, embedding).as_retriever(search_kwargs={"k": 3})

docs = retriever.get_relevant_documents(
    "What did the president say about Ketanji Brown Jackson?"
)
pretty_print_docs(docs)
Contextual Compression Retriever and LLMChainExtractor Doccument Compressors
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import OpenAI

llm = llm # Could be any llm of your choice
compressor = LLMChainExtractor.from_llm(llm)
contextual_compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

compressed_docs_compression_retriever = contextual_compression_retriever.get_relevant_documents(
    "What did the president say about Ketanji Jackson Brown"
)
pretty_print_docs(compressed_docs_compression_retriever)
original_contexts_len = len("\n\n".join([d.page_content for i, d in enumerate(docs)]))
compressed_contexts_len = len("\n\n".join([d.page_content for i, d in enumerate(compressed_docs_compression_retriever)]))

print("Original context length:", original_contexts_len)
print("Compressed context length:", compressed_contexts_len)
print("Compressed Ratio:", f"{original_contexts_len/(compressed_contexts_len + 1e-5):.2f}x")
chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
query = "What did the president say about Ketanji Jackson Brown"
chain.invoke({"query": query})
contextual_compression_retriever_chain = RetrievalQA.from_chain_type(llm=llm, retriever=contextual_compression_retriever)
contextual_compression_retriever_chain.invoke({"query": query})
Contextual Compression Retriver and EmbeddingsFilter Doccuments Compressor
from langchain.retrievers.document_compressors import EmbeddingsFilter


embeddings = embedding # could be any embedding of your choice
embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.79)
Embed_filter_compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter, base_retriever=retriever
)

compressed_docs_embfilter_retriever = Embed_filter_compression_retriever.get_relevant_documents(
    "What did the president say about Ketanji Jackson Brown"
)
pretty_print_docs(compressed_docs_embfilter_retriever)
original_contexts_len = len("\n\n".join([d.page_content for i, d in enumerate(docs)]))
compressed_contexts_len = len("\n\n".join([d.page_content for i, d in enumerate(compressed_docs_embfilter_retriever)]))

print("Original context length:", original_contexts_len)
print("Compressed context length:", compressed_contexts_len)
print("Compressed Ratio:", f"{original_contexts_len/(compressed_contexts_len + 1e-5):.2f}x")
chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
query = "What did the president say about Ketanji Jackson Brown"
chain.invoke({"query": query})
Embed_filter_retriver_chain = RetrievalQA.from_chain_type(llm=llm, retriever=contextual_compression_retriever)
Embed_filter_retriver_chain.invoke({"query": query})
Stringing Doccuments transformers and Compressor together using pipeline
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import EmbeddingsFilter

embeddings = embedding
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.78)
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter, relevant_filter]
)

from langchain.retrievers import ContextualCompressionRetriever
pipeline_compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor, base_retriever=retriever
)

pipeline_compressed_docs = pipeline_compression_retriever.get_relevant_documents(
    "What did the president say about Ketanji Jackson Brown?"
)
pretty_print_docs(pipeline_compressed_docs)
original_contexts_len = len("\n\n".join([d.page_content for i, d in enumerate(docs)]))
compressed_contexts_len = len("\n\n".join([d.page_content for i, d in enumerate(pipeline_compressed_docs)]))

print("Original context length:", original_contexts_len)
print("Compressed context length:", compressed_contexts_len)
print("Compressed Ratio:", f"{original_contexts_len/(compressed_contexts_len + 1e-5):.2f}x")
chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
query = "What did the president say about Ketanji Jackson Brown"
chain.invoke({"query": query})
pipeline_retriver_chain = RetrievalQA.from_chain_type(llm=llm, retriever=pipeline_compression_retriever)
pipeline_retriver_chain.invoke({"query": query})
LongLLMLingua 
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import os
api_key = os.getenv("OPENAI_API_KEY")
azure_endpoint = os.getenv("OPENAI_ENDPOINT")
api_version = os.getenv("OPENAI_API_VERSION")
model = os.getenv("OPENAI_MODEL")
deployment_name = os.getenv("OPENAI_DEPLOYMENT")
embed_model = os.getenv("OPENAI_EMBEDDING_MODEL")
embedding_deployment_name = os.getenv("OPENAI_EMBEDDING_DEPLOYMENT_NAME")

# Load OpenAI chat model
llm1 = AzureOpenAI(
    model= model,
    azure_deployment= deployment_name,
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
    # system_prompt= prompt,
    temperature=0.2,
)

embedding1 = AzureOpenAIEmbedding(
    model= embed_model,
    azure_deployment= embedding_deployment_name,
    api_key= api_key,
    azure_endpoint= azure_endpoint,
    api_version= api_version,
)

from llama_index.core import (
    VectorStoreIndex,
    download_loader,
    load_index_from_storage,
    StorageContext,
    ServiceContext
)

WikipediaReader = download_loader("WikipediaReader")
loader = WikipediaReader()
documents = loader.load_data(pages=['Mexican–American_War'])

service_context = ServiceContext.from_defaults(
    embed_model=embeddings,
    llm=llm
)

index = VectorStoreIndex.from_documents(documents, service_context=service_context)

retriever = index.as_retriever(similarity_top_k=3)

question = "What were the main outcomes of the war"
contexts = retriever.retrieve(question)

context_list = [n.get_content() for n in contexts]
context_list
# Setup LLMLingua
from llama_index.postprocessor.longllmlingua import LongLLMLinguaPostprocessor
node_postprocessor = LongLLMLinguaPostprocessor(
    device_map='cpu',
    instruction_str="Given the context, please answer the final question",
    target_token=300,
    rank_method="longllmlingua",
    additional_compress_kwargs={
        "condition_compare": True,
        "condition_in_question": "after",
        "context_budget": "+100",
        "reorder_context": "sort",  # enable document reorder,
        "dynamic_context_compression_ratio": 0.3,
    },
)
from llama_index.core.indices.query.schema import QueryBundle

new_retrieved_nodes = node_postprocessor.postprocess_nodes(
    retrieved_nodes, query_bundle=QueryBundle(query_str=question)
)

new_context_list = [n.get_content() for n in new_retrieved_nodes]
new_context_list
original_contexts = "\n\n".join([n.get_content() for n in retrieved_nodes])
compressed_contexts = "\n\n".join([n.get_content() for n in new_retrieved_nodes])
original_tokens = node_postprocessor._llm_lingua.get_token_length(original_contexts)
compressed_tokens = node_postprocessor._llm_lingua.get_token_length(compressed_contexts)

print("Original Tokens:", original_tokens)
print("Compressed Tokens:", compressed_tokens)
print("Compressed Ratio:", f"{original_tokens/(compressed_tokens + 1e-5):.2f}x")
