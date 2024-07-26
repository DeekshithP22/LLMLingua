%pip install llama-index
%pip install llmlingua
%pip install langchain-openai
%pip install llama-index-embeddings-azure-openai
%pip install llama-index-llms-azure-openai
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


llm = AzureOpenAI(
    model=model,
    deployment_name=deployment_name,
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

# You need to deploy your own embedding model as well as your own chat completion model
embed_model = AzureOpenAIEmbedding(
    model= embed_model,
    azure_deployment= embedding_deployment_name,
    api_key= api_key,
    azure_endpoint= azure_endpoint,
    api_version= api_version,
)
from llama_index.core import Settings
Settings.llm = llm
Settings.embed_model = embed_model
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, load_index_from_storage, StorageContext
# load documents
documents = SimpleDirectoryReader(
    input_files=["Basics_of_finmkts.pdf"]
).load_data()
index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever(similarity_top_k=5)
question = "Short term investment options"
answer = "RISD"
contexts = retriever.retrieve(question)
context_list = [n.get_content() for n in contexts]
len(context_list)
prompt = "\n\n".join(context_list + [question])
response = llm.complete(prompt)
print(str(response))
%pip install llama-index-postprocessor-longllmlingua
%pip install torch
# Setup LLMLingua
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.postprocessor.longllmlingua import LongLLMLinguaPostprocessor
node_postprocessor = LongLLMLinguaPostprocessor(
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
retrieved_nodes = retriever.retrieve(question)
synthesizer = CompactAndRefine()
from llama_index.indices.query.schema import QueryBundle

# outline steps in RetrieverQueryEngine for clarity:
# postprocess (compress), synthesize
new_retrieved_nodes = node_postprocessor.postprocess_nodes(
    retrieved_nodes, query_bundle=QueryBundle(query_str=question)
)
original_contexts = "\n\n".join([n.get_content() for n in retrieved_nodes])
compressed_contexts = "\n\n".join([n.get_content() for n in new_retrieved_nodes])

original_tokens = node_postprocessor._llm_lingua.get_token_length(original_contexts)
compressed_tokens = node_postprocessor._llm_lingua.get_token_length(compressed_contexts)

print(compressed_contexts)
print()
print("Original Tokens:", original_tokens)
print("Compressed Tokens:", compressed_tokens)
print("Compressed Ratio:", f"{original_tokens/(compressed_tokens + 1e-5):.2f}x")
response = synthesizer.synthesize(question, new_retrieved_nodes)
print(str(response))
retriever_query_engine = RetrieverQueryEngine.from_args(
    retriever, node_postprocessors=[node_postprocessor]
)
response = retriever_query_engine.query(question)
print(str(response))
