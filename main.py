#
# Copyright IBM Corp. 2024
# SPDX-License-Identifier: Apache-2.0
#

import os

# llama-parse is async-first, running the async code in a notebook requires the use of nest_asyncio
import nest_asyncio

nest_asyncio.apply()

from genai import Client, Credentials
from genai.extensions.langchain import LangChainEmbeddingsInterface
from genai.schema import TextEmbeddingParameters
from genai.extensions.langchain.chat_llm import LangChainChatInterface
from genai.schema import (
    DecodingMethod,
    ModerationHAP,
    ModerationParameters,
    TextGenerationParameters,
    TextGenerationReturnOptions,
)
from llama_index.core import VectorStoreIndex
from llama_parse import LlamaParse
from llama_index.core.retrievers import RecursiveRetriever
from MarkdownElementNodeParser import MarkdownElementNodeParser

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext


def create_llm(gen_ai_key):
    credentials = Credentials(
        api_key=gen_ai_key,
        api_endpoint="https://bam-api.res.ibm.com/v2/text/chat?version=2024-01-10",
    )
    client = Client(credentials=credentials)
    llm = LangChainChatInterface(
        model_id="meta-llama/llama-2-70b-chat",
        client=client,
        parameters=TextGenerationParameters(
            decoding_method=DecodingMethod.SAMPLE,
            max_new_tokens=200,
            min_new_tokens=10,
            temperature=0.5,
            top_k=50,
            top_p=1,
            return_options=TextGenerationReturnOptions(
                input_text=False, input_tokens=True
            ),
        ),
        moderations=ModerationParameters(
            # Threshold is set to very low level to flag everything (testing purposes)
            # or set to True to enable HAP with default settings
            hap=ModerationHAP(input=True, output=False, threshold=0.01)
        ),
    )
    return llm


def create_embeddings(gen_ai_key):
    credentials = Credentials(api_key=gen_ai_key)
    client = Client(credentials=credentials)
    embeddings = LangChainEmbeddingsInterface(
        client=client,
        model_id="sentence-transformers/all-minilm-l6-v2",
        parameters=TextEmbeddingParameters(truncate_input_tokens=True),
    )
    return embeddings


def parse_pdf(file_path, llama_cloud_api_key):
    parser = LlamaParse(
        api_key=llama_cloud_api_key,
        result_type="markdown",
        verbose=True,
        language="en",
        num_workers=2,
    )
    documents = parser.load_data(file_path)
    return documents


def build_recursive_index(documents):
    node_parser = MarkdownElementNodeParser(llm=None, num_workers=8)

    nodes = node_parser.get_nodes_from_documents(documents)

    base_nodes, objects = node_parser.get_nodes_and_objects(nodes)

    base_nodes, node_mappings = node_parser.get_base_nodes_and_mappings(nodes)

    # chroma db as vector store
    db = chromadb.PersistentClient(path="./chroma_db")

    chroma_collection = db.get_or_create_collection("my_collection")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(nodes=base_nodes, storage_context=storage_context)
    vector_ret = index.as_retriever(similarity_top_k=1)

    recursive_index = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": vector_ret},
        node_dict=node_mappings,
        verbose=False,
    )

    return recursive_index
