#
# Copyright IBM Corp. 2024
# SPDX-License-Identifier: Apache-2.0
#

# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import tempfile, os

from llama_index.llms.langchain import LangChainLLM
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from main import create_llm, create_embeddings, parse_pdf, build_recursive_index

st.set_page_config(page_title=":books: IBM Watsonx Complex PDF Chatbot")


@st.cache_resource(show_spinner=False)
def load_data(file_path, gen_ai_key, llama_cloud_api_key):
    with st.spinner(
        text="Loading and indexing the document - hang tight! This should take 1-2 minutes."
    ):
        documents = parse_pdf(file_path, llama_cloud_api_key)
        llm = create_llm(gen_ai_key=gen_ai_key)
        embeddings = create_embeddings(gen_ai_key=gen_ai_key)

        llm = LangChainLLM(llm=llm)
        embed_model = LangchainEmbedding(embeddings)

        Settings.llm = llm
        Settings.embed_model = embed_model

        recursive_index = build_recursive_index(documents)

        return recursive_index


def handle_user_input(user_question, recursive_index):

    reranker = FlagEmbeddingReranker(
        top_n=5,
        model="BAAI/bge-reranker-large",
    )

    recursive_query_engine = RetrieverQueryEngine.from_args(
        recursive_index, node_postprocessors=[reranker], verbose=False
    )
    response = recursive_query_engine.query(user_question)
    return response


def main():
    with st.sidebar:
        st.title(
            ":books: IBM WatsonX Complex PDF Chatbot using LlamaParse and Langchain"
        )
        st.write(
            "This chatbot is created using IBM WatsonX and LlamaParse to chat with PDF files with complex tables."
        )

        gen_ai_key = st.text_input(
            "Please enter your IBM Watsonx API key", key="gen_ai_key", type="password"
        )
        llama_cloud_api_key = st.text_input(
            "Please enter your Llama Cloud API key",
            key="llama_cloud_api_key",
            type="password",
        )

        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDF here and click on 'Process")

        if st.button("Process"):
            temp_dir = tempfile.mkdtemp()
            path = os.path.join(temp_dir, pdf_docs.name)
            with open(path, "wb") as f:
                f.write(pdf_docs.getvalue())
            # get recursive index
            recursive_index = load_data(path, gen_ai_key, llama_cloud_api_key)

            st.session_state.conversation = recursive_index

            st.success(
                "Document is successfully uploaded. You can ask questions!", icon="✅"
            )

    st.title("Chat with complex PDF using LlamaParse :books:")

    # initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "conversation" not in st.session_state:
        st.session_state.conversation = {}

    # display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    def clear_chat_history():
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask a question about your document"}
        ]

    st.sidebar.button("Clear Chat History", on_click=clear_chat_history)

    # accept user input
    if not gen_ai_key or not llama_cloud_api_key or not st.session_state.conversation:
        st.info("Please add your API keys to continue.")
        st.stop()
    if prompt := st.chat_input("Ask a question about your document"):
        # add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.spinner("Thinking...."):
            # display assistance response in chat message container
            response = handle_user_input(prompt, st.session_state.conversation)
            with st.chat_message("assistant"):
                st.markdown(response)
            # add assistance response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
