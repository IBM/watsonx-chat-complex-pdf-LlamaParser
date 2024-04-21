# Chat with Complex PDF Using IBM WatsonX, LlamaParser, Langchain and ChromaDB 📚 (Using LlamaParse for RAG Applications with LlamaIndex)

🚀 Chat seamlessly with complex PDF (with texts and tables) using IBM WatsonX, LlamaParser, Langchain &amp; ChromaDB Vector DB with Seamless Streamlit Deployment. Get instant, Accurate responses from Awesome IBM WatsonX Language Model. 📚💬 Transform your PDF experience now! 🔥✨

## 📝 Description

This PDF Chat Agent is a Streamlit-based web application designed to facilitate interactive conversations with a chatbot. The app allows users to upload complex PDF document(with text and complex tables), extract markdown information from them, and train a chatbot using this extracted content. Users can then engage in real-time conversations with the chatbot.

## 📢Run App with Streamlit Cloud

[Launch App On Streamlit](https://chat-complx-pdf-watsonx.streamlit.app/)

## 🎯 How It Works:

---

The application follows these steps to create supirior RAG pipeline to provide responses to your questions:

1. **PDF Loading and Parsing** : The app reads PDF document and parse it to markdown using LlamaParse. `LlamaParse` is an API created by `LlamaIndex` to efficiently parse and represent files for efficient retrieval and context augmentation using LlamaIndex frameworks. It seamlessly integrates with the advanced indexing/retrieval capabilities that the open-source framework offers, enabling users to build state-of-the-art document RAG.

2. **Create VectorStoreIndex** : Extract `base_nodes` and `objects` to create `VectorStoreIndex`

3. **Build Recursive Query Engine** : Build recursive retrieval/query engine using `MarkdownElementNodeParser` for parsing the LlamaParse output markdown results. It does this in 2 steps,

   - Initalize our reranker using `FlagEmbeddingReranker` powered by the `BAAI/bge-reranker-large`. We'll be leveraging [this](https://github.com/FlagOpen/FlagEmbedding) repo to leverage our `BAAI/bge-reranker-large`.
   - Set up our recursive query engine!

4. **Response Generation** : Generate reponse for the user prompt by querying the vector index using query engine.

---

## 🎯 Key Features

- **Built with IBM WatsonX API and Langchain Extension**: Yes! you heard it right.. this app is build using `IBM WatsonX Generative AI API` and `Langchain` extension to create LLM and Embeddings. There is no OOTB support for IBM WatsonX API in LlamaParse. This app is developed to leverage parsing capabilities of LlamaParse with IBM WatsonX.

- **Supports LlamaIndex V0.10**: What is unique and special about this?? IBM WatsonX LlamaIndex extension still uses `LlamaIndex V0.9`. `LlamaIndex V0.10` has several major changes, and one of the key changes is the replacement of `ServiceContext` with `Settings`. If you try to use LlamaParse (which uses LlamaIndex V0.10) along with IBM WatsonX LlamaIndex extension, it will not work.

- **Parsing PDF to Markdown using IBM WatsonX LLM**: This is an important. LlamaIndex `MarkdownElementNodeParser` uses `OPENAI Pydentic` under the hood. So it was breaking for some of the tables when using WatsonX LLM. To fix this, I modified the `BaseElementNodeParser`so that I can pass LLM as None and disable table summary option. Creating sumamry takes lot of time and it breakes for some table when using WatsonX LLM.

- **ChromaDB as Vector Store**: It uses `ChromaDB` as VectorStore to persist the data.

## ▶️Installation

Clone the repository:

`git clone https://github.ibm.com/kirtijha/watsonx-chat-with-complex-pdf-LlamaParse.git`

Install the required Python packages:

`pip install -r requirements.txt`

Set up your WatsonX API key from `https://bam.res.ibm.com/`

Set up your LlamaCloud API key from `https://cloud.llamaindex.ai/`

Run the Streamlit app:

`streamlit run chatbot.py`

---

## ©️ License 🪪

Distributed under the MIT License. See `LICENSE` for more information.

---
