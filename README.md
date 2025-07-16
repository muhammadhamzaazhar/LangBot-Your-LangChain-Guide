# LangChain Docs Helper Bot

An AI-powered assistant to help you explore **LangChain documentation** and its ecosystem. Built with **Streamlit**, **Hugging Face embeddings**, and **OpenRouter LLMs**, it retrieves relevant documentation from a **Pinecone vector database** and answers your queries interactively.

<br>

<p align="center">
  <img src="https://github.com/user-attachments/assets/26d99618-b28d-43e3-9d27-835eb62a318e" alt="App Screenshot"/>
</p>

## Features

- Retrieve answers from LangChain docs using semantic search.
- Context-aware responses with a history-aware retriever.
- Restricts answers to LangChain and related ecosystem.
- Streamlit web interface for interactive chat.

## Environment Variables

To run this project locally, you **must create a `.env` file** in the root directory with the following keys:

```bash
HUGGINGFACE_API_KEY
OPENROUTER_API_KEY
PINECONE_API_KEY

LANGCHAIN_TRACING              # (Optional)
LANGSMITH_ENDPOINT             # Required if tracing enabled
LANGCHAIN_API_KEY              # Required if tracing enabled
LANGCHAIN_PROJECT              # Required if tracing enabled
```

---

## How the Documentation Was Sourced

The underlying documentation for this assistant was recursively crawled from the official LangChain documentation website using

```bash
wget -A .html -r -P langchain-docs https://python.langchain.com/docs/introduction/
```

**Regarding the GitHub Repository:**

Please note that the GitHub repository for this project contains **a selection of the HTML files** used to train and inform this AI assistant, but **not the entirety of the crawled LangChain documentation**.

---

## Configuration Details

This project uses the following components in its AI pipeline:

- **Vector Store**:

  - [Pinecone](https://www.pinecone.io/) is used to store vector embeddings of the LangChain documentation for fast semantic search.

- **LLM Model**:

  - `mistralai/mistral-small-3.2-24b-instruct:free` is served via [OpenRouter](https://openrouter.ai/).
  - It is optimized for advanced reasoning, conversational interactions, retrieval-augmented generation (RAG).

- **Embeddings Model**:
  - [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) from Hugging Face is used to embed the documentation into high-dimensional vectors for similarity search.
    <br>
    > **Note**: During **document ingestion**, the embedding model is run **locally** to generate and store vector embeddings efficiently in Pinecone. During **inference (query time)**, the embedding model is accessed via **Hugging Face Inference API** to ensure lightweight and faster deployment of the assistant.

---

## Contributing

Contributions are welcome! If you find a bug or have suggestions for improvements, feel free to open an issue or submit a pull request.

---

## Contact

For any questions or feedback, reach out to **me** via [LinkedIn](https://www.linkedin.com/in/muhammad-hamza-azhar-996289314/) or open an issue on this repository.

---
