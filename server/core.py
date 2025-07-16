import os, torch
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain import hub
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    ChatPromptTemplate,
)
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

load_dotenv()

inference_api_key = os.getenv("HUGGINGFACE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
pinecone_index_name = "langchain-doc-index"

assert inference_api_key, "HUGGINGFACE_API_KEY not set in .env"
assert pinecone_api_key, "PINECONE_API_KEY not set in .env"
assert openrouter_api_key, "OPENROUTER_API_KEY not set in .env"

def get_retrieval_chain(query: str, chat_history: List[Dict[str, Any]]) -> dict:
    print("\nLoading BGE embeddings model...")
    try:
        embeddings = HuggingFaceEndpointEmbeddings(
            model="BAAI/bge-large-en-v1.5",
            huggingfacehub_api_token=inference_api_key
        )
        # embeddings = HuggingFaceEmbeddings(
        #     model_name="BAAI/bge-large-en-v1.5",
        #     model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        #     encode_kwargs={"normalize_embeddings": True}
        # )
    except Exception as e:
        print(f"Failed to load embeddings: {e}")
        raise

    print(f"Connecting to Pinecone index: {pinecone_index_name}...")
    try:
        docsearch = PineconeVectorStore(index_name=pinecone_index_name, embedding=embeddings)
    except Exception as e:
        print(f"Failed to connect to Pinecone: {e}")
        raise

    print("Loading LLM from Openrouter...")
    try:
        llm = ChatOpenAI(
            model="nvidia/llama-3.1-nemotron-ultra-253b-v1:free",  
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
            streaming=True,
            temperature=0.3,
        )
    except Exception as e:
        print(f"Failed to load LLM: {e}")
        raise

    print("Pulling prompts from LangChain hub...")
    try:
        # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        custom_system_prompt = (
            "You are an expert assistant for LangChain documentation and its ecosystem. "
            "Answer the user's question **only if it is directly related** to LangChain, its ecosystem, or related tools and libraries."
            "If the question is not related, respond:\n"
            "'❌ I can only answer questions about LangChain and its ecosystem.'\n\n"
            "Never attempt to answer questions outside this scope, even if you know the answer.\n"
            "Use ONLY the <context> below to formulate your response.\n\n"
            "<context>\n{context}\n</context>"
        )
        retrieval_qa_chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(custom_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),  
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    except Exception as e:
        print(f"Failed to pull prompts: {e}")
        raise

    print("Creating retriever and chain...")
    try:
        stuff_documents_chain = create_stuff_documents_chain(llm=llm, prompt=retrieval_qa_chat_prompt)
        history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=docsearch.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 6, "score_threshold": 0.8}
            ),
            prompt=rephrase_prompt,
        )
        # retrieval_chain = create_retrieval_chain(
        #     retriever=docsearch.as_retriever(
        #         search_type="similarity_score_threshold",
        #         search_kwargs={"k": 5, "score_threshold": 0.8}
        #     ),
        #     combine_docs_chain=stuff_documents_chain
        # )
        retrieval_chain = create_retrieval_chain(
            retriever=history_aware_retriever,
            combine_docs_chain=stuff_documents_chain
        )
    except Exception as e:
        print(f"Failed to create retrieval chain: {e}")
        raise

    print("Running retrieval chain on query...")
    try:
        return retrieval_chain.stream({"input": query, "chat_history": chat_history})
    except Exception as e:
        print(f"Error during retrieval chain execution: {e}")
        raise


if __name__ == "__main__":
    response = get_retrieval_chain(query = "What is LangChain?")
    print(response["result"])