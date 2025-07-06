import os, torch
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from tqdm import tqdm

load_dotenv()

inference_api_key = os.getenv("HUGGINGFACE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = "langchain-doc-index"
batch_size = 300

assert inference_api_key, "HUGGINGFACE_API_KEY not set in .env"
assert pinecone_api_key, "PINECONE_API_KEY not set in .env"

# Option 1: Use Hugging Face Inference API (remote) — Not recommended for large datasets
# Use this if you only need to embed small batches or can't run models locally
# embeddings = HuggingFaceEndpointEmbeddings(
#     model="BAAI/bge-large-en-v1.5",
#     huggingfacehub_api_token=inference_api_key
# )

# Option 2: Run embeddings locally using HuggingFaceBgeEmbeddings — Recommended for large-scale ingestion
# This loads the model on your own machine (CPU or GPU) and is suitable for processing thousands of chunks
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

def ingest_docs():
    try:
        print("\nLoading raw documents...")
        loader = ReadTheDocsLoader(
            path="langchain-docs",
            encoding="utf-8" 
        )
        raw_documents = list(tqdm(loader.lazy_load(), desc="Loading documents"))
        print(f"Loaded {len(raw_documents)} raw documents.")

        print("\nSplitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        documents = text_splitter.split_documents(raw_documents)
        print(f"Split into {len(documents)} chunks.")

        print("\nUpdating metadata and source URLs...")
        for doc in tqdm(documents, desc="Processing metadata"):
            source_url = doc.metadata["source"]
            if source_url:
                new_url = source_url.replace("langchain-docs\\", "https://")
                doc.metadata.update({"source": new_url})

        print(f"\nUploading {len(documents)} chunks to Pinecone index: {pinecone_index_name} in batches of {batch_size}...")
        print(f"This may take a few minutes depending on document size...")
        
        for i in tqdm(range(0, len(documents), batch_size), desc="Uploading to Pinecone"):
            batch = documents[i:i + batch_size]
            PineconeVectorStore.from_documents(
                documents=batch,
                embedding=embeddings,
                index_name=pinecone_index_name
            )

        print("\nVector store upload complete")
    except Exception as e:
        print(f"Error during ingestion: {e}")


if __name__ == "__main__":
    ingest_docs()