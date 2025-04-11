from langchain_huggingface import HuggingFaceEmbeddings # type: ignore
from langchain.vectorstores import FAISS # type: ignore
import pickle

# Load preprocessed text chunks
with open("handbook.txt", "r", encoding="utf-8") as f:
    text = f.read().split("\n\n")  # Splitting by paragraph

# Convert text into numerical embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_texts(text, embedding_model)

# Save FAISS index for future use
vector_store.save_local("faiss_index")
print("Embeddings successfully created and stored in FAISS.") 