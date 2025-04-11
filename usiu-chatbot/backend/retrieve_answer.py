from langchain_community.vectorstores import FAISS  # Import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Import embedding model
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import os
vector_store = FAISS.load_local(
    "faiss_index",
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    allow_dangerous_deserialization=True  # Make sure you're loading trusted data
)
os.environ["OPENAI_API_KEY"] = "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"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress tokenizers warning
llm = ChatOpenAI(model_name="gpt-4")  # You can also use "gpt-3.5-turbo"
retriever = vector_store.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
def get_answer(query):
    """Fetches the best answer for a given query using RAG."""
    response = qa_chain.invoke({"query": query})  # Use invoke instead of run
    return response["result"]  # Ensure correct extraction of response
if __name__ == "__main__":
    while True:
        query = input("Ask a question: ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = get_answer(query)
        print("\n🤖 Chatbot: ", answer, "\n")
