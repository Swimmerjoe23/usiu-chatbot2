from dotenv import load_dotenv
load_dotenv()

import os
import pdfplumber
import gradio as gr
import time
from typing import List
from pydantic import Field

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PDFPlumberLoader, PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.llms import Ollama
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI

# Change to the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

# Load and extract text
pdf_path = "266ac45d-student-handbook.pdf"
text = extract_text_from_pdf(pdf_path)

# Save the extracted text to handbook.txt
with open("handbook.txt", "w", encoding="utf-8") as file:
    file.write(text)
print(f"Text extracted and saved to handbook.txt")

# Set Google API Key (replace with your actual key)
import os
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Convert text into numerical embeddings
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# Update the LLM initialization with temperature
llm = GoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY
)

# Combined Retriever Class that inherits from BaseRetriever
class CombinedRetriever(BaseRetriever):
    base_retriever: BaseRetriever = Field(description="Base retriever for trained data")
    document_store: Chroma = Field(default_factory=lambda: Chroma(
        embedding_function=embedding_model,
        collection_name="uploaded_docs"
    ))

    class Config:
        arbitrary_types_allowed = True

    def add_documents(self, texts: List[Document]) -> None:
        """Add documents to the document store."""
        if texts:
            self.document_store.add_documents(texts)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents from both sources."""
        results = []
        
        # Get results from base retriever (your trained data)
        if self.base_retriever:
            try:
                base_results = self.base_retriever.get_relevant_documents(query)
                results.extend(base_results)
            except Exception as e:
                print(f"Error getting base retriever results: {e}")

        # Get results from uploaded documents
        try:
            upload_results = self.document_store.similarity_search(query, k=3)
            results.extend(upload_results)
        except Exception as e:
            print(f"Error getting document store results: {e}")

        return results

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version of get_relevant_documents."""
        return self._get_relevant_documents(query)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents([Document(page_content=text)])
# Create FAISS vector store
vectorstore = FAISS.from_documents(split_docs, embedding_model)

# Define the retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

def create_enhanced_prompt(query, relevant_docs):
    """Generate a response prompt without explicitly mentioning retrieved documents."""
    if relevant_docs:
        doc_texts = "\n".join([doc.page_content for doc in relevant_docs])
        prompt = f"""Answer the following question comprehensively, integrating relevant information seamlessly.
        
        Context:
        {doc_texts}
        
        Question: {query}
        
        Provide a clear, natural, and well-rounded response. Avoid explicitly mentioning the source of knowledge; instead, focus on delivering a useful answer in a conversational tone."""
    else:
        prompt = f"""Question: {query}
        
        Provide a comprehensive and natural response based on your knowledge.
        Be conversational and thorough in your explanation."""
    
    return prompt

# Initialize combined retriever with your existing retriever
combined_retriever = CombinedRetriever(base_retriever=retriever)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"  # Ensure only 'answer' is stored in memory
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=combined_retriever,
    memory=memory,
    verbose=True,
    chain_type="stuff",
    return_source_documents=True,
    return_generated_question=True,
    output_key="answer"  # Store only the answer in memory
)

def chatbot_response(user_input, chat_history):
    if chat_history is None:
        chat_history = []

    try:
        # Step 1: Retrieve relevant documents
        relevant_docs = combined_retriever._get_relevant_documents(user_input)
        doc_texts = "\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else ""

        # Step 2: Construct conversation context
        context = "\n".join([f"{role}: {msg}" for role, msg in chat_history[-8:]])

        # Step 3: Choose response strategy
        if relevant_docs:
            prompt = create_enhanced_prompt(user_input, relevant_docs)
        else:
            prompt = user_input
        
        response = llm.invoke(prompt)

        # Step 4: Ensure a detailed response
        if len(str(response)) < 100:
            response = llm.invoke(f"Can you elaborate more? {response}")

        response = str(response).strip()

    except Exception as e:
        response = f"Something went wrong: {e}"

    # Step 5: Update chat history
    chat_history.append(("You", user_input))
    chat_history.append(("AI", response))
    
    time.sleep(1)
    return chat_history, ""