import streamlit as st
import os
import requests
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
API_KEY = "K5QN2Uig3Xd1ihGAVkCyrGN5MIWyBHjp"
API_URL = "https://api.mistral.ai/v1/chat/completions"

# Define file path
FILE_PATH = r"C:\Users\Vaishnavi\Super_Nova\output.txt"  # Change this as needed

# Load text file data
def load_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Process and store document in FAISS (only if not already stored)
def initialize_vectorstore():
    if "vectorstore" not in st.session_state:
        text = load_text_file(FILE_PATH)

        # Split text into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        doc_splits = text_splitter.create_documents([text])

        # Embed documents
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(doc_splits, embeddings)

        # Store in session state (to avoid reloading on every query)
        st.session_state.vectorstore = vectorstore
        st.session_state.embeddings = embeddings
        st.session_state.docs = doc_splits

        st.success("Text document loaded and stored in FAISS!")

# Query Mistral AI with context restriction
def query_mistral_with_context(user_prompt):
    # Retrieve relevant documents from FAISS
    query_embedding = st.session_state.embeddings.embed_query(user_prompt)
    similar_docs = st.session_state.vectorstore.similarity_search_by_vector(query_embedding, k=5)

    # Extract relevant text from retrieved documents
    context = "\n\n".join([doc.page_content for doc in similar_docs])

    if not context:
        return "I couldn't find relevant information in the document."

    # Construct the prompt with context restriction
    prompt = f"""
    You are an AI assistant that must answer questions **only** based on the provided context. 
    If the answer is not found in the context, respond with "I don't know based on the given document."

    Context:
    {context}

    Question: {user_prompt}
    """

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {"model": "open-mistral-7b", "messages": [{"role": "user", "content": prompt}]}
    
    response = requests.post(API_URL, headers=headers, json=data)

    if response.status_code == 200:
        return response.json().get("choices")[0].get("message").get("content")
    else:
        st.error(f"Error querying Mistral API: {response.text}")
        return None

# Initialize FAISS on first run
initialize_vectorstore()

# Streamlit UI
st.title("Text Document Query Assistant")

# Query Input
prompt = st.text_input("Enter your query for Mistral AI:")

if st.button("Query Mistral"):
    if prompt and "vectorstore" in st.session_state:
        with st.spinner("Querying Mistral AI..."):
            response = query_mistral_with_context(prompt)
            if response:
                st.success("Query completed!")
                st.write(response)
