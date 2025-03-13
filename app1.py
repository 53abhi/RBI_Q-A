import streamlit as st
import os
import requests
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY", "K5QN2Uig3Xd1ihGAVkCyrGN5MIWyBHjp")
API_URL = "https://api.mistral.ai/v1/chat/completions"

# Streamlit UI
st.title("Text Document Query Assistant")

# File uploader instead of a fixed file path
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

# Load text file data
def load_text_file(file):
    return file.read().decode("utf-8") if file else None

# Process and store document in FAISS
def initialize_vectorstore(text):
    if "vectorstore" not in st.session_state and text:
        # Split text into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        doc_splits = text_splitter.create_documents([text])

        # Embed documents
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(doc_splits, embeddings)

        # Store in session state
        st.session_state.vectorstore = vectorstore
        st.session_state.embeddings = embeddings
        st.session_state.docs = doc_splits
        st.success("Text document loaded and stored in FAISS!")

# Query Mistral AI with context restriction
def query_mistral_with_context(user_prompt):
    if "vectorstore" not in st.session_state:
        return "No document has been uploaded yet."

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

# Load and process document when uploaded
if uploaded_file:
    text_content = load_text_file(uploaded_file)
    if text_content:
        initialize_vectorstore(text_content)
    else:
        st.error("Uploaded file is empty or could not be read.")

# Query Input
prompt = st.text_input("Enter your query for Mistral AI:")

if st.button("Query Mistral"):
    if prompt:
        with st.spinner("Querying Mistral AI..."):
            response = query_mistral_with_context(prompt)
            if response:
                st.success("Query completed!")
                st.write(response)
    else:
        st.warning("Please enter a query.")
