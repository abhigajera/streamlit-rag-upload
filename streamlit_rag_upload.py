import streamlit as st
import os
import tempfile
from typing import List, Dict

# CORRECTED GOOGLE GENAI IMPORT: Use 'from google import genai'
from google import genai 
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

# --- Configuration & Client Initialization ---

# 1. Access the key securely from the Streamlit secrets
try:
    GEMEINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("KeyError: GEMINI_API_KEY not found in Streamlit Secrets. Please check your secrets.toml or Streamlit Cloud settings.")
    st.stop() # Stop execution if key is missing

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 2. Initialize the GenAI Client globally 
try:
    # CORRECTED: Initialize the Client object using the secure key
    client = genai.Client(api_key=GEMEINI_API_KEY)
except Exception as e:
    st.sidebar.error(f"Failed to initialize GenAI Client. Error: {e}. Check your API key value.")
    client = None

# --- Core RAG Functions ---

@st.cache_resource
def get_embeddings_model():
    """Load the HuggingFace embeddings model once for efficiency."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )

def process_file_and_create_db(
    uploaded_file: st.runtime.uploaded_file_manager.UploadedFile,
    chunk_size: int = 512,       # 🌟 UPDATED: Smaller chunk size for faster retrieval
    chunk_overlap: int = 100     # 🌟 UPDATED: Higher overlap to retain context in small chunks
) -> Chroma:
    """
    Processes the uploaded PDF, chunks the text, and creates a temporary Chroma vector store.
    Uses optimized RecursiveCharacterTextSplitter parameters.
    """
    # 1. Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name

    try:
        # 2. Load the PDF
        loader = PyPDFLoader(temp_file_path)
        docs: List[Document] = loader.load()

        # 3. Split text into chunks using the optimized parameters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        docs_chunks: List[Document] = text_splitter.split_documents(docs)

        # 4. Create and store vector embeddings (in-memory, temporary)
        embeddings_function = get_embeddings_model()
        
        vector_db: Chroma = Chroma.from_documents(
            documents=docs_chunks,
            embedding=embeddings_function
        )
        return vector_db
        
    finally:
        # Clean up the temporary file immediately
        os.unlink(temp_file_path)

def generate_rag_prompt(query: str, context: str) -> str:
    """Formats the RAG prompt for the LLM."""
    prompt = ('''you are a helpful and informative bot that answers questions using text from the reference context included below.
                  Be sure to respond in a complete sentence, being comprehensive and clear, including all relevant background information.
                  However, you are talking to a non-technical audience, so be sure to break down complicated concepts and
                  strike a friendly and approachable tone.
                  If the context is irrelevant to the answer, you may ignore it.
                  QUESTION:'{query}'
                  CONTEXT:"{context}"
                  ANSWER:
               ''').format(query=query, context=context)
    return prompt

def generate_answer(prompt: str, genai_client: genai.Client) -> str:
    """Generates an answer using the Gemini API."""
    try:
        # CORRECTED: Use client.models.generate_content (preferred method)
        response = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"An error occurred while generating the answer: {e}"

def get_relevant_context_from_db(query: str, vector_db: Chroma) -> str:
    """Retrieves relevant text chunks from the vector database."""
    context = ""
    # Retrieve top 3 documents
    search_results = vector_db.similarity_search(query, k=3) 
    for result in search_results:
        context += result.page_content + "\n"
    return context 

# --- Streamlit UI and Logic ---

def main(genai_client: genai.Client):
    st.set_page_config(page_title="RAG Chatbot with File Upload 📄")
    st.title("RAG Chatbot with File Upload 📄")
    
    if genai_client is None:
        st.error("Cannot run the app. Gemini Client failed to initialize. Please fix your API key.")
        return

    # 1. SIDEBAR FOR FILE UPLOAD AND CHUNKING PARAMETERS
    with st.sidebar:
        st.header("Document & Chunking Settings ⚙️")
        
        # 🌟 NEW: Dynamic Inputs for R&D/Optimization
        chunk_size = st.number_input(
            "Chunk Size (Chars) - Smaller is Faster for Retrieval", 
            min_value=100, max_value=2000, value=512, step=50,
            help="The maximum size of text chunks. 512 is a good balance for speed and precision."
        )
        chunk_overlap = st.number_input(
            "Chunk Overlap (Chars) - Important for Context", 
            min_value=0, max_value=300, value=100, step=10,
            help="The overlap between consecutive chunks. Higher overlap prevents splitting key ideas."
        )
        
        st.subheader("Upload Document")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file to chat with:",
            type=["pdf"],
            help="Upload your document here. Processing may take a moment."
        )
        
        if st.button("Process Document with New Settings"):
            if uploaded_file is not None:
                # Clear previous chat history for the new document
                st.session_state["messages"] = [] 
                
                with st.spinner(f"Processing PDF (Size: {chunk_size}, Overlap: {chunk_overlap})..."):
                    # Pass the dynamic parameters to the processing function
                    vector_db = process_file_and_create_db(
                        uploaded_file, 
                        chunk_size=chunk_size, 
                        chunk_overlap=chunk_overlap
                    )
                    
                    # Store the vector DB in session state
                    st.session_state["vector_db"] = vector_db
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": f"✅ Document **{uploaded_file.name}** processed with **Chunk Size: {chunk_size}** and **Overlap: {chunk_overlap}**. You can now ask questions about its content."}
                    )
            else:
                st.sidebar.warning("Please upload a PDF file first!")

    # 2. MAIN CHAT AREA
    
    # Initialize chat history and vector_db placeholder if not present
    if "messages" not in st.session_state:
        welcome_text = generate_answer("Can You Quickly Introduce Yourself in a friendly and helpful way", genai_client)
        st.session_state["messages"] = [
            {"role": "assistant", "content": welcome_text}
        ]
    if "vector_db" not in st.session_state:
           st.session_state["vector_db"] = None

    # Check if a vector DB is ready for conversation
    db_ready = st.session_state["vector_db"] is not None

    # Display chat messages from history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Handle user input only if a document is processed (db_ready is True)
    if prompt := st.chat_input("Ask a question about the processed document...", disabled=not db_ready):
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Get the vector DB from the session state
        vector_db = st.session_state["vector_db"]

        # Generate the assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # 1. Retrieve Context
                context = get_relevant_context_from_db(prompt, vector_db)
                
                # 2. Generate Prompt
                rag_prompt = generate_rag_prompt(query=prompt, context=context)
                
                # 3. Get Answer
                answer = generate_answer(rag_prompt, genai_client) 
                st.write(answer)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    # Pass the globally initialized client to the main function
    main(client)