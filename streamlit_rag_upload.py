import streamlit as st
from google import genai# Assuming this is the module name for the SDK
import os
import tempfile
from typing import List, Dict

# LangChain Imports for Processing
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

# --- Configuration ---
# WARNING: Replace this with st.secrets for a real deployment!
GEMEINI_API_KEY = st.secrets["GEMINI_API_KEY"]
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

try:
    genai.configure(api_key=GEMEINI_API_KEY)
except Exception as e:
    st.sidebar.error(f"Failed to configure Gemini API. Please check your key. Error: {e}")

# --- Core RAG Functions ---

@st.cache_resource
def get_embeddings_model():
    """Load the HuggingFace embeddings model once for efficiency."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )

def process_file_and_create_db(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> Chroma:
    """
    Processes the uploaded PDF, chunks the text, and creates a temporary Chroma vector store.
    """
    # 1. Save the uploaded file to a temporary location
    # Streamlit file uploader provides file data, which needs to be saved for PyPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name

    try:
        # 2. Load the PDF using the temporary path
        loader = PyPDFLoader(temp_file_path)
        docs: List[Document] = loader.load()

        # 3. Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
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
    # Your original comprehensive and approachable prompt
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

def get_relevant_context_from_db(query: str, vector_db: Chroma) -> str:
    """Retrieves relevant text chunks from the vector database."""
    context = ""
    # Perform similarity search using the session's vector_db
    search_results = vector_db.similarity_search(query, k=3)
    for result in search_results:
        context += result.page_content + "\n"
    return context 

def generate_answer(prompt: str) -> str:
    """Generates an answer using the Gemini API."""
    try:
        model = genai.GenerativeModel(model_name="models/gemini-2.5-flash") 
        answer = model.generate_content(prompt)
        return answer.text
    except Exception as e:
        return f"An error occurred while generating the answer: {e}"

# --- Streamlit UI and Logic ---

def main():
    st.set_page_config(page_title="RAG Chatbot with File Upload 📄")
    st.title("RAG Chatbot with File Upload 📄")
    
    # 1. SIDEBAR FOR FILE UPLOAD
    with st.sidebar:
        st.header("Upload Document")
        
        # The st.file_uploader widget
        uploaded_file = st.file_uploader(
            "Choose a PDF file to chat with:",
            type=["pdf"],
            help="Upload your document here. Processing may take a moment."
        )
        

        # Use a button to trigger the processing
        if st.button("Process Document"):
            if uploaded_file is not None:
                # Clear previous chat history for the new document
                st.session_state["messages"] = [] 
                
                with st.spinner("Processing PDF and creating knowledge base..."):
                    vector_db = process_file_and_create_db(uploaded_file)
                    
                    # Store the vector DB in session state
                    st.session_state["vector_db"] = vector_db
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": f"✅ Document **{uploaded_file.name}** processed! You can now ask questions about its content."}
                    )
            else:
                st.sidebar.warning("Please upload a PDF file first!")

    # 2. MAIN CHAT AREA
    
    # Initialize chat history and vector_db placeholder if not present
    if "messages" not in st.session_state:
        welcome_text = generate_answer("Can You Quickly Introduce Yourself in a friendly and helpful way")
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
            with st.spinner("Thinking... Retrieving context and generating answer..."):
                # 1. Retrieve Context
                context = get_relevant_context_from_db(prompt, vector_db)
                
                # 2. Generate Prompt
                rag_prompt = generate_rag_prompt(query=prompt, context=context)
                
                # 3. Get Answer
                answer = generate_answer(rag_prompt)
                st.write(answer)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()