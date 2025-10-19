import streamlit as st
import os
import tempfile

# These are the correct, modern imports for the langchain library
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_core.documents import Document

# --- Page Configuration ---
st.set_page_config(
    page_title="Chat with Your PDF",
    page_icon="📄",
    layout="wide"
)

# --- PDF Processing Function ---
# This function is cached to avoid reprocessing the PDF on every interaction
@st.cache_resource
def process_pdf(uploaded_file):
    """Load, split, embed, and index a PDF file."""
    try:
        # Use a temporary file to handle the uploaded data securely
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        st.info(f"📄 Loading and processing '{uploaded_file.name}'...")

        # 1. Load the PDF document
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load_and_split()

        # 2. Split the document into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        documents = text_splitter.split_documents(pages)

        # 3. Create text embeddings
        st.info("Creating text embeddings and vector store... (This might take a moment)")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # 4. Create a Chroma vector store to index the embeddings
        vector_db = Chroma.from_documents(
            documents,
            embedding=embeddings
        )
        
        # 5. Clean up the temporary file
        os.unlink(tmp_file_path)

        st.success("✅ PDF processed successfully! Ready for questions.")
        return vector_db

    except Exception as e:
        st.error(f"An error occurred while processing the PDF: {e}")
        return None

# --- App UI ---
st.title("📄 Chat with Your PDF")
st.markdown("""
This app allows you to ask questions about any PDF document you upload. 
It uses an AI model to find and generate answers directly from the document's content.
""")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Settings")

    # API Key input
    api_key = st.text_input(
        "🔑 Enter your OpenRouter API Key",
        type="password",
        help="Get your key from https://openrouter.ai/"
    )
    
    # File uploader for the PDF
    uploaded_file = st.file_uploader(
        "Upload your PDF Report",
        type="pdf"
    )

    st.markdown("---")
    st.info("How to Use:\n1. Enter your API key.\n2. Upload a PDF.\n3. Ask questions!")

# --- Chat History Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! Please enter your API key and upload a PDF to begin chatting."
    }]

# --- Display Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Main Chat Functionality ---
# Only proceed if both API key and a file are provided
if api_key and uploaded_file:
    vector_db = process_pdf(uploaded_file)

    if vector_db:
        # Chat input box appears only after PDF is processed
        if prompt := st.chat_input("Ask a question about the document..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing the document..."):
                    try:
                        # Initialize memory and the QA chain
                        memory = ConversationBufferMemory(
                            memory_key="chat_history", return_messages=True
                        )
                        llm = ChatOpenAI(
                            model="openai/gpt-3.5-turbo",
                            temperature=0.2,
                            openai_api_base="https://openrouter.ai/api/v1",
                            max_tokens=700,
                            openai_api_key=api_key
                        )
                        qa_chain = ConversationalRetrievalChain.from_llm(
                            llm=llm,
                            retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
                            memory=memory
                        )

                        # Get the answer
                        result = qa_chain.invoke({"question": prompt})
                        response = result["answer"]

                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

                    except Exception as e:
                        st.error(f"⚠️ An error occurred: {e}")

# Handle cases where prerequisites are not met
elif not api_key and uploaded_file:
    st.warning("Please enter your OpenRouter API key to start chatting.")
elif api_key and not uploaded_file:
    st.info("Please upload a PDF to begin.")

