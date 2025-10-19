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
    page_title="Chat with RIL Annual Report",
    page_icon="📄",
    layout="wide"
)

# --- Hardcoded File Path ---
# The app will look for this specific PDF file.
PDF_PATH = "RIL-IAR-2025.pdf"

# --- PDF Processing Function ---
# This function is cached to avoid reprocessing the PDF on every app run.
@st.cache_resource
def process_pdf(file_path):
    """Load, split, embed, and index the specified PDF file."""
    try:
        st.info(f"📄 Loading and processing '{os.path.basename(file_path)}'...")

        # 1. Load the PDF document
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()

        # 2. Split the document into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        documents = text_splitter.split_documents(pages)

        # 3. Create text embeddings
        st.info("Creating text embeddings... (This may take a moment on first run)")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # 4. Create a Chroma vector store to index the embeddings
        vector_db = Chroma.from_documents(
            documents,
            embedding=embeddings
        )
        
        st.success("✅ Report processed successfully! Ready for your questions.")
        return vector_db

    except Exception as e:
        st.error(f"An error occurred while processing the PDF: {e}")
        return None

# --- App UI ---
st.title("📊 Chat with the Reliance Annual Report 2024-25")
st.markdown("Ask questions about Reliance Industries Limited’s latest Integrated Annual Report.")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Settings")

    # API Key input
    api_key_input = st.text_input(
        "🔑 Enter your OpenRouter API Key",
        type="password",
        help="Get your key from https://openrouter.ai/"
    )
    
    st.markdown("---")
    
    # Check if the hardcoded PDF exists
    if not os.path.exists(PDF_PATH):
        st.error(f"❌ '{PDF_PATH}' not found!")
        st.error("Please make sure the PDF file is in the same directory as the app.")
    else:
        st.success(f"✅ Found '{os.path.basename(PDF_PATH)}'.")
    
    st.markdown("---")
    st.info("How to Use:\n1. Enter your API key.\n2. The RIL report is loaded automatically.\n3. Ask questions in the chat!")

# --- Chat History Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! Please enter your API key to begin asking questions about the RIL Annual Report."
    }]

# --- Display Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Main Chat Functionality ---
# Only proceed if both API key and the PDF file exist
if api_key_input and os.path.exists(PDF_PATH):
    vector_db = process_pdf(PDF_PATH)

    if vector_db:
        # Chat input box appears only after PDF is processed
        if prompt := st.chat_input("Ask a question about the RIL report..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Searching the report and generating an answer..."):
                    try:
                        # Initialize memory and the QA chain if not already in session state
                        if 'qa_chain' not in st.session_state:
                            memory = ConversationBufferMemory(
                                memory_key="chat_history", return_messages=True
                            )
                            # CORRECTED INITIALIZATION: Using modern 'base_url' and 'api_key' parameters
                            llm = ChatOpenAI(
                                model="openai/gpt-3.5-turbo",
                                temperature=0.2,
                                base_url="https://openrouter.ai/api/v1",
                                max_tokens=700,
                                api_key=api_key_input
                            )
                            st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
                                llm=llm,
                                retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
                                memory=memory
                            )

                        # Get the answer
                        result = st.session_state.qa_chain.invoke({"question": prompt})
                        response = result["answer"]

                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

                    except Exception as e:
                        st.error(f"⚠️ An error occurred while getting the answer: {e}")

# Handle cases where prerequisites are not met
elif not api_key_input:
    st.warning("Please enter your OpenRouter API key in the sidebar to start the chat.")

