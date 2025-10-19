import streamlit as st
import os
import langchain
import sys
st.write("LangChain version:", langchain.__version__)

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document

# --- Page Configuration ---
st.set_page_config(
    page_title="Chat with Reliance Annual Report 2024–25",
    page_icon="📊",
    layout="wide"
)

# --- File Path ---
PDF_PATH = "D:/SEM-VII/NLP/Assigment/RIL-IAR-2025.pdf"

# --- PDF Processing Function ---
@st.cache_resource
def process_pdf(file_path):
    """Load, split, embed, and index the Reliance Annual Report PDF."""
    try:
        st.info("📄 Loading and processing the Reliance Annual Report...")

        # Load PDF
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()

        # Split into chunks
        pdf_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        docs = pdf_splitter.split_documents(pages)
        documents = [Document(page_content=doc.page_content) for doc in docs]

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # Create vector store
        vector_db = Chroma.from_documents(
            documents,
            embedding=embeddings
        )

        st.success("✅ PDF processed successfully! Ready for questions.")
        return vector_db

    except Exception as e:
        st.error(f"Error while processing PDF: {e}")
        return None


# --- App UI ---
st.title("📊 Chat with Reliance Annual Report 2024–25")
st.markdown("""
Ask questions about Reliance Industries Limited’s **Integrated Annual Report (FY 2024–25)**.  
The app uses OpenRouter’s GPT-3.5 model to generate answers directly from the PDF content.
""")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Settings")

    api_key = st.text_input(
        "🔑 Enter your OpenRouter API Key",
        type="password",
        help="Get your free key from https://openrouter.ai/"
    )

    st.markdown("---")
    st.markdown(f"**Using File:** `{os.path.basename(PDF_PATH)}`")

    if not os.path.exists(PDF_PATH):
        st.error("❌ Reliance PDF not found. Please check the path.")
    else:
        st.success("✅ Reliance PDF found and ready.")

    st.markdown("---")
    st.markdown("""
    **How to Use:**
    1. Enter your OpenRouter API key.
    2. Ask questions about the annual report.
    3. Type naturally — I’ll extract answers from the report!
    """)

# --- Chat History Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello 👋! Ask me anything about Reliance Industries’ Integrated Annual Report 2024–25."
    }]

# --- Display Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Main Chat Functionality ---
if api_key and os.path.exists(PDF_PATH):
    vector_db = process_pdf(PDF_PATH)

    if vector_db:
        if prompt := st.chat_input("Ask a question about the Reliance Annual Report..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing the report..."):
                    try:
                        # Set up memory and model
                        memory = ConversationBufferMemory(
                            memory_key="chat_history",
                            return_messages=True
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
                            retriever=vector_db.as_retriever(search_kwargs={"k": 4}),
                            memory=memory
                        )

                        result = qa_chain({"question": prompt})
                        response = result["answer"]

                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

                    except Exception as e:
                        st.error(f"⚠️ Error: {e}")
else:
    st.warning("Please enter your OpenRouter API key to start chatting.")


