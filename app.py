import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
# CORRECTED IMPORT: This module is now in its own package
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
# CORRECTED IMPORT: This module is now in its own package
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
# CORRECTED IMPORT: This module is now in its own package
from langchain_core.documents import Document
import tempfile

# --- Page Configuration ---
st.set_page_config(
    page_title="Chat with Reliance Annual Report 2024–25",
    page_icon="📊",
    layout="wide"
)

# --- PDF Processing Function ---
@st.cache_resource
def process_pdf(uploaded_file):
    """Load, split, embed, and index a PDF file."""
    try:
        # Use a temporary file to handle the uploaded data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        st.info(f"📄 Loading and processing '{uploaded_file.name}'...")

        # Load PDF
        loader = PyPDFLoader(tmp_file_path)
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
        
        # Clean up the temporary file
        os.unlink(tmp_file_path)

        st.success("✅ PDF processed successfully! Ready for questions.")
        return vector_db

    except Exception as e:
        st.error(f"Error while processing PDF: {e}")
        return None


# --- App UI ---
st.title("📊 Chat with Reliance Annual Report 2024–25")
st.markdown("""
Ask questions about any uploaded PDF, such as the **Reliance Industries Limited Integrated Annual Report**.  
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
    
    # CORRECTED LOGIC: Use a file uploader instead of a hardcoded path
    uploaded_file = st.file_uploader(
        "Upload your PDF Report",
        type="pdf"
    )

    st.markdown("---")
    st.markdown("""
    **How to Use:**
    1. Enter your OpenRouter API key.
    2. Upload the PDF you want to chat with.
    3. Ask questions about the report!
    """)

# --- Chat History Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello 👋! Please enter your API key and upload a PDF to begin."
    }]

# --- Display Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Main Chat Functionality ---
if api_key and uploaded_file:
    vector_db = process_pdf(uploaded_file)

    if vector_db:
        if prompt := st.chat_input("Ask a question about the report..."):
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

                        result = qa_chain.invoke({"question": prompt})
                        response = result["answer"]

                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

                    except Exception as e:
                        st.error(f"⚠️ Error: {e}")
elif not api_key and uploaded_file:
    st.warning("Please enter your OpenRouter API key to start chatting.")
elif api_key and not uploaded_file:
    st.info("Please upload a PDF to begin.")

