import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
import tempfile

# --- Page Configuration ---
st.set_page_config(
    page_title="Chat with any PDF",
    page_icon="📄",
    layout="wide"
)

# --- Function to process the PDF and create the vector store ---
@st.cache_resource
def process_pdf(uploaded_file):
    """
    Loads, splits, and embeds the PDF content, then creates a vector store.
    Uses caching to avoid reprocessing the same file.
    """
    try:
        # Create a temporary file to store the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        st.info("Loading and processing the PDF... this may take a moment.")
        
        # Load the PDF
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load_and_split()

        # Split the document into smaller chunks
        pdf_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        docs = pdf_splitter.split_documents(pages)
        documents = [Document(page_content=doc.page_content) for doc in docs]

        # Create embeddings
        st.info("Creating text embeddings and vector store...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Create the vector store
        vector_db = Chroma.from_documents(
            documents,
            embedding=embeddings
        )
        
        # Clean up the temporary file
        os.unlink(tmp_file_path)
        
        st.success("PDF processed successfully! You can now ask questions.")
        return vector_db
    
    except Exception as e:
        st.error(f"An error occurred while processing the PDF: {e}")
        return None

# --- Main App Logic ---

st.title("📄 Chat with Your PDF")
st.markdown("Upload a PDF document and ask questions about its content. Your API key and data are not stored.")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Configuration")
    
    # API Key Input
    api_key = st.text_input(
        "Enter your OpenRouter API Key", 
        type="password",
        help="Get your key from https://openrouter.ai/"
    )

    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload a PDF file", 
        type="pdf"
    )

    st.markdown("---")
    st.markdown(
        "**How it works:**\n"
        "1. Enter your OpenRouter API key.\n"
        "2. Upload a PDF document.\n"
        "3. The app will process the PDF to create a searchable knowledge base.\n"
        "4. Ask questions in the chat window!"
    )

# --- Chat Interface ---

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Please upload a PDF and enter your API key to get started."}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process inputs and handle chat
if uploaded_file and api_key:
    # Set API key for the LLM
    os.environ["OPENROUTER_API_KEY"] = api_key

    # Process the PDF and get the vector store
    vector_db = process_pdf(uploaded_file)

    if vector_db:
        # Accept user input
        if prompt := st.chat_input("Ask a question about the PDF"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Set up the conversational AI model
                        memory = ConversationBufferMemory(
                            memory_key="chat_history", 
                            return_messages=True
                        )
                        llm = ChatOpenAI(
                            model="openai/gpt-3.5-turbo",
                            temperature=0.2,
                            openai_api_base="https://openrouter.ai/api/v1",
                            max_tokens=500,
                            openai_api_key=os.environ["OPENROUTER_API_KEY"]
                        )
                        
                        # Create the conversational chain
                        qa_chain = ConversationalRetrievalChain.from_llm(
                            llm=llm,
                            retriever=vector_db.as_retriever(),
                            memory=memory
                        )
                        
                        # Get the answer
                        answer = qa_chain({"question": prompt})
                        response = answer["answer"]
                        
                        st.markdown(response)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})

                    except Exception as e:
                        error_message = f"An error occurred: {e}"
                        st.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})

elif uploaded_file and not api_key:
    st.warning("Please enter your OpenRouter API key in the sidebar to start chatting.")
elif not uploaded_file and api_key:
     st.info("Please upload a PDF document to begin.")
