import os
import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# ---------------------- CACHED BACKEND FUNCTIONS ----------------------

@st.cache_resource(show_spinner="🔍 Processing your document...")
def create_vector_db(uploaded_file):
    """Creates a Chroma vector database from the uploaded PDF file."""
    # Use tempfile to save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    
    try:
        # Load & Split
        loader = PyPDFLoader(tmp_path)
        pages = loader.load_and_split()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        docs = text_splitter.split_documents(pages)
        documents = [Document(page_content=doc.page_content) for doc in docs]
        
        # Create Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Create Vector DB
        vector_db = Chroma.from_documents(documents, embedding=embeddings)
        return vector_db
    
    finally:
        # Clean up temp file
        os.unlink(tmp_path)

@st.cache_resource
def get_llm(api_key):
    """Initializes and returns the ChatOpenAI LLM."""
    return ChatOpenAI(
        model="openai/gpt-3.5-turbo",
        temperature=0.2,
        base_url="https://openrouter.ai/api/v1",
        max_tokens=500,
        api_key=api_key
    )

def get_response(llm, retriever, chat_history, question):
    """Generates a response using the RAG pipeline."""
    # Create RAG prompt template
    template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Keep the answer concise and relevant.

Context: {context}

Chat History: {chat_history}

Question: {question}

Answer:"""
    prompt_template = ChatPromptTemplate.from_template(template)

    # Format chat history (using "user" and "assistant" roles)
    chat_history_text = "No previous conversation."
    if chat_history:
        formatted = []
        for role, msg in chat_history[-6:]:  # Last 3 exchanges
            formatted.append(f"{role.capitalize()}: {msg}")
        chat_history_text = "\n".join(formatted)

    # Retrieve relevant documents
    relevant_docs = retriever.invoke(question)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    context = format_docs(relevant_docs)
    
    # Format the prompt
    formatted_prompt = prompt_template.format(
        context=context,
        chat_history=chat_history_text,
        question=question
    )
    
    # Get response from LLM
    answer = llm.invoke(formatted_prompt).content
    return answer

# ---------------------- MAIN APP UI ----------------------

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="📄 Smart PDF Chatbot",
    layout="wide",
    page_icon="🤖"
)

# --- SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# --- SIDEBAR ---
with st.sidebar:
    st.title("📚 PDF Q&A")
    st.markdown("Upload a PDF document and chat with it instantly!")
    
    uploaded_file = st.file_uploader("📤 Upload a PDF file", type=["pdf"])
    
    api_key = st.text_input(
        "🔑 Enter your OpenRouter API Key",
        type="password",
        help="Get your free key from https://openrouter.ai/"
    )
    
    if api_key:
        st.success("✅ API key entered!")
    else:
        st.warning("⚠️ Please enter your OpenRouter API key to start chatting.")
    
    if st.button("🗑️ Clear Chat & Reset"):
        st.session_state.chat_history = []
        st.session_state.vector_db = None
        # Clear the cache for the vector DB
        create_vector_db.clear()
        st.rerun()

# --- MAIN CHAT AREA ---
st.title("🤖 AI-Powered PDF Chatbot")

# Handle file upload logic
if uploaded_file:
    # This will only run if the file is new, thanks to caching
    if st.session_state.vector_db is None:
        st.session_state.vector_db = create_vector_db(uploaded_file)
        st.success("✅ Document processed! Ready to chat.")

# Display chat history
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

# Chat input
if user_question := st.chat_input("Ask a question about your document..."):
    if not api_key:
        st.warning("Please enter your OpenRouter API key in the sidebar to chat.")
        st.stop()
    
    if st.session_state.vector_db is None:
        st.warning("Please upload a PDF file first.")
        st.stop()

    # Add user message to history and display
    st.session_state.chat_history.append(("user", user_question))
    with st.chat_message("user"):
        st.markdown(user_question)

    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            llm = get_llm(api_key)
            retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 3})
            
            answer = get_response(
                llm, 
                retriever, 
                st.session_state.chat_history, 
                user_question
            )
            
            st.markdown(answer)
    
    # Add bot response to history
    st.session_state.chat_history.append(("assistant", answer))

elif not st.session_state.chat_history:
     # Initial welcome message if no history
    st.info("Upload a PDF in the sidebar to get started!")
