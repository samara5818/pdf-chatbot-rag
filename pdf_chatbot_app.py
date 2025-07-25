import os
import shutil
import streamlit as st
from dotenv import load_dotenv
import base64
from PIL import Image

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

def get_base64_of_file(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()




# Fix for OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# --- Session State ---
if "username" not in st.session_state:
    st.session_state.username = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Login Page ---
if st.session_state.username is None:
    st.set_page_config(page_title="PDF ChatBot Login", layout="centered")
    bg_image = get_base64_of_file("login_background.jpg")

    

    st.markdown(f"""
        <style>
            html, body {{
                height: 100%;
                margin: 0;
                padding: 0;
            }}
            
            .stApp {{
                background-image: url("data:image/jpg;base64,{bg_image}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
                display: flex;
                justify-content: center;
                align-items: center;
            }}
                

            .block-container {{
                backdrop-filter: blur(4px);
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                backdrop-filter: blur(4px);
                background-color: rgba(255, 255, 255, 0.75);
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 0 30px rgba(0,0,0,0.1);
                text-align: center;
            }}
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <style>
            .login-title {
                text-align: center;
                font-size: 36px;
                font-weight: 800;
                color: #262730;
            }
            .login-sub {
                text-align: center;
                font-size: 20px;
                font-weight: 400;
                margin-top: -1rem;
                margin-bottom: 2rem;
            }
            .login-input label {
                font-size: 16px !important;
                font-weight: 500;
            }
        </style>
    """, unsafe_allow_html=True)

    
    st.markdown("<h1 class='login-title'>PDF ChatBot - Welcome</h1>", unsafe_allow_html=True)
    st.markdown("<p class='login-sub'>Chat with your PDF</p>", unsafe_allow_html=True)


    with st.container():
        username_input = st.text_input("👤 Your Name", label_visibility="visible")

    if st.button("Continue", use_container_width=True) and username_input.strip():
        st.session_state.username = username_input.strip()
        st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)


    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
        <style>
            .sidebar-title {
                font-size: 24px;
                font-weight: bold;
                color: #4B8BBE;
                margin-bottom: 0.5rem;
            }

            .sidebar-subtitle {
                font-size: 18px;
                margin-bottom: 1rem;
            }

            .sidebar-button {
                background-color: #4B8BBE; 
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
            }

            .sidebar-button:hover {
                background-color: #357ABD; 
                color: white;
            }
        </style> 
                """, unsafe_allow_html=True)
    st.markdown(f"<div class='sidebar-title'>Hello, {st.session_state.username}!</div>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Personnal Chat bot-RAG based chat bot</div>', unsafe_allow_html=True)

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
    
    if st.button("Refresh"):
        st.rerun()
    
    if st.button(" Clear Vector Store"):
        if os.path.exists("vector_index/faiss_index"):
            shutil.rmtree("vector_index/faiss_index")
            st.success("Vector store cleared!")
            st.rerun()

# --- Page Setup ---
st.set_page_config(page_title="PDF ChatBot", layout="wide")
st.title(" Chat with your PDFs (Gemini + LangChain)")

# --- Chat Log Download ---
def chat_to_text():
    lines = []
    for msg in st.session_state.chat_history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            name = st.session_state.username if msg["role"] == "user" else "Assistant"
            lines.append(f"{name}: {msg['content']}")
    return "\n".join(lines)

if st.session_state.chat_history:
    st.download_button(
        label="⬇️ Download Chat Log",
        data=chat_to_text(),
        file_name="chat_log.txt",
        mime="text/plain",
        help="Download the chat history as a text file."
    )

# Embedding and Faiss load
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db=None
retriever=None
qa_chain=None

if os.path.exists("vector_index/faiss_index/index.faiss"):
    db= FAISS.load_local(
        "vector_index/faiss_index", 
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = db.as_retriever()

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash",
        temperature=0.3,
        max_output_tokens=1024,
        top_p=0.95,
        top_k=40
    
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False
    )

# --- File Upload ---
uploaded_files = st.file_uploader("📁 Upload PDF(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    with st.spinner(" Processing uploaded PDFs..."):
        all_pages = []
        upload_dir = os.path.join(os.getcwd(), "pdf_uploads")
        os.makedirs(upload_dir, exist_ok=True)


        for i, uploaded_file in enumerate(uploaded_files):
            temp_path = os.path.join(upload_dir, f"temp_{i}.pdf")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

            try:
                loader = PyMuPDFLoader(temp_path)
                pages = loader.load()
                for j, doc in enumerate(pages):
                    doc.metadata["page"] = j + 1
                all_pages.extend(pages)
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {e}")

        if not all_pages:
            st.error("⚠️ No valid pages found. Please upload valid text-based PDFs.")
            st.stop()

        # Split and embed text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(all_pages)

        if not documents:
            st.error("⚠️ No text could be extracted. Please upload text-based PDFs.")
            st.stop()

        if os.path.exists("vector_index/faiss_index"):
            db = FAISS.load_local(
                "vector_index/faiss_index", 
                embeddings,
                allow_dangerous_deserialization=True
            )
            db.add_documents(documents)
        else:
            db = FAISS.from_documents(documents, embeddings)
        
        db.save_local("vector_index/faiss_index")
        retriever = db.as_retriever()

        # LLM & Memory
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-1.5-flash",
            temperature=0.3,
            max_output_tokens=1024,
            top_p=0.95,
            top_k=40
        )

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # QA Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=False
        )

# --- Chat Input ---
if qa_chain:
    user_input = st.chat_input("Ask something about the PDFs...")
    if user_input:
        response = qa_chain.invoke({"query": user_input})
        answer = response.get("answer") or response.get("result") or str(response)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

# --- Chat Display ---
for msg in st.session_state.chat_history:
    if isinstance(msg, dict) and "role" in msg and "content" in msg:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if not uploaded_files and not os.path.exists("vector_index/faiss_index/index.faiss"):
    st.info(" Please upload PDF(s) to begin chatting.")
