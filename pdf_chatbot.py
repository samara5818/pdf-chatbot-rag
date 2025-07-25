import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Load environment variables from .env file
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Load the PDF document
pdf_files = ["test.pdf"]
all_pages = []

for file in pdf_files:
    loader = PyPDFLoader(file)
    pages = loader.load()
    all_pages.extend(pages)

# Split the document into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(pages)

for i, page in enumerate(pages):
    print(f"\n--- Page {i+1} ---\n")
    print(page.page_content[:500])  # Print first 500 chars


# Create embeddings for the document chunks
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = FAISS.from_documents(documents, embeddings)

# Create a retrieval-based question-answering chain
retriever = db.as_retriever()

# set up the gemini llm
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    temperature=0.3,
    max_output_tokens=1024,
    top_p=0.95,
    top_k=40
)

# build retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=False
)   

# CLI loop for user interaction
print("\n Chatbot with Memory Ready! Type 'exit' to quit.\n")

while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break

    response = qa_chain.invoke({"query": query})
    print("Bot:", response.get("answer") or response.get("result") or response)
    print("Source Documents:", response.get("source_documents", []))

