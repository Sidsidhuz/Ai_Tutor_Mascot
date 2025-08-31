
# LangChain RAG pipeline using ChromaDB and Ollama (Llama 3)

import os

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss

# Load and split the markdown file by headings
def split_by_headings(md_text):
    import re
    # Split on lines that start with #, ##, or ###
    pattern = re.compile(r"^#{1,3} .*$", re.MULTILINE)
    splits = []
    last = 0
    for match in pattern.finditer(md_text):
        if match.start() != 0:
            splits.append(md_text[last:match.start()].strip())
        last = match.start()
    splits.append(md_text[last:].strip())
    # Remove empty chunks
    return [chunk for chunk in splits if chunk]

DOC_PATH = os.path.join(os.path.dirname(__file__), "knowledgebase.md")
with open(DOC_PATH, "r", encoding="utf-8") as f:
    md_text = f.read()
chunks = split_by_headings(md_text)

# Embedding model and FAISS index
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

def retrieve(query, k=4):
    query_vec = model.encode([query])
    distances, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]]

def rag_query(question: str, history: list = []):
    # Retrieve relevant chunks
    retrieved_chunks = retrieve(question, k=4)
    context = "\n\n".join(retrieved_chunks)
    print("\n[DEBUG] Retrieved context for question:", question)
    print(context)

    # Strict system prompt
    prompt = (
        "You are an expert AI tutor. Answer ONLY from the provided context. "
        "If the answer is not in the context, say 'Not in my knowledge base.'\n\n"
        f"Context:\n{context}\n\nUser: {question}\nAssistant:"
    )

    # Call Llama 3 via Ollama (assumes ollama is running with llama3)
    import subprocess
    result = subprocess.run([
        "ollama", "run", "llama3"
    ], input=prompt.encode("utf-8"), capture_output=True)
    answer = result.stdout.decode("utf-8").strip()

    # Simple emotion detection
    if "?" in question:
        emotion = "curious"
    elif "sorry" in question.lower():
        emotion = "sad"
    else:
        emotion = "neutral"

    return answer, emotion

# Path to your knowledge base
DOC_PATH = os.path.join(os.path.dirname(__file__), "knowledgebase.md")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

# Load and split the markdown file
loader = UnstructuredMarkdownLoader(DOC_PATH)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

# Create or load Chroma vector store
embedding = OllamaEmbeddings(model="llama3")
vectorstore = Chroma.from_documents(splits, embedding, persist_directory=CHROMA_PATH)

# LLM setup (Ollama)
llm = Ollama(model="llama3")

def rag_query(question: str, history: list = []):
    # Retrieve relevant docs
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    print("\n[DEBUG] Retrieved context for question:", question)
    print(context)

    # Build prompt with context and history
    history_text = ""
    for turn in history:
        history_text += f"User: {turn['question']}\nAssistant: {turn['answer']}\n"
    prompt = (
        "You are a helpful AI tutor. You must answer ONLY using the context below. "
        "If the answer is not in the context, say 'I don't know.' Do not use your own knowledge.\n\n"
        f"Context:\n{context}\n\nUser: {question}\nAssistant:"
    )

    # Get answer from Llama 3
    answer = llm.invoke(prompt)

    # Simple emotion detection (improve as needed)
    if "?" in question:
        emotion = "curious"
    elif "sorry" in question.lower():
        emotion = "sad"
    else:
        emotion = "neutral"

    return answer, emotion
