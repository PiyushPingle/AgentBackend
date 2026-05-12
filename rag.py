import fitz  # PyMuPDF
import chromadb
import os
import requests

from PIL import Image
import pytesseract

from sentence_transformers import SentenceTransformer
from groq import Groq

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# CONFIG
# -----------------------------
PDF_PATH = r"C:\Users\piyus\OneDrive\Desktop\GENAI\Marketing Agent\MarketResearch.pdf"
CHROMA_PATH = r"C:\RAG\chroma_db"
COLLECTION_NAME = "rag_collection"

os.environ["GROQ_API_KEY"] = "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"
os.environ["SERPER_API_KEY"] = "5a9438aadd7ce89b9b6b60264232fea3487a5da2" #https://serper.dev/api-keys

SERPER_API_KEY = os.environ["5a9438aadd7ce89b9b6b60264232fea3487a5da2"]

# -----------------------------
# MODELS
# -----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
llm_client = Groq(api_key=os.environ["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"])

# -----------------------------
# GLOBALS
# -----------------------------
vectorizer = None
tfidf_matrix = None
documents_store = []

# -----------------------------
# PDF LOADER
# -----------------------------
def load_pdf(path):
    doc = fitz.open(path)
    pages = []

    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({
                "page": i + 1,
                "text": text
            })

    return pages

# -----------------------------
# CHUNKING
# -----------------------------
def chunk_text(pages, chunk_size=1000, overlap=200):
    chunks = []
    global_id = 0

    for page in pages:
        text = page["text"]
        page_num = page["page"]

        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            if len(chunk.strip()) > 50:
                chunks.append({
                    "id": global_id,
                    "text": chunk,
                    "metadata": {
                        "page": page_num,
                        "source": "MarketResearch.pdf"
                    }
                })
                global_id += 1

            start += chunk_size - overlap

    return chunks

# -----------------------------
# KEYWORD INDEX
# -----------------------------
def build_keyword_index(chunks):
    global vectorizer, tfidf_matrix, documents_store

    documents_store = [c["text"] for c in chunks]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents_store)

# -----------------------------
# EMBEDDINGS
# -----------------------------
def create_embeddings(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = embedding_model.encode(texts)

    records = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        records.append({
            "id": str(chunk["id"]),
            "text": chunk["text"],
            "embedding": emb.tolist(),
            "metadata": chunk["metadata"]
        })

    return records

# -----------------------------
# STORE IN CHROMA
# -----------------------------
def store_in_chroma(records):
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    collection.add(
        ids=[r["id"] for r in records],
        embeddings=[r["embedding"] for r in records],
        documents=[r["text"] for r in records],
        metadatas=[r["metadata"] for r in records]
    )

    return collection

# -----------------------------
# RETRIEVAL (HYBRID SEARCH)
# -----------------------------
def retrieve(query, collection, k=4, alpha=0.7):

    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
        include=["documents", "metadatas", "distances"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    query_vec = vectorizer.transform([query])
    keyword_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    scored = []

    for i, (doc, meta) in enumerate(zip(docs, metas)):
        vector_score = 1 / (1 + distances[i])
        keyword_score = keyword_scores[i] if i < len(keyword_scores) else 0

        final_score = alpha * vector_score + (1 - alpha) * keyword_score

        scored.append((final_score, doc, meta))

    scored.sort(reverse=True, key=lambda x: x[0])

    top = scored[:k]

    return [x[1] for x in top], [x[2] for x in top]

# -----------------------------
# CONTEXT BUILDER
# -----------------------------
def build_context(docs, metas):
    context = ""
    for chunk, meta in zip(docs, metas):
        context += f"[Page {meta['page']}]\n{chunk}\n\n"
    return context

# -----------------------------
# LLM CALL
# -----------------------------
def call_llm(prompt):
    response = llm_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

# -----------------------------
# WEB SEARCH
# -----------------------------
def serper_search(query):
    url = "https://google.serper.dev/search"

    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, json={"q": query})
    data = response.json()

    results = data.get("organic", [])

    context = ""
    for r in results[:5]:
        context += f"Title: {r.get('title')}\nSnippet: {r.get('snippet')}\n\n"

    return context

# -----------------------------
# ROUTER
# -----------------------------
def route_agent(query):
    q = query.lower()

    if any(x in q for x in ["trend", "future", "forecast", "growth"]):
        return "trend"
    elif any(x in q for x in ["competitor", "vs", "compare", "alternatives"]):
        return "competitor"
    else:
        return "marketing"

# -----------------------------
# AGENTS
# -----------------------------
def marketing_agent(query, context):
    prompt = f"You are a senior marketing strategist.\n\nContext:\n{context}\n\nQuestion:\n{query}"
    return call_llm(prompt)

def trend_agent(query, context):
    prompt = f"You are a market trend analyst.\n\nContext:\n{context}\n\nQuestion:\n{query}"
    return call_llm(prompt)

def competitor_agent(query, context):
    prompt = f"You are a competitive intelligence analyst.\n\nContext:\n{context}\n\nQuestion:\n{query}"
    return call_llm(prompt)

# -----------------------------
# IMAGE OCR
# -----------------------------
def extract_text_from_image(image_path):
    image = Image.open(image_path)
    return pytesseract.image_to_string(image)

def image_query_pipeline(image_path, collection):

    query = extract_text_from_image(image_path)

    print("\n🖼️ OCR Text:\n", query)

    docs, metas = retrieve(query, collection)
    context = build_context(docs, metas)

    if len(context) < 200:
        context = serper_search(query)

    agent = route_agent(query)

    if agent == "marketing":
        return marketing_agent(query, context)
    elif agent == "trend":
        return trend_agent(query, context)
    else:
        return competitor_agent(query, context)

# -----------------------------
# MAIN
# -----------------------------
def main():

    print("\n📄 Loading PDF...")
    pages = load_pdf(PDF_PATH)

    print("\n✂️ Chunking...")
    chunks = chunk_text(pages)

    print("\n🔤 Keyword Index...")
    build_keyword_index(chunks)

    print("\n🧠 Embeddings...")
    records = create_embeddings(chunks)

    print("\n💾 Chroma DB...")
    collection = store_in_chroma(records)

    while True:

        choice = input("\nType (text/image/exit): ").lower()

        if choice == "exit":
            break

        # ---------------- TEXT ----------------
        if choice == "text":
            query = input("\n🔍 Ask: ")

            docs, metas = retrieve(query, collection)
            context = build_context(docs, metas)

            if len(context) < 200:
                context = serper_search(query)

            agent = route_agent(query)

            if agent == "marketing":
                answer = marketing_agent(query, context)
            elif agent == "trend":
                answer = trend_agent(query, context)
            else:
                answer = competitor_agent(query, context)

            print("\n🤖 Answer:\n", answer)

        # ---------------- IMAGE ----------------
        elif choice == "image":
            path = input("\n📷 Image path: ")
            answer = image_query_pipeline(path, collection)

            print("\n🤖 Answer:\n", answer)

if __name__ == "__main__":
    main()