import os
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
import time

# Set Streamlit page config first
st.set_page_config(page_title="Semantic Book Recommender", layout="wide")

# Load environment variables
load_dotenv()

# --- Load books data ---
@st.cache_data
def load_books():
    books_df = pd.read_csv("books_with_emotions.csv")
    books_df["large_thumbnail"] = books_df["thumbnail"].fillna("") + "&fife=w800"
    books_df["large_thumbnail"] = np.where(
        books_df["large_thumbnail"].isna(),
        "cover_not_found.jpg",
        books_df["large_thumbnail"],
    )
    books_df["small_thumbnail"] = books_df["thumbnail"].fillna("") + "&fife=w200"
    return books_df

books = load_books()

# --- Load or Generate FAISS Vector DB ---
@st.cache_resource
def get_or_create_faiss():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Check if FAISS index already exists
    if not os.path.exists("faiss_index/index.faiss"):
        st.info("🔄 Generating vector index, this may take a moment...")
        loader = UnstructuredFileLoader("tagged_description.txt")
        raw_documents = loader.load()

        splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=50, separator="\n")
        documents = splitter.split_documents(raw_documents)

        db = FAISS.from_documents(documents, embeddings)
        db.save_local("faiss_index")
        st.success("✅ Vector index saved to disk.")
    else:
        # Loading existing FAISS index without extra messages
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    return db

db_books = get_or_create_faiss()

# --- Recommendation Logic ---
def retrieve_semantic_recommendations(query: str, category: str = None, tone: str = None,
                                      initial_top_k: int = 50, final_top_k: int = 16) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness", ascending=False)

    return book_recs

# --- Streamlit UI ---
# Add custom CSS
st.markdown("""
    <style>
        body { font-family: 'Segoe UI', sans-serif; }
        .block-container { padding: 2rem 4rem; }
        h1 { color: #3B3B98; }
        .stButton>button {
            background-color: #3B3B98;
            color: white;
            padding: 0.5rem 1.5rem;
            border-radius: 0.5rem;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #5758BB;
        }
        .recommendation-card {
            background-color: #f4f4f5;
            padding: 1rem;
            border-radius: 0.75rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        .stImage > img {
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# Input section
st.title("📚 Semantic Book Recommender")
query = st.text_input("Enter a description of a book (e.g., A story about forgiveness):")

categories = ["All"] + sorted(books["categories"].dropna().astype(str).unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

col1, col2 = st.columns(2)
with col1:
    category = st.selectbox("Select a category:", categories, index=0)
with col2:
    tone = st.selectbox("Select an emotional tone:", tones, index=0)

if st.button("🔍 Find Recommendations") and query:
    # Show a spinner while generating the recommendations
    with st.spinner("🔄 Searching for the best matches..."):
        # Optional: Add progress bar during processing
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.1)  # Simulate some work being done
            progress_bar.progress(i + 1)

        recs = retrieve_semantic_recommendations(query, category, tone)
        st.subheader("Recommended Books")

        if recs.empty:
            st.warning("No recommendations found. Please try a different query or category.")
        else:
            for _, row in recs.iterrows():
                with st.container():
                    col_img, col_txt = st.columns([1, 5])
                    with col_img:
                        st.image(row["large_thumbnail"], width=500)
                    with col_txt:
                        st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
                        description = " ".join(row["description"].split()[:100]) + "..."
                        authors = row["authors"].split(";")
                        if len(authors) == 2:
                            authors_str = f"{authors[0]} and {authors[1]}"
                        elif len(authors) > 2:
                            authors_str = f"{', '.join(authors[:-1])}, and {authors[-1]}"
                        else:
                            authors_str = row["authors"]
                        st.markdown(f"**{row['title']}** by *{authors_str}*  \n{description}")
                        st.markdown('</div>', unsafe_allow_html=True)
    
    st.success("✅ Recommendations displayed!")
