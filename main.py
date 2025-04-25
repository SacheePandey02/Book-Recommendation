import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from PIL import Image
import requests
from io import BytesIO

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# Load book data
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover_not_found.jpg",
    books["large_thumbnail"]
)
books["small_thumbnail"] = books["thumbnail"] + "&fife=w200"

# Load and process documents
file_path = "tagged_description.txt"
loader = UnstructuredFileLoader(file_path)
raw_documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db_books = Chroma.from_documents(documents=documents, embedding=embeddings)

# Recommendation function
def retreive_semantic_recommendations(query, category=None, tone=None, initial_top_k=50, final_top_k=16):
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

# Streamlit UI
st.title("📚 Semantic Book Recommender")

query = st.text_input("Please enter a description of a book:", placeholder="e.g., A story about forgiveness")

categories = ["All"] + sorted(books["categories"].dropna().astype(str).unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

category = st.selectbox("Select a category:", categories, index=0)
tone = st.selectbox("Select an emotional tone:", tones, index=0)

if st.button("Find recommendations"):
    if not query:
        st.warning("Please enter a description to get recommendations.")
    else:
        with st.spinner("Fetching recommendations..."):
            results = retreive_semantic_recommendations(query, category, tone)
            st.subheader("Recommendations")

            for _, row in results.iterrows():
                col1, col2 = st.columns([1, 4])

                with col1:
                    try:
                        response = requests.get(row["large_thumbnail"])
                        img = Image.open(BytesIO(response.content))
                        st.image(img, width=100)
                    except:
                        st.image("cover_not_found.jpg", width=100)

                with col2:
                    authors_split = row["authors"].split(";")
                    if len(authors_split) == 2:
                        authors_str = f"{authors_split[0]} and {authors_split[1]}"
                    elif len(authors_split) > 2:
                        authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
                    else:
                        authors_str = row["authors"]

                    description = row["description"]
                    truncated_desc = " ".join(description.split()[:30]) + "..."
                    st.markdown(f"**{row['title']}** by *{authors_str}*")
                    st.caption(truncated_desc)
