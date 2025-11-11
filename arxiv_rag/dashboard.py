# ============================================================================
# 3. VISUALIZATION DASHBOARD - dashboard.py (Streamlit)
# ============================================================================

import streamlit as st
from explore_data import DataExplorer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import umap

st.set_page_config(page_title="RAG Data Explorer", layout="wide")

st.title("🔍 ArXiv RAG System - Data Explorer")

# Initialize
@st.cache_resource
def init_explorer():
    return DataExplorer()

explorer = init_explorer()

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", [
    "📊 Overview",
    "🔎 Search Papers",
    "📄 Inspect Chunks",
    "🗺️ Embedding Space",
    "🧪 Test Queries"
])

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================
if page == "📊 Overview":
    st.header("Data Ingestion Overview")
    
    # Get statistics
    with st.spinner("Loading statistics..."):
        stats = explorer.get_statistics()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Chunks", f"{stats['total_chunks']:,}")
    with col2:
        st.metric("Unique Papers", f"{stats['unique_papers']:,}")
    with col3:
        st.metric("Avg Chunks/Paper", f"{stats['avg_chunks_per_paper']:.1f}")
    with col4:
        coverage = (stats['total_chunks'] / stats['unique_papers'])
        st.metric("Data Density", f"{coverage:.1f}x")
    
    # Category distribution
    st.subheader("Paper Categories")
    cat_df = pd.DataFrame(
        list(stats['categories'].items()),
        columns=['Category', 'Count']
    ).sort_values('Count', ascending=False)
    
    fig = px.bar(cat_df, x='Category', y='Count', 
                 title='Distribution by ArXiv Category')
    st.plotly_chart(fig, use_container_width=True)
    
    # Chunk distribution
    st.subheader("Chunks per Paper Distribution")
    chunk_df = pd.DataFrame(
        list(stats['chunk_distribution'].items()),
        columns=['Total Chunks', 'Papers']
    )
    
    fig = px.bar(chunk_df, x='Total Chunks', y='Papers',
                 title='How many chunks does each paper have?')
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 2: SEARCH PAPERS
# ============================================================================
elif page == "🔎 Search Papers":
    st.header("Search Ingested Papers")
    
    search_term = st.text_input("Enter keyword to search papers:", "transformer")
    
    if st.button("Search"):
        with st.spinner("Searching..."):
            results = explorer.search_papers(search_term, limit=20)
        
        if results.empty:
            st.warning(f"No papers found matching '{search_term}'")
        else:
            st.success(f"Found {len(results)} papers")
            
            # Display results
            for idx, row in results.iterrows():
                with st.expander(f"📄 {row['title']}"):
                    st.write(f"**Authors:** {row['authors']}")
                    st.write(f"**Categories:** {row['categories']}")
                    st.write(f"**Published:** {row['published']}")

# ============================================================================
# PAGE 3: INSPECT CHUNKS
# ============================================================================
elif page == "📄 Inspect Chunks":
    st.header("Inspect Document Chunks")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        paper_id = st.text_input("Paper ID (optional):", "")
    with col2:
        n_samples = st.number_input("Number of samples:", 1, 10, 3)
    
    if st.button("Load Chunks"):
        with st.spinner("Loading chunks..."):
            samples = explorer.get_sample_chunks(
                paper_id if paper_id else None,
                n=n_samples
            )
        
        for i, chunk in enumerate(samples, 1):
            st.subheader(f"Chunk {i}")
            st.write(f"**Paper:** {chunk['paper_title']}")
            st.write(f"**Chunk ID:** {chunk['chunk_id']}")
            st.write(f"**Length:** {chunk['length']} characters")
            st.text_area(f"Content Preview {i}", chunk['text_preview'], height=150)
            st.divider()

# ============================================================================
# PAGE 4: EMBEDDING SPACE VISUALIZATION
# ============================================================================
elif page == "🗺️ Embedding Space":
    st.header("Visualize Embedding Space")
    st.info("Sample random chunks and visualize their embeddings in 2D")
    
    sample_size = st.slider("Sample size:", 100, 1000, 500)
    method = st.radio("Reduction method:", ["UMAP", "PCA"])
    
    if st.button("Generate Visualization"):
        with st.spinner("Generating embedding visualization..."):
            # Sample data
            total = explorer.collection.count()
            random_offset = np.random.randint(0, max(1, total - sample_size))
            
            data = explorer.collection.get(
                limit=sample_size,
                offset=random_offset,
                include=["embeddings", "metadatas"]
            )
            
            embeddings = np.array(data["embeddings"])
            categories = [m.get("categories", "Unknown").split(", ")[0] 
                         for m in data["metadatas"]]
            titles = [m.get("title", "Unknown")[:50] + "..." 
                     for m in data["metadatas"]]
            
            # Reduce dimensions
            if method == "UMAP":
                reducer = umap.UMAP(n_components=2, random_state=42)
            else:
                reducer = PCA(n_components=2, random_state=42)
            
            embeddings_2d = reducer.fit_transform(embeddings)
            
            # Create DataFrame
            plot_df = pd.DataFrame({
                'x': embeddings_2d[:, 0],
                'y': embeddings_2d[:, 1],
                'category': categories,
                'title': titles
            })
            
            # Plot
            fig = px.scatter(
                plot_df, x='x', y='y', color='category',
                hover_data=['title'],
                title=f'{method} Projection of Document Embeddings',
                width=800, height=600
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"Visualized {sample_size} document chunks")

# ============================================================================
# PAGE 5: TEST QUERIES
# ============================================================================
elif page == "🧪 Test Queries":
    st.header("Test Retrieval System")
    
    query = st.text_area("Enter your query:", "What are the latest advances in transformers?")
    top_k = st.slider("Number of results:", 1, 10, 5)
    
    if st.button("Run Query"):
        with st.spinner("Retrieving documents..."):
            # Query vector store
            from src.vectorstore.embeddings import EmbeddingGenerator
            
            embedding_gen = EmbeddingGenerator()
            query_embedding = embedding_gen.generate_embeddings([query])
            
            results = explorer.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
        
        st.success(f"Retrieved {len(results['ids'][0])} documents")
        
        # Display results
        for i in range(len(results['ids'][0])):
            similarity = 1 - results['distances'][0][i]
            
            with st.expander(
                f"Result {i+1} - Similarity: {similarity:.3f} - "
                f"{results['metadatas'][0][i]['title']}"
            ):
                st.write(f"**Paper ID:** {results['metadatas'][0][i]['paper_id']}")
                st.write(f"**Authors:** {results['metadatas'][0][i]['authors']}")
                st.write(f"**Categories:** {results['metadatas'][0][i]['categories']}")
                st.divider()
                st.write("**Retrieved Text:**")
                st.write(results['documents'][0][i])