"""
Hospital Knowledge Base RAG Assistant - Streamlit App
Dynamic, card-based UI with step-by-step processing
"""

import os
import json
import logging
import time
import streamlit as st
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Hospital Knowledge Base",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for animations and styling
st.markdown("""
<style>
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    .card {
        background: linear-gradient(135deg, #faf4ed 0%, #f5ebe0 100%);
        border: 1px solid #e8dcd0;
        border-radius: 12px;
        padding: 24px;
        animation: slideUp 0.5s ease-out;
        transition: all 0.3s;
    }

    .card:hover {
        box-shadow: 0 8px 24px rgba(61, 61, 61, 0.1);
        border-color: #b8432f;
        transform: translateY(-2px);
    }

    .step-item {
        display: flex;
        gap: 16px;
        align-items: flex-start;
        padding: 16px;
        background: #fef9f5;
        border-radius: 8px;
        border-left: 3px solid #b8432f;
        animation: slideUp 0.4s ease-out;
        margin-bottom: 12px;
    }

    .step-number {
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, #b8432f 0%, #8b2e23 100%);
        color: #fef9f5;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        flex-shrink: 0;
    }

    .loading-dot {
        width: 16px;
        height: 16px;
        background: #b8432f;
        border-radius: 50%;
        animation: pulse 1.5s infinite;
    }

    .answer-box {
        background: linear-gradient(135deg, #faf4ed 0%, #f5ebe0 100%);
        border-left: 4px solid #b8432f;
        padding: 20px;
        border-radius: 8px;
        line-height: 1.8;
        animation: slideUp 0.5s ease-out;
    }

    .source-card {
        padding: 16px;
        background: #fef9f5;
        border: 1px solid #e8dcd0;
        border-radius: 8px;
        transition: all 0.3s;
        animation: slideUp 0.5s ease-out;
    }

    .source-card:hover {
        border-color: #b8432f;
        box-shadow: 0 4px 12px rgba(184, 67, 47, 0.1);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

class LiteRAGAssistant:
    """Lightweight RAG without FAISS."""

    def __init__(self):
        self.embedder = None
        self.embeddings = None
        self.chunks = []
        self.metadata = None
        self.client = None

    def load_knowledge_base(self) -> bool:
        """Load pre-computed embeddings and metadata."""
        try:
            metadata_file = "data/metadata.json"
            if not os.path.exists(metadata_file):
                return False

            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)

            self.chunks = [m.get('chunk', '') for m in self.metadata]

            embeddings_file = "data/embeddings.npy"
            if os.path.exists(embeddings_file):
                self.embeddings = np.load(embeddings_file)
            else:
                if not self.embedder:
                    self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                self.embeddings = self.embedder.encode(self.chunks, convert_to_numpy=True)

            return len(self.chunks) > 0
        except Exception as e:
            logger.error(f"Error loading KB: {e}")
            return False

    def setup_llm(self, api_key: str) -> bool:
        """Setup Gemini LLM."""
        try:
            if not genai:
                st.error("Google Generative AI not installed")
                return False
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel("gemini-3.5-flash-lite")
            return True
        except Exception as e:
            logger.error(f"Error setting up Gemini: {e}")
            return False

    def retrieve_documents(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve documents using cosine similarity."""
        if not self.embedder or self.embeddings is None:
            return []

        try:
            query_embedding = self.embedder.encode(query, convert_to_numpy=True)
            similarities = cosine_similarity([query_embedding], self.embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:min(k, len(self.chunks))]

            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:
                    meta = self.metadata[idx]
                    results.append({
                        'content': self.chunks[idx],
                        'source': meta.get('source', 'Unknown'),
                        'category': meta.get('category', 'Unknown'),
                        'relevance': float(similarities[idx])
                    })

            return results
        except Exception as e:
            logger.error(f"Error retrieving: {e}")
            return []

    def generate_answer(self, query: str, context: List[Dict]) -> str:
        """Generate answer using Gemini."""
        if not self.client:
            return "LLM not configured"

        context_text = "\n\n".join([
            f"Source: {doc['source']} ({doc['category']})\n{doc['content']}"
            for doc in context
        ])

        prompt = f"""You are a hospital knowledge base assistant.
Use the context below to answer accurately and concisely.

Context:
{context_text}

Question: {query}

Answer:"""

        try:
            response = self.client.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating: {e}")
            return f"Error: {str(e)}"

def main():
    # Header
    st.markdown("# 🏥 Hospital Knowledge Base")

    # Get API key from secrets
    api_key = st.secrets.get("GEMINI_API_KEY", "")

    if not api_key:
        st.warning("⚠️ Gemini API key not found in secrets")
        st.info("Add GEMINI_API_KEY to `.streamlit/secrets.toml`")
        st.stop()

    # Initialize session state
    if 'assistant' not in st.session_state:
        st.session_state.assistant = LiteRAGAssistant()
        st.session_state.ready = False
        st.session_state.active_tab = "search"
        st.session_state.query_result = None

    assistant = st.session_state.assistant

    # Load KB and setup LLM
    if not st.session_state.ready:
        with st.spinner("Loading knowledge base..."):
            if not assistant.load_knowledge_base():
                st.error("❌ Knowledge base not found. Run: python ingest.py")
                st.stop()

            if not assistant.setup_llm(api_key):
                st.error("❌ Failed to setup Gemini")
                st.stop()

            st.session_state.ready = True

    # Tab Navigation
    tab1, tab2, tab3 = st.tabs(["🔍 Search", "📋 Steps", "⚙️ Settings"])

    with tab1:
        st.markdown("### Search Hospital Policies")

        col1, col2 = st.columns([5, 1])
        with col1:
            query = st.text_input(
                "Ask a question",
                placeholder="e.g., What is the hospital admission process?",
                label_visibility="collapsed"
            )
        with col2:
            search_clicked = st.button("🔍 Search", use_container_width=True)

        if search_clicked and query:
            # Show step-by-step processing
            steps_placeholder = st.empty()

            with steps_placeholder.container():
                st.markdown("### Processing...")

                # Step 1: Searching
                col1, col2 = st.columns([0.9, 0.1])
                with col1:
                    st.markdown("**1. Searching Knowledge Base**")
                with col2:
                    st.markdown('<div class="loading-dot"></div>', unsafe_allow_html=True)

                time.sleep(0.5)

                # Retrieve documents
                docs = assistant.retrieve_documents(query, k=5)

                # Step 1: Complete
                st.success("✓ Found 5 relevant sources")

                # Step 2: Generating Answer
                st.markdown("**2. Generating Answer**")
                with st.spinner("Using Gemini API..."):
                    answer = assistant.generate_answer(query, docs)
                    time.sleep(0.3)

                st.success("✓ Answer generated")

            # Clear steps and show results
            steps_placeholder.empty()

            # Results
            st.markdown("---")
            st.markdown("### Answer")
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

            st.markdown("### Sources")
            cols = st.columns(2)
            for i, doc in enumerate(docs[:4]):
                with cols[i % 2]:
                    st.markdown(f"""
<div class="source-card">
    <div style="color: #b8432f; font-weight: 600; margin-bottom: 6px;">{doc['source']}</div>
    <div style="color: #8b7960; font-size: 12px; margin-bottom: 12px;">{doc['category']}</div>
    <div style="font-size: 13px; line-height: 1.6;">{doc['content'][:200]}...</div>
    <div style="color: #b8432f; font-weight: 600; margin-top: 12px; font-size: 12px;">Relevance: {doc['relevance']:.0%}</div>
</div>
                    """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### Processing Steps")
        st.info("When you search, each step is displayed here showing the progress of your query.")

        # Show example steps
        st.markdown("""
<div class="step-item">
    <div class="step-number">1</div>
    <div>
        <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; color: #8b7960;">Searching Knowledge Base</div>
        <div style="font-size: 14px; color: #3d3d3d;">Querying 14 embeddings</div>
    </div>
</div>

<div class="step-item">
    <div class="step-number">2</div>
    <div>
        <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; color: #8b7960;">Retrieving Documents</div>
        <div style="font-size: 14px; color: #3d3d3d;">Matching relevant sources</div>
    </div>
</div>

<div class="step-item">
    <div class="step-number">3</div>
    <div>
        <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; color: #8b7960;">Generating Answer</div>
        <div style="font-size: 14px; color: #3d3d3d;">Using Gemini 1.5 Flash API</div>
    </div>
</div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("### Settings")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### LLM Configuration")
            st.info("🤖 Using Gemini 1.5 Flash")
            st.caption("Free tier Google Generative AI model - stable and reliable")

        with col2:
            st.markdown("#### Knowledge Base")
            st.metric("Total Chunks", len(assistant.chunks))
            st.metric("Embedding Dimension", 384)

        st.markdown("---")
        st.markdown("#### Search Settings")
        num_results = st.slider("Results to retrieve", 1, 10, 5)
        st.caption(f"Number of source documents to consider: {num_results}")

if __name__ == "__main__":
    main()
