"""
Hospital Knowledge Base RAG Assistant using Streamlit.
Supports both Groq and Google Generative AI APIs.
"""

import os
import json
import logging
import streamlit as st
import numpy as np
import faiss
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

# LLM imports
try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Hospital Knowledge Base Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

class RAGAssistant:
    """RAG-based hospital knowledge base assistant."""

    def __init__(self):
        self.embeddings = None
        self.faiss_index = None
        self.metadata = None
        self.chunks = []
        self.llm_provider = None
        self.client = None

    def initialize_embeddings(self):
        """Initialize embeddings model."""
        if self.embeddings is None:
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={"device": "cpu"}
                )
                logger.info("Embeddings model initialized")
            except Exception as e:
                logger.error(f"Error initializing embeddings: {e}")
                return False
        return True

    def load_faiss_index(self, index_path: str = "data/faiss_index") -> bool:
        """Load FAISS index and metadata."""
        try:
            if not os.path.exists(index_path):
                logger.warning(f"FAISS index not found at {index_path}")
                return False

            # Load FAISS index
            index_file = os.path.join(index_path, "index.faiss")
            if os.path.exists(index_file):
                self.faiss_index = faiss.read_index(index_file)
                logger.info(f"FAISS index loaded from {index_file}")

            # Load metadata
            metadata_file = os.path.join(index_path, "..", "metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                self.chunks = [m.get('chunk', '') for m in self.metadata]
                logger.info(f"Loaded {len(self.metadata)} chunks from metadata")

            return self.faiss_index is not None
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            return False

    def setup_llm(self, provider: str, api_key: str) -> bool:
        """Setup LLM provider."""
        try:
            if provider == "groq":
                if not Groq:
                    st.error("Groq library not installed. Install with: pip install groq")
                    return False
                self.client = Groq(api_key=api_key)
                self.llm_provider = "groq"
                logger.info("Groq client initialized")
                return True

            elif provider == "gemini":
                if not genai:
                    st.error("Google Generative AI library not installed. Install with: pip install google-generativeai")
                    return False
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel("gemini-1.5-flash")
                self.llm_provider = "gemini"
                logger.info("Gemini client initialized with gemini-1.5-flash")
                return True
        except Exception as e:
            logger.error(f"Error setting up LLM: {e}")
            return False

        return False

    def retrieve_documents(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant documents from FAISS index."""
        if not self.initialize_embeddings() or self.faiss_index is None:
            return []

        try:
            # Embed query
            query_embedding = self.embeddings.embed_query(query)
            query_vector = np.array([query_embedding]).astype('float32')

            # Search FAISS index
            distances, indices = self.faiss_index.search(query_vector, min(k, len(self.chunks)))

            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.metadata):
                    meta = self.metadata[idx]
                    results.append({
                        'content': self.chunks[idx],
                        'source': meta.get('source', 'Unknown'),
                        'category': meta.get('category', 'Unknown'),
                        'distance': float(distance),
                        'relevance': 1 / (1 + float(distance))
                    })

            return results
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

    def generate_answer(self, query: str, context: List[Dict]) -> str:
        """Generate answer using LLM."""
        if not self.client or not self.llm_provider:
            return "LLM not configured"

        # Prepare context
        context_text = "\n\n".join([
            f"Source: {doc['source']} (Category: {doc['category']})\n{doc['content']}"
            for doc in context
        ])

        prompt = f"""You are a helpful hospital knowledge base assistant.
Use the following context from hospital policies and guidelines to answer the question accurately.

Context:
{context_text}

Question: {query}

Answer: Provide a clear, concise answer based on the context. If the context doesn't contain relevant information, say so explicitly."""

        try:
            if self.llm_provider == "groq":
                response = self.client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1024,
                    top_p=1
                )
                return response.choices[0].message.content

            elif self.llm_provider == "gemini":
                response = self.client.generate_content(prompt)
                return response.text

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"

def main():
    """Streamlit main application."""

    # Sidebar configuration
    st.sidebar.title("🏥 Hospital KB Assistant")
    st.sidebar.markdown("---")

    # LLM Provider Selection
    st.sidebar.subheader("LLM Configuration")
    provider = st.sidebar.radio(
        "Select LLM Provider",
        ["Groq", "Gemini"],
        help="Choose the AI provider for generating answers"
    )

    # Get API key from secrets or input
    if provider == "Groq":
        api_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""
    else:
        api_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""

    if not api_key:
        st.sidebar.warning(f"⚠️ {provider} API key not found in Streamlit secrets")
        st.sidebar.info(f"Add your {provider} API key to `.streamlit/secrets.toml`")
        st.stop()

    # Initialize session state
    if 'assistant' not in st.session_state:
        st.session_state.assistant = RAGAssistant()
        st.session_state.initialized = False

    # Initialize assistant
    if not st.session_state.initialized:
        with st.spinner("Initializing assistant..."):
            assistant = st.session_state.assistant

            # Load FAISS index
            if not assistant.load_faiss_index():
                st.error("❌ Failed to load knowledge base. Please run `python ingest.py` first.")
                st.stop()

            # Setup LLM
            provider_key = "groq" if provider == "Groq" else "gemini"
            if not assistant.setup_llm(provider_key, api_key):
                st.error(f"❌ Failed to setup {provider}. Check API key.")
                st.stop()

            st.session_state.initialized = True

    assistant = st.session_state.assistant

    # Main content
    st.title("🏥 Hospital Knowledge Base Assistant")
    st.markdown("Ask questions about hospital policies, clinical guidelines, and procedures.")

    # Query input
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Enter your question:",
            placeholder="e.g., What is the hospital admission process?",
            label_visibility="collapsed"
        )
    with col2:
        search_button = st.button("🔍 Search", use_container_width=True)

    # Settings
    with st.sidebar:
        st.subheader("Search Settings")
        num_results = st.slider("Number of references to retrieve", 1, 10, 5)

    # Process query
    if search_button and query:
        with st.spinner("Searching knowledge base..."):
            # Retrieve relevant documents
            documents = assistant.retrieve_documents(query, k=num_results)

            if not documents:
                st.error("❌ No relevant documents found. Please try a different query or ingest PDFs.")
            else:
                # Generate answer
                with st.spinner(f"Generating answer using {provider}..."):
                    answer = assistant.generate_answer(query, documents)

                # Display results
                st.markdown("---")
                st.subheader("📝 Answer")
                st.markdown(answer)

                # Display sources
                st.markdown("---")
                st.subheader("📚 Sources")

                # Create tabs for each source
                tabs = st.tabs([f"Source {i+1}" for i in range(len(documents))])

                for tab, doc in zip(tabs, documents):
                    with tab:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Source", doc['source'])
                            st.metric("Category", doc['category'])
                        with col2:
                            st.metric("Relevance", f"{doc['relevance']:.2%}")

                        st.markdown("**Content Preview:**")
                        st.markdown(f"> {doc['content'][:300]}...")

    # Info section
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.markdown("""
    **Hospital Knowledge Base Assistant**

    A RAG (Retrieval-Augmented Generation) application that:
    - Retrieves relevant hospital policies and guidelines
    - Generates accurate answers using LLMs
    - Cites sources for transparency

    **Setup Required:**
    1. Run `python ingest.py` to process PDFs
    2. Add API key to `.streamlit/secrets.toml`
    """)

    # Stats
    if st.session_state.initialized:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Knowledge Base Stats")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Total Chunks", len(assistant.chunks))
        with col2:
            st.metric("Dimensions", 384)

if __name__ == "__main__":
    main()
