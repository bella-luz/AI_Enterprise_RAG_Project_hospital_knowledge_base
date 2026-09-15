"""
Lightweight Hospital RAG Assistant - No FAISS dependency
Uses simple cosine similarity for retrieval instead of FAISS
"""

import os
import json
import logging
import streamlit as st
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# LLM imports
try:
    from groq import Groq
except ImportError:
    Groq = None

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
    page_title="Hospital KB Assistant",
    page_icon="🏥",
    layout="wide"
)

class LiteRAGAssistant:
    """Lightweight RAG without FAISS."""

    def __init__(self):
        self.embedder = None
        self.embeddings = None
        self.chunks = []
        self.metadata = None
        self.llm_provider = None
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

            # Load embeddings from disk if available
            embeddings_file = "data/embeddings.npy"
            if os.path.exists(embeddings_file):
                self.embeddings = np.load(embeddings_file)
            else:
                # Generate embeddings on first load
                if not self.embedder:
                    self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                self.embeddings = self.embedder.encode(self.chunks, convert_to_numpy=True)

            return len(self.chunks) > 0
        except Exception as e:
            logger.error(f"Error loading KB: {e}")
            return False

    def setup_llm(self, provider: str, api_key: str) -> bool:
        """Setup LLM provider."""
        try:
            if provider == "groq":
                if not Groq:
                    st.error("Groq not installed")
                    return False
                self.client = Groq(api_key=api_key)
                self.llm_provider = "groq"
                return True
            elif provider == "gemini":
                if not genai:
                    st.error("Google Generative AI not installed")
                    return False
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel("gemini-1.5-flash")
                self.llm_provider = "gemini"
                return True
        except Exception as e:
            logger.error(f"Error setting up LLM: {e}")
            return False
        return False

    def retrieve_documents(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve documents using cosine similarity."""
        if not self.embedder or self.embeddings is None:
            return []

        try:
            # Embed query
            query_embedding = self.embedder.encode(query, convert_to_numpy=True)

            # Calculate similarity
            similarities = cosine_similarity([query_embedding], self.embeddings)[0]

            # Get top-k
            top_indices = np.argsort(similarities)[::-1][:min(k, len(self.chunks))]

            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Relevance threshold
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
        """Generate answer using LLM."""
        if not self.client or not self.llm_provider:
            return "LLM not configured"

        context_text = "\n\n".join([
            f"Source: {doc['source']} ({doc['category']})\n{doc['content']}"
            for doc in context
        ])

        prompt = f"""You are a hospital knowledge base assistant.
Use the context below to answer accurately.

Context:
{context_text}

Question: {query}

Answer:"""

        try:
            if self.llm_provider == "groq":
                response = self.client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1024
                )
                return response.choices[0].message.content
            elif self.llm_provider == "gemini":
                response = self.client.generate_content(prompt)
                return response.text
        except Exception as e:
            logger.error(f"Error generating: {e}")
            return f"Error: {str(e)}"

def main():
    st.title("🏥 Hospital Knowledge Base Assistant")
    st.markdown("AI-powered hospital policy assistant with source citations")

    # Sidebar
    st.sidebar.title("Configuration")
    provider = st.sidebar.radio("LLM Provider", ["Groq", "Gemini"])

    # Get API key
    if provider == "Groq":
        api_key = st.secrets.get("GROQ_API_KEY", "")
    else:
        api_key = st.secrets.get("GEMINI_API_KEY", "")

    if not api_key:
        st.warning(f"⚠️ {provider} API key not found in secrets")
        st.stop()

    # Initialize
    if 'assistant' not in st.session_state:
        st.session_state.assistant = LiteRAGAssistant()
        st.session_state.ready = False

    assistant = st.session_state.assistant

    if not st.session_state.ready:
        with st.spinner("Loading knowledge base..."):
            if not assistant.load_knowledge_base():
                st.error("❌ Knowledge base not found. Run: python ingest.py")
                st.stop()

            provider_key = "groq" if provider == "Groq" else "gemini"
            if not assistant.setup_llm(provider_key, api_key):
                st.error(f"❌ Failed to setup {provider}")
                st.stop()

            st.session_state.ready = True

    # Search interface
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("Ask a question:", placeholder="e.g., What is the admission process?")
    with col2:
        search = st.button("🔍 Search")

    k = st.sidebar.slider("Results", 1, 10, 5)

    if search and query:
        with st.spinner("Searching..."):
            docs = assistant.retrieve_documents(query, k)

            if not docs:
                st.error("❌ No relevant documents found")
            else:
                with st.spinner(f"Generating answer..."):
                    answer = assistant.generate_answer(query, docs)

                st.markdown("---")
                st.subheader("📝 Answer")
                st.write(answer)

                st.markdown("---")
                st.subheader("📚 Sources")

                for i, doc in enumerate(docs, 1):
                    with st.expander(f"Source {i}: {doc['source']} ({doc['category']})"):
                        st.metric("Relevance", f"{doc['relevance']:.1%}")
                        st.write(doc['content'][:500] + "...")

    st.sidebar.markdown("---")
    st.sidebar.info(f"**Knowledge Base**: {len(assistant.chunks)} chunks loaded")

if __name__ == "__main__":
    main()
