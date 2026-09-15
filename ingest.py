"""
Ingest hospital knowledge base PDFs into FAISS vector store.
Reads PDFs, chunks text, creates embeddings, and saves FAISS index.
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple
import numpy as np
import faiss
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFIngestor:
    """Ingest hospital PDFs and create FAISS index."""

    def __init__(self, pdf_dir: str = "hospital_knowledge_base", output_dir: str = "data"):
        self.pdf_dir = pdf_dir
        self.output_dir = output_dir
        self.index_path = os.path.join(output_dir, "faiss_index")
        self.metadata_path = os.path.join(output_dir, "metadata.json")

        # Create output directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.index_path, exist_ok=True)

        # Initialize embeddings model (lightweight, offline-capable)
        logger.info("Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        self.embedding_dim = 384  # Dimension of all-MiniLM-L6-v2

    def extract_pdf_content(self, pdf_path: str) -> Tuple[str, str]:
        """Extract text content and metadata from PDF."""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"

            filename = os.path.basename(pdf_path)
            return text, filename
        except Exception as e:
            logger.error(f"Error reading {pdf_path}: {e}")
            return "", ""

    def get_all_pdfs(self) -> List[str]:
        """Recursively get all PDF files from directory."""
        pdfs = []
        for root, dirs, files in os.walk(self.pdf_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdfs.append(os.path.join(root, file))
        return pdfs

    def chunk_texts(self, texts: List[str], chunk_size: int = 1000,
                   chunk_overlap: int = 100) -> Tuple[List[str], List[dict]]:
        """Split texts into chunks and preserve metadata."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = []
        metadata = []

        for text_item in texts:
            if not text_item.get('content'):
                continue

            split_chunks = splitter.split_text(text_item['content'])
            for chunk in split_chunks:
                if chunk.strip():
                    chunks.append(chunk)
                    metadata.append({
                        'source': text_item['filename'],
                        'category': text_item['category'],
                        'chunk_index': len([m for m in metadata if m['source'] == text_item['filename']])
                    })

        logger.info(f"Created {len(chunks)} chunks from {len(texts)} documents")
        return chunks, metadata

    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Create embeddings for text chunks."""
        logger.info(f"Creating embeddings for {len(chunks)} chunks...")
        embeddings = self.embeddings.embed_documents(chunks)
        return np.array(embeddings).astype('float32')

    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        """Create FAISS index from embeddings."""
        logger.info(f"Creating FAISS index with {embeddings.shape[0]} vectors...")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index

    def ingest(self):
        """Main ingestion pipeline."""
        logger.info(f"Starting ingestion from {self.pdf_dir}")

        # Step 1: Extract PDFs
        pdf_files = self.get_all_pdfs()
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.pdf_dir}")
            return

        logger.info(f"Found {len(pdf_files)} PDF files")

        # Step 2: Extract content with metadata
        texts = []
        for pdf_path in pdf_files:
            content, filename = self.extract_pdf_content(pdf_path)
            if content:
                # Extract category from path
                category = Path(pdf_path).parent.name
                texts.append({
                    'content': content,
                    'filename': filename,
                    'category': category,
                    'path': pdf_path
                })
                logger.info(f"Extracted: {filename} from {category}")

        if not texts:
            logger.warning("No text content extracted from PDFs")
            return

        # Step 3: Chunk texts
        chunks, metadata = self.chunk_texts(texts)

        # Step 4: Create embeddings
        embeddings = self.create_embeddings(chunks)

        # Step 5: Create FAISS index
        faiss_index = self.create_faiss_index(embeddings)

        # Step 6: Save FAISS index
        faiss.write_index(faiss_index, os.path.join(self.index_path, "index.faiss"))
        logger.info(f"Saved FAISS index to {self.index_path}")

        # Step 7: Save metadata
        metadata_with_chunks = []
        for i, (chunk, meta) in enumerate(zip(chunks, metadata)):
            metadata_with_chunks.append({
                'id': i,
                'chunk': chunk[:100] + "..." if len(chunk) > 100 else chunk,
                'source': meta['source'],
                'category': meta['category'],
                'chunk_index': meta['chunk_index']
            })

        with open(self.metadata_path, 'w') as f:
            json.dump(metadata_with_chunks, f, indent=2)
        logger.info(f"Saved metadata to {self.metadata_path}")

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"Ingestion Summary:")
        logger.info(f"  - PDF files processed: {len(pdf_files)}")
        logger.info(f"  - Documents extracted: {len(texts)}")
        logger.info(f"  - Text chunks created: {len(chunks)}")
        logger.info(f"  - Embedding dimension: {embeddings.shape[1]}")
        logger.info(f"  - FAISS index created: {self.index_path}")
        logger.info(f"{'='*60}\n")

def main():
    """Run the ingestion pipeline."""
    ingestor = PDFIngestor()
    ingestor.ingest()

if __name__ == "__main__":
    main()
