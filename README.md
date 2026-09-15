# 🏥 Hospital Knowledge Base RAG Assistant

A Retrieval-Augmented Generation (RAG) application that leverages hospital policies and guidelines to answer questions accurately with cited sources.

## Features

✨ **Core Features:**
- **PDF Knowledge Base**: Organized hospital policies and clinical guidelines
- **Vector Embeddings**: Uses HuggingFace embeddings for semantic search
- **FAISS Indexing**: Fast similarity search across knowledge base
- **Multi-LLM Support**: Works with Groq or Google Generative AI
- **Source Citation**: Provides references for all answers
- **Streamlit UI**: Interactive web interface

## Project Structure

```
hospital_knowledge_base/
├── admin_policies/          # Administrative policies
├── clinical_guidelines/      # Clinical and medical guidelines
├── hr_policies/             # Human resources policies
├── patient_care/            # Patient care protocols
└── emergency_protocols/      # Emergency response procedures

data/
├── faiss_index/             # FAISS vector index (created by ingest.py)
└── metadata.json            # Chunk metadata and sources
```

## Setup Instructions

### 1. Prerequisites

- Python 3.8+
- Git
- API key from Groq or Google Generative AI

### 2. Clone Repository

```bash
git clone https://github.com/bella-luz/hospital_knowledge_base.git
cd hospital_knowledge_base
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate Embeddings and FAISS Index

Process hospital PDFs and create the vector index:

```bash
python ingest.py
```

This will:
- Extract text from all hospital PDFs
- Split text into chunks (1000 tokens with 100 token overlap)
- Create embeddings using sentence-transformers
- Build FAISS index for fast retrieval
- Save metadata for source tracking

Output: `data/faiss_index/` and `data/metadata.json`

### 5. Configure API Keys

Create `.streamlit/secrets.toml`:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml` and add your API key:

```toml
# For Groq
GROQ_API_KEY = "your_actual_api_key_here"

# For Google Generative AI
GEMINI_API_KEY = "your_actual_api_key_here"
```

**⚠️ Never commit secrets.toml to git - it's in .gitignore**

### 6. Run Streamlit Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Getting API Keys

### Groq API Key
1. Go to https://console.groq.com/
2. Sign up or log in
3. Create new API key
4. Copy and paste into secrets.toml

### Google Generative AI API Key
1. Go to https://makersuite.google.com/app/apikey
2. Create new API key
3. Copy and paste into secrets.toml

## Usage

1. **Ask Questions**: Type your question in the search box
2. **Select Provider**: Choose Groq or Gemini from sidebar
3. **Adjust Results**: Set number of reference documents (1-10)
4. **Review Answer**: See AI-generated answer with source citations
5. **Check Sources**: Click on source tabs to view full context

## Example Queries

- "What is the hospital admission process?"
- "What are the infection control guidelines?"
- "What should I do in case of a code blue?"
- "What are the patient rights?"
- "What is the medication administration protocol?"

## How RAG Works

1. **Retrieval**: Query is converted to embeddings, searched in FAISS index
2. **Context**: Top-K relevant documents are retrieved (default K=5)
3. **Generation**: LLM generates answer using retrieved context
4. **Citation**: Sources are displayed for transparency and verification

## Performance Considerations

- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- **Chunking**: 1000 tokens per chunk with 100 token overlap
- **Index Type**: FAISS IndexFlatL2 for exact similarity search
- **Inference**: ~2-5 seconds per query (including LLM response)

## Troubleshooting

### Issue: "FAISS index not found"
**Solution**: Run `python ingest.py` first to create the index

### Issue: API key error
**Solution**: 
- Ensure `.streamlit/secrets.toml` exists (not `.example`)
- Verify API key is correct
- Restart Streamlit: `Ctrl+C` then `streamlit run app.py`

### Issue: Slow responses
**Solution**:
- Reduce number of retrieved documents
- Use GPU if available (modify ingest.py embeddings device)
- Check internet connection for LLM API calls

### Issue: Poor answer quality
**Solution**:
- Verify retrieved documents are relevant
- Add more specific PDFs to knowledge base
- Re-run ingest.py after adding new documents

## Deployment to Streamlit Cloud

1. Push repository to GitHub (already done)
2. Go to https://share.streamlit.io
3. Deploy from GitHub
4. Add secrets in Streamlit Cloud settings:
   - Go to app settings → Secrets
   - Add GROQ_API_KEY or GEMINI_API_KEY

## Adding More Documents

1. Add PDF files to appropriate subdirectory in `hospital_knowledge_base/`
2. Run `python ingest.py` again
3. Restart Streamlit app
4. New documents will be available for queries

## License

MIT License - Feel free to use and modify

## Support

For issues or questions:
- Check troubleshooting section above
- Review GitHub issues
- Check Streamlit documentation

---

**Built with ❤️ using LangChain, FAISS, and Streamlit**
