# GitHub Push Instructions

The local repository is ready to push, but you need to create the repository on GitHub first.

## Step 1: Create Repository on GitHub

1. Go to https://github.com/new
2. Enter repository name: **hospital_knowledge_base**
3. Add description: "RAG-based hospital knowledge base assistant with Streamlit UI"
4. Choose visibility: **Public** (recommended for portfolio)
5. Do NOT initialize with README (we have one already)
6. Click **Create repository**

## Step 2: Push Local Code

After creating the repository, run this command:

```bash
git push -u origin main
```

If you're prompted for authentication:
- **On Windows**: Use GitHub credential helper or create Personal Access Token
- **GitHub PAT Method**:
  1. Go to https://github.com/settings/tokens
  2. Click "Generate new token" → "Generate new token (classic)"
  3. Give it a name like "hospital-rag"
  4. Select scopes: `repo` (full control of private repositories)
  5. Generate and copy the token
  6. Paste token when prompted for password

## Step 3: Verify Upload

After pushing, visit: https://github.com/bella-luz/hospital_knowledge_base

You should see:
- ✅ All Python files (app.py, ingest.py)
- ✅ Folder structure (hospital_knowledge_base/, .streamlit/)
- ✅ README.md
- ✅ requirements.txt
- ✅ .gitignore (hiding secrets and PDFs)

## Next Steps

1. **Run ingestion locally**:
   ```bash
   python ingest.py
   ```

2. **Test Streamlit app**:
   ```bash
   streamlit run app.py
   ```

3. **Deploy to Streamlit Cloud**:
   - Go to https://share.streamlit.io
   - Connect GitHub account
   - Select `bella-luz/hospital_knowledge_base` repo
   - Main file path: `app.py`
   - Deploy!
   - Add API keys in Streamlit Cloud → App settings → Secrets

## Troubleshooting

### Push Error: "fatal: repository not found"
- Verify repository exists at https://github.com/bella-luz/hospital_knowledge_base
- Check repository name spelling

### Authentication Failed
- Use Personal Access Token (PAT) instead of password
- Check token has `repo` scope
- Ensure token hasn't expired

### Streamlit Deployment Issues
- Ensure `.streamlit/secrets.toml` exists locally but NOT in git
- Run `python ingest.py` before deployment
- Data folder will need to be created during Streamlit deployment

That's it! Your RAG assistant will be live on Streamlit Cloud.
