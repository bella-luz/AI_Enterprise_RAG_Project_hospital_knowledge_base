# 🚀 QUICKSTART - 5 Minutes to Live App

## ✅ What's Already Done
- ✅ 9 sample hospital policy PDFs created
- ✅ FAISS vector index generated (14 embeddings)
- ✅ Streamlit app built & tested
- ✅ GitHub repo structure ready
- ✅ Ingestion pipeline tested & working

## 🎯 Next 5 Steps (That's It!)

### 1️⃣ Create GitHub Repo (1 min)
```
https://github.com/new
- Name: hospital_knowledge_base
- Description: RAG hospital knowledge base
- Public
- Create (don't init with README)
```

### 2️⃣ Push Code (1 min)
```bash
cd "C:\Users\dandy\Desktop\Iqra\AI training\Enterprise RAG"
git push -u origin main
```

### 3️⃣ Get API Key (1 min)

**PICK ONE:**

**🚀 Groq (Recommended)**
- Go: https://console.groq.com/
- Sign up
- Generate API key
- Copy it

**OR**

**🔵 Gemini**
- Go: https://makersuite.google.com/app/apikey
- Sign in
- Create key
- Copy it

### 4️⃣ Deploy to Streamlit (2 min)
```
1. https://share.streamlit.io
2. "New app"
3. Connect GitHub
4. Select: bella-luz/hospital_knowledge_base
5. Main file: app.py
6. Deploy
```

### 5️⃣ Add API Key to Streamlit (1 min)
```
1. Go to: https://your-app.streamlit.app (wait for it to load)
2. Click "Manage app" (top right)
3. Settings → Secrets
4. Paste:

For Groq:
GROQ_API_KEY = "your_key_here"

For Gemini:
GEMINI_API_KEY = "your_key_here"

5. Save
```

**🎉 LIVE IN 5 MINUTES!**

---

## ✨ Try It Out

Ask questions like:
- "What is the hospital admission process?"
- "What are infection control guidelines?"
- "What is a code blue procedure?"

---

## 📖 Need More Help?

- **Detailed setup**: Read `DEPLOYMENT_GUIDE.md`
- **Git issues**: Read `GITHUB_SETUP.md`
- **How it works**: Read `README.md`
- **Local testing**: Run `streamlit run app.py` after setting up `.streamlit/secrets.toml`

---

## 🛠️ File Summary

| File | Purpose |
|------|---------|
| `app.py` | Streamlit web app (no changes needed) |
| `ingest.py` | PDF processing pipeline (ready to use) |
| `requirements.txt` | Dependencies (already versioned) |
| `hospital_knowledge_base/` | 9 sample policies (organized by category) |
| `data/faiss_index/` | Generated vector index |
| `data/metadata.json` | Source tracking |
| `.streamlit/secrets.toml.example` | Template (copy & fill) |

---

## ✅ Checklist

- [ ] Create GitHub repo
- [ ] Push code: `git push -u origin main`
- [ ] Get Groq or Gemini API key
- [ ] Deploy to Streamlit Cloud
- [ ] Add API key to Streamlit secrets
- [ ] Test with sample query
- [ ] Share link with team!

---

**Ready? Start with Step 1! 🚀**
