# 🚀 Deployment Guide — Sikandar's AI Chatbot

## Final Project Structure

```
portfolio-chatbot/          ← root folder (name it anything)
├── app.py                  ← Streamlit app (main entry point)
├── build_vectordb.py       ← Run locally ONCE to build the DB
├── requirements.txt        ← Python dependencies
├── .gitignore
├── .streamlit/
│   └── secrets.toml        ← Local secrets ONLY (never commit)
└── chroma_db/              ← Built by build_vectordb.py — COMMIT THIS
    ├── chroma.sqlite3
    └── ...
```

---

## Step 1 — Build the Vector Database Locally

Run this ONCE on your machine before deploying.
This creates the `chroma_db/` folder that Streamlit Cloud will use.

```bash
# Install dependencies
pip install -r requirements.txt

# Add your tokens to a local .env file:
# HUGGINGFACEHUB_API_TOKEN=hf_xxxxx
# GITHUB_TOKEN=ghp_xxxxx

# Build the DB
python build_vectordb.py
```

✅ You should see: "Vector DB ready with X indexed chunks."

---

## Step 2 — Set Up GitHub Repository

```bash
git init
git add .
# IMPORTANT: make sure chroma_db/ is being added (it must NOT be in .gitignore)
git commit -m "Initial commit — chatbot + vector DB"
git remote add origin https://github.com/YOUR_USERNAME/portfolio-chatbot.git
git push -u origin main
```

> ⚠️ Verify `chroma_db/` is committed:
> `git ls-files | grep chroma_db` — you should see files listed.

---

## Step 3 — Deploy on Streamlit Cloud (Free)

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **"New app"**.
3. Select your repository and branch (`main`).
4. Set **Main file path** to: `app.py`
5. Click **"Advanced settings"** → **Secrets** and paste:

```toml
HF_TOKEN = "hf_your_actual_token_here"
HUGGINGFACEHUB_API_TOKEN = "hf_your_actual_token_here"
```

6. Click **Deploy**.

⏱️ First deploy takes 3–5 minutes (installing dependencies).
Your app will be live at: `https://your-app-name.streamlit.app`

---

## Step 4 — Updating the Chatbot Later

### If you update the biography or add new projects:
```bash
# Re-run the build script locally
python build_vectordb.py

# Push the updated DB
git add chroma_db/
git commit -m "Rebuild vector DB with updated content"
git push
```
Streamlit Cloud auto-redeploys on every push.

### If you only change the UI or prompts:
```bash
git add app.py
git commit -m "Update UI / prompt"
git push
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `HF_TOKEN not found` | Add it in Streamlit Cloud → App Settings → Secrets |
| `chroma_db not found` | Make sure you committed the `chroma_db/` folder |
| Slow responses (30s+) | Normal — HuggingFace free tier has queue delays |
| `ModuleNotFoundError` | Check `requirements.txt` has all packages |
| App crashes on cold start | Streamlit Cloud spins down free apps; first load is slow |

---

## Optional: Custom Domain

Streamlit Cloud free tier gives you a `.streamlit.app` URL.
For a custom domain (like `chat.sfs.surge.sh`), upgrade to Streamlit Teams
or use a reverse proxy (Cloudflare Tunnel, etc.).
