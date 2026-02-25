# RLHF Preference Collector

A Streamlit app that generates two responses from the same language model for a given prompt, lets a human annotator pick the better response (or mark a tie), and exports the data as `{prompt, chosen, rejected}` pairs for downstream RLHF / DPO training.

## Features

- **Dual generation** – two independent completions from the same model & settings
- **Locked responses** – once generated, responses stay fixed until you vote or skip
- **Preference recording** – Prefer A, Prefer B, Tie, or Skip
- **Generation metadata** – model name, token counts, latency, finish reason
- **Export formats** – JSONL (training-ready), full JSON, CSV
- **Supabase integration** (optional) – every preference is persisted to a Postgres table in real time; falls back to session-local storage when Supabase is not configured

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env and add your DASHSCOPE_API_KEY

# 3. Run the app
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a new app pointing to `app.py`.
3. Open **Settings → Secrets** and paste the contents of `.streamlit/secrets.toml.example`, filling in your real keys:
   ```toml
   DASHSCOPE_API_KEY = "sk-..."
   SUPABASE_URL = "https://your-project.supabase.co"
   SUPABASE_KEY = "your-anon-key"
   ```
4. The app reads secrets via `st.secrets` automatically — no `.env` file needed on the cloud.

## Supabase Setup (Optional)

1. Create a project at [supabase.com](https://supabase.com).
2. Open the **SQL Editor** and run `supabase_migration.sql` to create the `preferences` table.
3. Copy your project URL and anon key into `.env`:
   ```
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_KEY=your-anon-key
   ```
4. Restart the app — records will now be saved to Supabase automatically.

## Export Format

The **Download training JSONL** button produces one JSON object per line:

```json
{"prompt": "Explain quantum computing", "chosen": "...", "rejected": "..."}
```

Tie votes are excluded from the training export (but included in the full JSON / CSV downloads).
