import json
import os
import time
import uuid
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Supabase helpers (optional – gracefully degrades to local-only mode)
# ---------------------------------------------------------------------------

_supabase_client = None


def _get_supabase():
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        return None

    try:
        from supabase import create_client

        _supabase_client = create_client(url, key)
        return _supabase_client
    except Exception as exc:
        st.warning(f"Supabase connection failed: {exc}")
        return None


def save_to_supabase(record: dict) -> bool:
    client = _get_supabase()
    if client is None:
        return False
    try:
        client.table("preferences").insert(record).execute()
        return True
    except Exception as exc:
        st.warning(f"Supabase insert failed: {exc}")
        return False


def load_from_supabase() -> list[dict] | None:
    client = _get_supabase()
    if client is None:
        return None
    try:
        resp = client.table("preferences").select("*").order("created_at", desc=True).execute()
        return resp.data
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Qwen (DashScope) helpers – uses the OpenAI-compatible endpoint
# ---------------------------------------------------------------------------

DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

AVAILABLE_MODELS = [
    "qwen-turbo-latest",
    "qwen-plus-latest",
    "qwen-max-latest",
]


def get_openai_client() -> OpenAI | None:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE_URL)


def generate_response(
    client: OpenAI,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
) -> dict:
    """Generate a single completion and return the response text + metadata."""
    t0 = time.perf_counter()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    elapsed = time.perf_counter() - t0
    choice = completion.choices[0]
    return {
        "text": choice.message.content,
        "model": completion.model,
        "finish_reason": choice.finish_reason,
        "prompt_tokens": completion.usage.prompt_tokens,
        "completion_tokens": completion.usage.completion_tokens,
        "total_tokens": completion.usage.total_tokens,
        "latency_s": round(elapsed, 3),
    }


# ---------------------------------------------------------------------------
# Session-state initialization
# ---------------------------------------------------------------------------

def init_state():
    defaults = {
        "response_a": None,
        "response_b": None,
        "current_prompt": "",
        "generation_locked": False,
        "history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="RLHF Preference Collector", layout="wide")
    init_state()

    st.title("RLHF Preference Collector")
    st.caption("Generate two responses from the same model, pick the better one, and export preference pairs for training.")

    # ---- Sidebar: settings ------------------------------------------------
    with st.sidebar:
        st.header("Settings")
        model = st.selectbox("Model", AVAILABLE_MODELS, index=0)
        temperature = st.slider("Temperature", 0.0, 2.0, 1.0, 0.05)
        max_tokens = st.slider("Max tokens", 64, 4096, 1024, 64)
        system_prompt = st.text_area("System prompt", value="You are a helpful assistant.", height=100)

        st.divider()
        supabase_status = "Connected" if _get_supabase() else "Not configured"
        st.caption(f"Supabase: **{supabase_status}**")

        st.divider()
        st.header("Export")
        _render_export_section()

    # ---- Main area --------------------------------------------------------
    prompt = st.text_area("Enter your prompt", height=120, key="prompt_input")

    generate_disabled = not prompt.strip() or st.session_state.generation_locked
    if st.button("Generate Responses", type="primary", disabled=generate_disabled):
        client = get_openai_client()
        if client is None:
            st.error("Set `DASHSCOPE_API_KEY` in your `.env` file or environment.")
            return

        with st.spinner("Generating response A…"):
            st.session_state.response_a = generate_response(
                client, prompt, model, temperature, max_tokens, system_prompt,
            )
        with st.spinner("Generating response B…"):
            st.session_state.response_b = generate_response(
                client, prompt, model, temperature, max_tokens, system_prompt,
            )
        st.session_state.current_prompt = prompt
        st.session_state.generation_locked = True
        st.rerun()

    # ---- Display responses side-by-side -----------------------------------
    if st.session_state.generation_locked:
        ra = st.session_state.response_a
        rb = st.session_state.response_b

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Response A")
            st.markdown(ra["text"])
            with st.expander("Metadata"):
                st.json({k: v for k, v in ra.items() if k != "text"})
        with col_b:
            st.subheader("Response B")
            st.markdown(rb["text"])
            with st.expander("Metadata"):
                st.json({k: v for k, v in rb.items() if k != "text"})

        st.divider()
        st.subheader("Which response is better?")

        btn_cols = st.columns([1, 1, 1, 1])
        with btn_cols[0]:
            if st.button("Prefer A", use_container_width=True):
                _record_preference("response_a")
        with btn_cols[1]:
            if st.button("Prefer B", use_container_width=True):
                _record_preference("response_b")
        with btn_cols[2]:
            if st.button("Tie", use_container_width=True):
                _record_preference("tie")
        with btn_cols[3]:
            if st.button("Skip / Discard", use_container_width=True):
                _reset_generation()
                st.rerun()

    # ---- History table ----------------------------------------------------
    all_records = _get_all_records()
    if all_records:
        st.divider()
        st.subheader(f"Preference History ({len(all_records)} records)")
        df = pd.DataFrame(all_records)
        display_cols = [c for c in ["created_at", "prompt", "preference", "model"] if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True)


# ---------------------------------------------------------------------------
# Recording & export helpers
# ---------------------------------------------------------------------------

def _record_preference(preference: str):
    ra = st.session_state.response_a
    rb = st.session_state.response_b

    if preference == "response_a":
        chosen, rejected = ra["text"], rb["text"]
    elif preference == "response_b":
        chosen, rejected = rb["text"], ra["text"]
    else:
        chosen, rejected = ra["text"], rb["text"]

    record = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt": st.session_state.current_prompt,
        "response_a": ra["text"],
        "response_b": rb["text"],
        "chosen": chosen,
        "rejected": rejected,
        "preference": preference,
        "model": ra["model"],
        "temperature": ra.get("temperature"),
        "meta_a": json.dumps({k: v for k, v in ra.items() if k != "text"}),
        "meta_b": json.dumps({k: v for k, v in rb.items() if k != "text"}),
    }

    saved = save_to_supabase(record)
    st.session_state.history.append(record)

    if saved:
        st.toast("Saved to Supabase", icon="✅")
    else:
        st.toast("Saved locally (Supabase not configured)", icon="💾")

    _reset_generation()
    st.rerun()


def _reset_generation():
    st.session_state.response_a = None
    st.session_state.response_b = None
    st.session_state.current_prompt = ""
    st.session_state.generation_locked = False


def _get_all_records() -> list[dict]:
    """Merge Supabase records (if available) with local session history."""
    remote = load_from_supabase()
    if remote is not None:
        local_ids = {r["id"] for r in remote}
        merged = list(remote)
        for r in st.session_state.history:
            if r["id"] not in local_ids:
                merged.append(r)
        return merged
    return list(st.session_state.history)


def _render_export_section():
    records = _get_all_records()
    if not records:
        st.info("No preference data yet.")
        return

    st.caption(f"{len(records)} records available")

    # DPO / training format: {prompt, chosen, rejected}
    training_data = [
        {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
        for r in records
        if r.get("preference") != "tie"
    ]

    if training_data:
        jsonl = "\n".join(json.dumps(row, ensure_ascii=False) for row in training_data)
        st.download_button(
            "Download training JSONL",
            data=jsonl,
            file_name="preferences_training.jsonl",
            mime="application/jsonl",
        )

    full_json = json.dumps(records, indent=2, ensure_ascii=False)
    st.download_button(
        "Download full JSON",
        data=full_json,
        file_name="preferences_full.json",
        mime="application/json",
    )

    df = pd.DataFrame(records)
    csv = df.to_csv(index=False)
    st.download_button(
        "Download CSV",
        data=csv,
        file_name="preferences.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
