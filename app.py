# app/app.py
"""
Redesigned Streamlit interface for "Equity Research News Tool".
- Clean layout with a left control panel and a main content area.
- Uses module helpers from langchain_config exactly as before.
- Everything is copy-paste ready — replace your existing app.py with this file.
"""

import streamlit as st
from dotenv import load_dotenv
import logging
from typing import Optional, List, Dict
from datetime import datetime

# load .env (for API keys)
load_dotenv()

# Import helpers from langchain_config (keeps your existing backend logic)
from langchain_config import (
    get_summary,
    get_summary_cached_module,
    get_news_articles,
    summarize_articles_llm,
    estimate_tokens,
    clear_module_cache,
)

# ------------------ Page config & logger ------------------------
st.set_page_config(page_title="Equity Research — News & Summaries", layout="wide")
logger = logging.getLogger("equity_research_app")
logging.basicConfig(level=logging.INFO)

# ------------------ Helper functions ----------------------------
@st.cache_data(ttl=60 * 60 * 24)
def get_summary_cached_ui(query: str, max_articles: int, ttl_override: Optional[int] = None) -> str:
    """Wrapper to call your module-level cached function. Kept separate for st.cache_data control."""
    return get_summary_cached_module(query, max_articles)


def pretty_article_card(a: Dict) -> str:
    """Return a small markdown snippet for a single article."""
    title = a.get("title") or "<no title>"
    src = a.get("source", {}).get("name") if a.get("source") else a.get("source")
    url = a.get("url")
    published = a.get("publishedAt") or a.get("published") or ""
    if published:
        try:
            published = datetime.fromisoformat(published.replace("Z", "+00:00")).strftime("%Y-%m-%d")
        except Exception:
            pass
    desc = a.get("description") or ""
    # Create compact markdown — Streamlit will render it nicely in columns
    if url:
        return f"**[{title}]({url})**  \n*{src} — {published}*  \n{desc}"
    return f"**{title}**  \n*{src} — {published}*  \n{desc}"


# ------------------ Sidebar controls (left) ---------------------
with st.sidebar:
    st.title("Controls")

    # Main query input
    query = st.text_input("Query (company, sector, event)", value="", key="query_input")

    # Article & date controls
    max_articles = st.slider("Max articles", min_value=5, max_value=100, value=20, step=5)
    date_from = st.date_input("From (optional)", value=None)
    date_to = st.date_input("To (optional)", value=None)

    st.markdown("---")

    # Cache TTL control — this only affects Streamlit cached UI wrapper
    cache_ttl_minutes = st.number_input(
        "Cache TTL (minutes)", min_value=1, max_value=1440, value=60
    )

    st.write("Tip: keep `max articles` small while developing to save tokens.")
    st.markdown("---")

    # Buttons
    run_button = st.button("Run — Fetch & Summarize")
    st.button("Clear module & streamlit caches", key="clear_all") and (st.cache_data.clear(), clear_module_cache(), st.success("Cleared caches."))

    st.markdown("---")
    st.caption("This app uses NewsAPI (developer key limits apply) and OpenAI for summarization — watch token usage.")


# ------------------ Main content area ---------------------------
# Top row: status, token estimate, history quick links
col_left, col_right = st.columns([3, 1])
with col_left:
    st.header("Equity Research — News Summaries")
    st.write("Enter a query on the left, then click **Run** to fetch articles and generate a concise LLM summary.")

with col_right:
    # Compact history selector if available
    if "history" in st.session_state and st.session_state.history:
        if st.selectbox("Recent queries", options=[h['query'] for h in st.session_state.history[:10]]):
            pass


# Run flow (button-driven)
if run_button:
    if not query or not query.strip():
        st.warning("Please enter a non-empty query in the sidebar.")
    else:
        # Show a top-level spinner and perform fetch + summarize
        try:
            with st.spinner("Fetching articles from NewsAPI..."):
                # Keep backwards compatibility: pass max_articles; if you add date support to get_news_articles, include dates
                articles = get_news_articles(query, max_articles=max_articles)
        except Exception as e:
            st.error(f"Failed to fetch articles: {e}")
            logger.exception("NewsAPI fetch failed")
            articles = []

        if not articles:
            st.info("No articles found for this query — try changing the query or increasing 'Max articles'.")
        else:
            # Top metrics
            st.markdown("### Fetched articles")
            st.write(f"Found **{len(articles)}** articles for **{query}**")

            # Show first 30 articles in two-column cards for quick scanning
            cards_per_row = 2
            for i in range(0, min(len(articles), 30), cards_per_row):
                cols = st.columns(cards_per_row)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx >= len(articles):
                        break
                    with col:
                        st.markdown(pretty_article_card(articles[idx]))

            # Show token estimate for concatenated text
            concat_text = "\n\n".join(
                [(a.get("title") or "") + " — " + (a.get("description") or "") for a in articles]
            )
            try:
                tokens_est = estimate_tokens(concat_text)
                st.info(f"Estimated tokens for LLM input (approx): {tokens_est}")
                if tokens_est > 8000:
                    st.warning("High token estimate — reduce `max articles` to control cost.")
            except Exception:
                logger.exception("Token estimation failed")

            # Summarize with caching wrapper
            try:
                with st.spinner("Generating summary with LLM (cached)..."):
                    # Note: st.cache_data TTL is applied on the wrapper; we pass max_articles so cache key differs by that
                    summary = get_summary_cached_ui(query, max_articles)
            except Exception as e:
                st.error(f"Error during summarization: {e}")
                logger.exception("Summarization error")
                summary = None

            if summary:
                st.markdown("---")
                st.markdown("## Summary")
                st.write(summary)

                st.download_button("Download summary (txt)", data=summary, file_name="summary.txt")

                # Save into session history for quick access
                if "history" not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.insert(0, {"query": query, "summary": summary})

# Sidebar: show history and let user expand
if "history" in st.session_state and st.session_state.history:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Recent queries")
    for i, h in enumerate(st.session_state.history[:10]):
        if st.sidebar.button(f"Load: {h['query']}", key=f"load_{i}"):
            # Load into main area by setting query and pre-filling a small preview
            st.session_state['query_input'] = h['query']
            st.experimental_rerun()

# Developer tools & debug (collapsible)
with st.expander("Developer tools & debug"):
    st.write("Inspect or clear caches used by Streamlit and the module-level cache.")
    if st.button("Clear Streamlit cache only", key="clear_streamlit"):
        try:
            st.cache_data.clear()
            st.success("Cleared Streamlit cache.")
        except Exception:
            st.error("Failed to clear Streamlit cache.")
    st.write("Module-level cached function available: get_summary_cached_module(query, max_articles)")

# Footer / notes
st.markdown("---")
st.markdown(
    "**Notes:** This tool uses NewsAPI (subject to developer key limits) and an LLM for summarization. "
    "Keep `max articles` low during iteration to reduce API calls and token costs."
)

# Small helper: show session state for debugging when a special flag is present
if st.experimental_get_query_params().get("debug"):
    st.write("Session state:", dict(st.session_state))
