import os
import logging
from functools import lru_cache
from typing import List, Optional

from dotenv import load_dotenv
from newsapi import NewsApiClient
from tenacity import retry, wait_exponential, stop_after_attempt

# Import ChatOpenAI from the partner package to avoid langchain_community issues
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load environment
load_dotenv()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("langchain_config")

# Config / Keys
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))

if not OPENAI_KEY:
    logger.warning("OPENAI_API_KEY not found in environment. ChatOpenAI may not work until set.")
if not NEWSAPI_KEY:
    logger.warning("NEWSAPI_KEY not found in environment. NewsAPI calls will fail until set.")

# Clients
# ChatOpenAI typically reads key from env; passing explicitly for clarity
openai = ChatOpenAI(temperature=OPENAI_TEMPERATURE, openai_api_key=OPENAI_KEY)
newsapi = NewsApiClient(api_key=NEWSAPI_KEY)

# Prompt used for final summarization
PROMPT_TEMPLATE = """
You are an AI assistant helping an equity research analyst. Given the
query and article text below, write a concise, actionable summary with:
- 3 short bullet headlines (one line each)
- the potential impact for investors (1-2 short paragraphs)
- suggested next steps / follow-ups (2-3 action items)
Query: {query}
Articles: {articles}
"""
prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["query", "articles"])
llm_chain = LLMChain(llm=openai, prompt=prompt)

# Token estimation
try:
    import tiktoken

    def estimate_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))

    logger.info("tiktoken available: using for token estimation.")
except Exception:
    def estimate_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
        # simple heuristic: 1 token ≈ 4 characters
        return max(1, int(len(text) / 4))

    logger.info("tiktoken not available: using heuristic token estimator.")


# NewsAPI fetch with retry/backoff
@retry(wait=wait_exponential(multiplier=1, min=2, max=20), stop=stop_after_attempt(3))
def _fetch_page(
    query: str,
    page: int = 1,
    page_size: int = 20,
    from_param: Optional[str] = None,
    to_param: Optional[str] = None,
):
    """
    Low-level NewsAPI get_everything call with retries.
    """
    logger.debug("Calling NewsAPI: query=%s page=%d page_size=%d from=%s to=%s",
                 query, page, page_size, from_param, to_param)
    # NewsApiClient.get_everything uses 'from_param' and 'to' as parameter names
    return newsapi.get_everything(
        q=query,
        language="en",
        sort_by="relevancy",
        page=page,
        page_size=page_size,
        from_param=from_param,
        to=to_param,
    )


def get_news_articles(query: str, max_articles: int = 40) -> List[dict]:
    """
    Fetch up to max_articles from NewsAPI (handles pagination).
    Returns list of article dicts.
    """
    logger.info("Fetching up to %d articles for query=%s", max_articles, query)
    page_size = 20
    max_pages = (max_articles + page_size - 1) // page_size
    articles: List[dict] = []
    try:
        for p in range(1, max_pages + 1):
            res = _fetch_page(query, page=p, page_size=page_size)
            if res.get("status") != "ok":
                logger.warning("NewsAPI returned non-ok status: %s", res.get("status"))
                break
            page_articles = res.get("articles", []) or []
            if not page_articles:
                logger.debug("No more articles returned on page %d", p)
                break
            articles.extend(page_articles)
            if len(articles) >= max_articles:
                break
    except Exception as e:
        logger.exception("Error fetching articles from NewsAPI: %s", e)
        raise
    logger.info("Fetched %d articles (returning %d)", len(articles), min(len(articles), max_articles))
    return articles[:max_articles]


# Normalization & chunking
def normalize_article_text(a: dict) -> str:
    """
    Compose a compact text for an article: title — description — content
    """
    parts = []
    title = a.get("title") or ""
    description = a.get("description") or ""
    content = a.get("content") or ""
    if title:
        parts.append(title.strip())
    if description:
        parts.append(description.strip())
    if content:
        parts.append(content.strip())
    return " — ".join([p for p in parts if p])


def chunk_texts(texts: List[str], max_chars: int = 2500) -> List[str]:
    """
    Greedy concatenation of texts into chunks with length <= max_chars (approx).
    Returns list of chunk strings.
    """
    chunks: List[str] = []
    cur = ""
    for t in texts:
        if not t:
            continue
        # if adding t would exceed max_chars, finalize current chunk (if non-empty)
        if len(cur) + len(t) + 2 > max_chars:
            if cur:
                chunks.append(cur)
            cur = t
        else:
            cur = (cur + "\n\n" + t) if cur else t
    if cur:
        chunks.append(cur)
    logger.debug("Built %d chunks (max_chars=%d)", len(chunks), max_chars)
    return chunks


# Summarization flow
def summarize_articles_llm(query: str, articles: List[dict], max_chunk_chars: int = 2500) -> str:
    """
    Summarize article texts by:
      1) normalize each article
      2) chunk texts into manageable sizes
      3) call the llm_chain for each chunk to get partial summaries
      4) combine partials into final summary via another LLM pass
    """
    logger.info("Starting summarization for query=%s with %d articles", query, len(articles))
    texts = [normalize_article_text(a) for a in articles]
    texts = [t for t in texts if t]
    if not texts:
        logger.info("No usable article texts found.")
        return "No usable articles found."

    chunks = chunk_texts(texts, max_chars=max_chunk_chars)

    # Estimate tokens (rough) to log potential cost
    estimated_tokens = sum(estimate_tokens(c) for c in chunks)
    logger.info("Estimated tokens for summarization (approx): %d", estimated_tokens)

    partials: List[str] = []
    for i, c in enumerate(chunks, start=1):
        try:
            logger.info("Summarizing chunk %d/%d (chars=%d)", i, len(chunks), len(c))
            # use predict(...) rather than run(dict) for consistent behavior
            partial_summary = llm_chain.predict(query=query, articles=c)
            partials.append(partial_summary)
        except Exception as e:
            logger.exception("LLM error on chunk %d: %s", i, e)
            partials.append(f"[Error summarizing chunk {i}: {e}]")

    # Combine partials into one final summary
    try:
        combine_prompt = PromptTemplate(
            template=(
                "You are an AI assistant. Combine the following partial summaries into one concise analyst-style "
                "summary. Keep it short and structured: bullets for key points, investor impact, and next steps.\n\n"
                "Partials:\n{partials}"
            ),
            input_variables=["partials"],
        )
        combine_chain = LLMChain(llm=openai, prompt=combine_prompt)
        final_summary = combine_chain.predict(partials="\n\n".join(partials))
        logger.info("Final summary created.")
    except Exception as e:
        logger.exception("Error combining partial summaries: %s", e)
        # fallback: join partials
        final_summary = "\n\n".join(partials)
    return final_summary


# Public helper with optional caching
@lru_cache(maxsize=128)
def get_summary_cached_module(query: str, max_articles: int = 40) -> str:
    """
    In-memory cached wrapper for get_summary (good during development).
    Use get_summary() for a non-cached call.
    """
    return get_summary(query, max_articles)


def clear_module_cache():
    """Clear the LRU cache used by get_summary_cached_module."""
    get_summary_cached_module.cache_clear()
    logger.info("Cleared module LRU cache.")


def get_summary(query: str, max_articles: int = 40) -> str:
    """
    Public function: fetch articles and return a final summary string.
    Raises exceptions on NewsAPI failure (so caller can handle).
    """
    if max_articles <= 0:
        raise ValueError("max_articles must be positive")
    logger.info("get_summary called: query=%s max_articles=%d", query, max_articles)

    try:
        articles = get_news_articles(query, max_articles=max_articles)
    except Exception as e:
        logger.exception("Failed to fetch articles for query=%s", query)
        # return an informative message rather than raising if you prefer:
        return f"Error fetching articles: {e}"

    # If no articles found, return early
    if not articles:
        return "No articles found for the query."

    # Summarize (this will call the LLM)
    summary = summarize_articles_llm(query, articles)
    return summary
