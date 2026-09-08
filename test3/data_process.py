"""Chinese FAQ QA preprocessing & retrieval.

Functions expected by faq_test.py:
    read_corpus(path)             -> list[str]      (raw lines, stripped)
    get_question_list(questions)  -> list[list[str]] (jieba tokens per question)
    input_question_process(qs, q) -> list[list[str]] (qs with the tokenized user input appended)
    ques_idx_cosine_sim(q, c)     -> int | None     (best-matching index in c, or None)
"""

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# TfidfVectorizer's default token pattern requires 2+ word chars, which drops
# many single Chinese characters. We already pre-tokenize with jieba and join
# tokens with spaces, so we treat each space-separated token as one feature.
_TOKEN_PATTERN = r"(?u)\S+"

# Tokens made entirely of punctuation / whitespace / pure digits don't help retrieval.
_PUNCT_CHARS = set("，。！？、；：""''《》<>（）()【】[]·…—-.!?;:,'\"`~@#$%^&*+=/\\|")


def read_corpus(path):
    """Read a UTF-8 text file (one document per line) and return stripped strings."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _tokenize(text):
    """Precise-mode jieba tokenization, dropping pure-punctuation tokens."""
    if not text:
        return []
    tokens = []
    for tok in jieba.cut(text):
        tok = tok.strip()
        if not tok:
            continue
        if all(ch in _PUNCT_CHARS or not ch.isalnum() for ch in tok):
            # pure punctuation or whitespace-only fragment
            continue
        tokens.append(tok)
    return tokens


def get_question_list(questions):
    """Tokenize every question into a list of tokens."""
    return [_tokenize(q) for q in questions]


def input_question_process(questions_list, input_ques):
    """Append the tokenized user input as the last element of a copy of questions_list.

    The caller uses questions_list[-1] as the query and questions_list[0:-1] as the corpus.
    """
    combined = [list(q) for q in questions_list]
    combined.append(_tokenize(input_ques))
    return combined


def ques_idx_cosine_sim(query_tokens, corpus_tokens_list):
    """Return the index of the corpus question most similar to the query, or None.

    Uses TF-IDF + cosine similarity over jieba tokens. Returns None when the corpus
    is empty, the query has no usable tokens, or no positive similarity is found.
    """
    if not corpus_tokens_list:
        return None

    query_text = " ".join(query_tokens).strip()
    if not query_text:
        return None

    corpus_texts = [" ".join(toks) for toks in corpus_tokens_list]
    # Drop any corpus entries that became empty after tokenization, remembering
    # their original indices so the returned index still points into the caller's list.
    kept = [(i, t) for i, t in enumerate(corpus_texts) if t.strip()]
    if not kept:
        return None
    kept_indices, kept_texts = zip(*kept)

    try:
        vectorizer = TfidfVectorizer(token_pattern=_TOKEN_PATTERN)
        tfidf = vectorizer.fit_transform(list(kept_texts) + [query_text])
    except ValueError:
        return None

    corpus_vec = tfidf[:-1]
    query_vec = tfidf[-1]
    sims = cosine_similarity(query_vec, corpus_vec).flatten()

    if sims.size == 0 or sims.max() <= 0:
        return None
    return int(kept_indices[sims.argmax()])
