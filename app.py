# import streamlit as st
# import pandas as pd
# import spelling
# import re
# import json
# import os
# import traceback

# # This is to avoid streamlit always rerun
# @st.cache_resource
# def setup():
#     return spelling.setup()

# # try:
# #     lm, vocab, suggester = setup()
# # except Exception as e:
# #     st.error("❌ setup() crashed. Full traceback below:")
# #     st.code(traceback.format_exc())
# #     st.stop()


# if "analysis_result" not in st.session_state:
#     st.session_state.analysis_result = None

# if "last_input" not in st.session_state:
#     st.session_state.last_input = ""

# # Page config 

# st.set_page_config(
#     page_title="Spell & Grammar Checker",
#     page_icon="📝",
#     layout="centered")

# # Settings

# st.sidebar.title("⚙ Settings")

# check_spelling = st.sidebar.checkbox("Check Spelling", value=True)
# check_grammar = st.sidebar.checkbox("Check Grammar", value=True)
# show_suggestions = st.sidebar.checkbox("Show Suggestions", value=True)
# highlight_errors = st.sidebar.checkbox("Highlight Errors", value=True)

# # Header

# st.title("📝 Spell & Grammar Checker")
# st.caption("Spelling and Grammar Correction System")

# st.markdown("---")

# # Input

# st.subheader("Input Text")

# input_text = st.text_area(
#     "Enter your text below (max 500 characters):",
#     height=180,
#     max_chars=500,
#     placeholder="Type or paste your text here...")


# # Button, use to insert input 

# def run_analysis():
#     if input_text.strip():
#         lm, vocab, suggester = setup()
#         st.session_state.analysis_result = spelling.model(lm, vocab, suggester, paragraph=input_text, mode="c")
#         st.session_state.last_input = input_text


# st.button("🔍 Check Text", on_click=run_analysis)


# # Results part

# st.markdown("---")
# st.subheader("Results")

# result = st.session_state.analysis_result

# if not result:
#     st.info("Click **Check Text** to analyze your input.")
# else:
#     words = st.session_state.last_input.split()
#     char_count = len(st.session_state.last_input)

    
# # Text statistics
    
#     with st.expander("📊 Text Statistics", expanded=True):
#         sentences = [
#             s.strip()
#             for s in re.split(r"[.!?]+", st.session_state.last_input)
#             if s.strip()]

#         st.write(f"**Words:** {len(words)}")
#         st.write(f"**Characters:** {char_count}")
#         st.write(f"**Sentences:** {len(sentences)}")

# # Spelling Errors
#     if check_spelling:
#         with st.expander("❌ Spelling Errors (Non-word)", expanded=True):
#             spelling_items = result.get(("spelling", -1), [])

#             if not spelling_items:
#                 st.success("No spelling errors found.")
#             else:
#                 for item in spelling_items:
#                     for (word, idx), suggestions in item.items():
#                         st.markdown(f"**Word:** `{word}`")

#                         formatted = [
#                         f"{cand} (ED={dist})"
#                         for cand, dist in suggestions]

#                         st.markdown("**Suggestions:** " + ", ".join(formatted))
#                         st.divider()

# # Grammar Errors
#     if check_grammar:
#         with st.expander("⚠ Grammar Errors (Real-word)", expanded=True):
#             context_items = result.get(("context", -2), [])

#             if not context_items:
#                 st.success("No grammar errors found.")
#             else:
#                 for item in context_items:
#                     for (word, idx), suggestions in item.items():
#                         st.markdown(f"**Word:** `{word}`")

#                         formatted = [
#                         f"{cand} (ED={dist})"
#                         for cand, dist in suggestions
#                         ]

#                         st.markdown("**Suggestions:** " + ", ".join(formatted))
#                         st.divider()

# # Suggested sentence
#     if show_suggestions:
#         with st.expander("💡 Suggested Corrections", expanded=True):
#             suggestion = result.get(("suggested", -3))
#             if suggestion:
#                 st.success(suggestion)
#             else:
#                 st.write("No suggestion generated.")

# @st.cache_data
# def load_word_list(path="unigrams.json"):
#     with open(path, "r", encoding="utf-8") as f:
#         word_freq = json.load(f)   # word_freq is a dict: {word: freq}

#     words = list(word_freq.keys())
#     return sorted(words), word_freq

# word_list, word_freq = load_word_list()  # uses word_freq.json by default


# # Highlight & Interactive Fix
# if highlight_errors and result:

#     st.markdown("### ✨ Interactive Highlight & Correction")

#     if "chosen_replacements" not in st.session_state:
#         st.session_state.chosen_replacements = {}

#     token_items = [
#         (tok, idx, sugg)
#         for (tok, idx), sugg in result.items()
#         if isinstance(idx, int) and idx >= 0 and sugg
#     ]

#     if not token_items:
#         st.info("No words require correction.")
#     else:
#         for word, idx, suggestions in token_items:
#             st.markdown(f"**Word:** `{word}`")

#             option_labels = ["(keep original)"] + [
#                 f"{cand} ({dist})" for cand, dist in suggestions
#             ]

#             selected = st.radio(
#                 label=f"Choose replacement for '{word}'",
#                 options=option_labels,
#                 key=f"choice_{idx}",
#                 horizontal=True
#             )

#             if selected != "(keep original)":
#                 chosen_word = selected.split(" (")[0]
#                 st.session_state.chosen_replacements[idx] = chosen_word
#             else:
#                 st.session_state.chosen_replacements.pop(idx, None)

#             st.markdown("---")



# # Build corrected sentence
#     original_tokens = [
#         tok for (tok, idx) in result.keys()
#         if isinstance(idx, int) and idx >= 0
#     ]

#     corrected_tokens = []

#     for i, tok in enumerate(original_tokens):
#         if i in st.session_state.chosen_replacements:
#             corrected_tokens.append(st.session_state.chosen_replacements[i])
#         else:
#             corrected_tokens.append(tok)

#     corrected_sentence = " ".join(corrected_tokens)

#     st.markdown("### ✅ Updated Sentence")
#     st.success(corrected_sentence)


# # Dictionary Browser
# st.markdown("---")
# st.subheader("📚 Dictionary Browser")

# col1, col2 = st.columns(2)
# with col1:
#     show_index = st.checkbox("Show Index", value=True)
# with col2:
#     show_word = st.checkbox("Show Word", value=True)

# search_query = st.text_input("Search word:")

# if search_query.strip():
#     filtered_words = [w for w in word_list if search_query.lower() in w.lower()]

#     df = pd.DataFrame({"Word": filtered_words})
#     df.index = range(1, len(df) + 1)

#     display_df = df.copy()

#     if not show_word:
#         display_df = display_df.drop(columns=["Word"])

#     if not show_index:
#         display_df = display_df.reset_index(drop=True)

#     row_height = 35
#     max_height = 280
#     table_height = min(max_height, row_height * (len(display_df) + 1))

#     st.write(f"Showing {len(filtered_words)} words")

#     st.dataframe(display_df, width="stretch", height=table_height)
# else:
#     st.info("Type a keyword above to search the dictionary.")

# # Description

# st.markdown("---")
# st.subheader("📖 How to Use")

# st.markdown("""
# 1. Enter your text in the input area  
# 2. Click **Check Text**  
# 3. Review:
#    - Text statistics  
#    - Spelling errors  
#    - Grammar errors  
#    - Suggested corrected sentence  
#    - Change highlighted words 
# 4. Use Dictionary Browser to explore vocabulary
# """)

# # Features
# st.subheader("Features")

# st.markdown("""
# ✅ Real-time spell checking  
# ✅ Grammar correction suggestions  
# ✅ Text statistics  
# ✅ Customizable checking options  
# ✅ Clean, user-friendly interface  
# """)


import streamlit as st
import pandas as pd
import spelling
import re
import json
import os
import traceback

import os, time
import streamlit as st

try:
    import psutil
except ImportError:
    psutil = None

if "last_ram_log" not in st.session_state:
    st.session_state.last_ram_log = 0.0

def setup():
    return spelling.setup()

lm, vocab, suggester = setup()

def log_ram_every(seconds: int = 30):
    if not psutil:
        return
    now = time.time()
    if now - st.session_state.last_ram_log >= seconds:
        p = psutil.Process(os.getpid())
        rss_mb = p.memory_info().rss / (1024 ** 2)
        print(f"[RAM] RSS={rss_mb:.1f} MB")
        st.session_state.last_ram_log = now


# ----------------------------
# Suggestion formatting helpers
# ----------------------------
def _iter_suggestion_pairs(suggestions):
    """Yield (candidate, dist) pairs from many possible formats.

    Supported inputs:
      - list[str]                     -> (cand, None)
      - list[tuple[cand, dist]]       -> (cand, dist)
      - dict[cand, dist]              -> (cand, dist)
      - single value (str/tuple/etc.) -> (value, None) or (cand, dist)
    """
    if suggestions is None:
        return []

    if isinstance(suggestions, dict):
        iterable = list(suggestions.items())
    elif isinstance(suggestions, (list, tuple, set)):
        iterable = list(suggestions)
    else:
        iterable = [suggestions]

    out = []
    for elem in iterable:
        if isinstance(elem, (list, tuple)) and len(elem) == 2:
            cand, dist = elem
        else:
            cand, dist = elem, None
        out.append((cand, dist))
    return out


def format_suggestions(suggestions, *, with_ed: bool):
    """Return a de-duplicated list of labels for the UI."""
    labels = []
    for cand, dist in _iter_suggestion_pairs(suggestions):
        cand_s = str(cand)
        if dist is None:
            labels.append(cand_s)
        else:
            labels.append(f"{cand_s} (ED={dist})" if with_ed else f"{cand_s} ({dist})")

    seen = set()
    deduped = []
    for s in labels:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped


# This is to avoid streamlit always rerun
@st.cache_resource
def setup():
    return spelling.setup()

# try:
#     lm, vocab, suggester = setup()
# except Exception as e:
#     st.error("❌ setup() crashed. Full traceback below:")
#     st.code(traceback.format_exc())
#     st.stop()

log_ram_every(10)

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "last_input" not in st.session_state:
    st.session_state.last_input = ""

# Page config 

st.set_page_config(
    page_title="Spell & Grammar Checker",
    page_icon="📝",
    layout="centered")

# Settings

st.sidebar.title("⚙ Settings")

check_spelling = st.sidebar.checkbox("Check Spelling", value=True)
check_grammar = st.sidebar.checkbox("Check Grammar", value=True)
show_suggestions = st.sidebar.checkbox("Show Suggestions", value=True)
highlight_errors = st.sidebar.checkbox("Highlight Errors", value=True)

# Header

st.title("📝 Spell & Grammar Checker")
st.caption("Spelling and Grammar Correction System")

st.markdown("---")

# Input

st.subheader("Input Text")

input_text = st.text_area(
    "Enter your text below (max 500 characters):",
    height=180,
    max_chars=500,
    placeholder="Type or paste your text here...")


# Button, use to insert input 

def run_analysis():
    if input_text.strip():
        st.session_state.analysis_result = spelling.model(lm, vocab, suggester, paragraph=input_text, mode="c")
        st.session_state.last_input = input_text


st.button("🔍 Check Text", on_click=run_analysis)


# Results part

st.markdown("---")
st.subheader("Results")

result = st.session_state.analysis_result

if not result:
    st.info("Click **Check Text** to analyze your input.")
else:
    words = st.session_state.last_input.split()
    char_count = len(st.session_state.last_input)

    
# Text statistics
    
    with st.expander("📊 Text Statistics", expanded=True):
        sentences = [
            s.strip()
            for s in re.split(r"[.!?]+", st.session_state.last_input)
            if s.strip()]

        st.write(f"**Words:** {len(words)}")
        st.write(f"**Characters:** {char_count}")
        st.write(f"**Sentences:** {len(sentences)}")

# Spelling Errors
    if check_spelling:
        with st.expander("❌ Spelling Errors (Non-word)", expanded=True):
            spelling_items = result.get(("spelling", -1), [])

            if not spelling_items:
                st.success("No spelling errors found.")
            else:
                for item in spelling_items:
                    for (word, idx), suggestions in item.items():
                        st.markdown(f"**Word:** `{word}`")

                        formatted = format_suggestions(suggestions, with_ed=True)

                        st.markdown("**Suggestions:** " + ", ".join(formatted))
                        st.divider()

# Grammar Errors
    if check_grammar:
        with st.expander("⚠ Grammar Errors (Real-word)", expanded=True):
            context_items = result.get(("context", -2), [])

            if not context_items:
                st.success("No grammar errors found.")
            else:
                for item in context_items:
                    for (word, idx), suggestions in item.items():
                        st.markdown(f"**Word:** `{word}`")

                        formatted = format_suggestions(suggestions, with_ed=True)

                        st.markdown("**Suggestions:** " + ", ".join(formatted))
                        st.divider()

# Suggested sentence
    if show_suggestions:
        with st.expander("💡 Suggested Corrections", expanded=True):
            suggestion = result.get(("suggested", -3))
            if suggestion:
                st.success(suggestion)
            else:
                st.write("No suggestion generated.")

@st.cache_data
def load_word_list(path="unigrams.json"):
    with open(path, "r", encoding="utf-8") as f:
        word_freq = json.load(f)   # word_freq is a dict: {word: freq}

    words = list(word_freq.keys())
    return sorted(words), word_freq

word_list, word_freq = load_word_list()  # uses word_freq.json by default


# Highlight & Interactive Fix
if highlight_errors and result:

    st.markdown("### ✨ Interactive Highlight & Correction")

    if "chosen_replacements" not in st.session_state:
        st.session_state.chosen_replacements = {}

    token_items = [
        (tok, idx, sugg)
        for (tok, idx), sugg in result.items()
        if isinstance(idx, int) and idx >= 0 and sugg
    ]

    if not token_items:
        st.info("No words require correction.")
    else:
        for word, idx, suggestions in token_items:
            st.markdown(f"**Word:** `{word}`")

            option_labels = ["(keep original)"] + format_suggestions(suggestions, with_ed=False)

            selected = st.radio(
                label=f"Choose replacement for '{word}'",
                options=option_labels,
                key=f"choice_{idx}",
                horizontal=True
            )

            if selected != "(keep original)":
                chosen_word = selected.split(" (")[0]
                st.session_state.chosen_replacements[idx] = chosen_word
            else:
                st.session_state.chosen_replacements.pop(idx, None)

            st.markdown("---")



# Build corrected sentence
    original_tokens = [
        tok for (tok, idx) in result.keys()
        if isinstance(idx, int) and idx >= 0
    ]

    corrected_tokens = []

    for i, tok in enumerate(original_tokens):
        if i in st.session_state.chosen_replacements:
            corrected_tokens.append(st.session_state.chosen_replacements[i])
        else:
            corrected_tokens.append(tok)

    corrected_sentence = " ".join(corrected_tokens)

    st.markdown("### ✅ Updated Sentence")
    st.success(corrected_sentence)


# Dictionary Browser
st.markdown("---")
st.subheader("📚 Dictionary Browser")

col1, col2 = st.columns(2)
with col1:
    show_index = st.checkbox("Show Index", value=True)
with col2:
    show_word = st.checkbox("Show Word", value=True)

search_query = st.text_input("Search word:")

if search_query.strip():
    filtered_words = [w for w in word_list if search_query.lower() in w.lower()]

    df = pd.DataFrame({"Word": filtered_words})
    df.index = range(1, len(df) + 1)

    display_df = df.copy()

    if not show_word:
        display_df = display_df.drop(columns=["Word"])

    if not show_index:
        display_df = display_df.reset_index(drop=True)

    row_height = 35
    max_height = 280
    table_height = min(max_height, row_height * (len(display_df) + 1))

    st.write(f"Showing {len(filtered_words)} words")

    st.dataframe(display_df, width="stretch", height=table_height)
else:
    st.info("Type a keyword above to search the dictionary.")

# Description

st.markdown("---")
st.subheader("📖 How to Use")

st.markdown("""
1. Enter your text in the input area  
2. Click **Check Text**  
3. Review:
   - Text statistics  
   - Spelling errors  
   - Grammar errors  
   - Suggested corrected sentence  
   - Change highlighted words 
4. Use Dictionary Browser to explore vocabulary
""")

# Features
st.subheader("Features")

st.markdown("""
✅ Real-time spell checking  
✅ Grammar correction suggestions  
✅ Text statistics  
✅ Customizable checking options  
✅ Clean, user-friendly interface  
""")

