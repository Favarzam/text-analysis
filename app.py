import streamlit as st
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import defaultdict, deque

# Page configuration
st.set_page_config(
    page_title="Text Similarity Analyzer",
    page_icon="📝",
    layout="wide"
)

# Title and description
st.title("📝 Text Similarity Analyzer")
st.markdown("""
Compare two texts word-by-word to find similarities and differences.
Works with all languages and ignores punctuation. Perfect for comparing documents, translations, or revisions.
""")

# Function to calculate Jaccard similarity
def jaccard_similarity(text1, text2):
    """Calculate Jaccard similarity between two texts using normalized words"""
    # Normalize words using the same function as highlighting
    # This ensures consistency between the score and highlighting
    normalized_words1 = set()
    for word in text1.split():
        normalized = normalize_word(word)
        if normalized:  # Only add non-empty normalized words
            normalized_words1.add(normalized)
    
    normalized_words2 = set()
    for word in text2.split():
        normalized = normalize_word(word)
        if normalized:  # Only add non-empty normalized words
            normalized_words2.add(normalized)
    
    # Calculate intersection and union
    intersection = normalized_words1.intersection(normalized_words2)
    union = normalized_words1.union(normalized_words2)
    
    # Return similarity percentage
    if len(union) == 0:
        return 0.0
    return (len(intersection) / len(union)) * 100

# Function to calculate Cosine similarity using TF-IDF
def cosine_similarity_tfidf(text1, text2):
    """Calculate cosine similarity between two texts using TF-IDF"""
    if not text1.strip() or not text2.strip():
        return 0.0
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit and transform the texts
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    return similarity * 100

# Function to calculate Sequence Matcher similarity
def sequence_similarity(text1, text2):
    """Calculate similarity using Python's difflib SequenceMatcher"""
    matcher = difflib.SequenceMatcher(None, text1, text2)
    return matcher.ratio() * 100

# Function to calculate character-level similarity
def character_similarity(text1, text2):
    """Calculate character-level Jaccard similarity"""
    chars1 = set(text1.lower())
    chars2 = set(text2.lower())
    
    intersection = chars1.intersection(chars2)
    union = chars1.union(chars2)
    
    if len(union) == 0:
        return 0.0
    return (len(intersection) / len(union)) * 100

# Initialize session state for storing texts and analysis state
if 'text1' not in st.session_state:
    st.session_state.text1 = ""
if 'text2' not in st.session_state:
    st.session_state.text2 = ""
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'show_diff' not in st.session_state:
    st.session_state.show_diff = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Helper function to escape HTML
def escape_html(text):
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')

# Helper function to normalize words for accurate matching
def normalize_word(word):
    """
    Normalize a word for comparison using strict rules:
    1. Strip all surrounding whitespace
    2. Remove ALL punctuation and special characters
    3. Convert to lowercase
    4. Keep only alphanumeric characters (including Unicode like Chinese, Arabic, etc.)
    
    Returns the normalized word, or empty string if nothing remains
    
    Examples:
    - "Visual:" -> "visual"
    - "  Definition  " -> "definition"
    - "(Carbonos/Kohlenstoffe/碳原子):" -> "carbonoskohlenstoffe碳原子"
    - "  word  " -> "word"
    """
    if not word:
        return ""
    
    # First strip any surrounding whitespace
    word = word.strip()
    
    if not word:
        return ""
    
    # Remove all punctuation and special characters but keep alphanumeric and unicode
    # This handles: quotes, parentheses, colons, slashes, hyphens, etc.
    # \w matches [a-zA-Z0-9_] plus Unicode letter characters
    normalized = re.sub(r'[^\w]', '', word, flags=re.UNICODE)
    
    # Convert to lowercase and strip any remaining whitespace
    normalized = normalized.lower().strip()
    
    return normalized

def generate_variants(normalized_word):
    """
    Generate conservative variants of a normalized word to improve matching.
    Handles common English suffixes while keeping the risk of false positives low.
    """
    variants = set()
    
    if not normalized_word:
        return variants
    
    variants.add(normalized_word)
    word_len = len(normalized_word)
    
    if word_len <= 3:
        return variants
    
    # Handle plural forms
    if normalized_word.endswith("ies") and word_len > 4:
        variants.add(normalized_word[:-3] + "y")
    if normalized_word.endswith("es") and word_len > 4:
        variants.add(normalized_word[:-2])
    if normalized_word.endswith("s") and word_len > 3:
        variants.add(normalized_word[:-1])
    
    # Handle simple verb forms
    if normalized_word.endswith("ing") and word_len > 5:
        variants.add(normalized_word[:-3])
        variants.add(normalized_word[:-3] + "e")
    if normalized_word.endswith("ed") and word_len > 4:
        variants.add(normalized_word[:-2])
        variants.add(normalized_word[:-1])
    
    # Remove very short variants to avoid noise
    variants = {variant for variant in variants if len(variant) >= 2}
    
    return variants

def tokenize_text(text):
    """
    Split text into tokens preserving original form alongside a normalized version
    and a set of matching variants.
    Uses split() without arguments to handle any amount of whitespace.
    """
    tokens = []
    # split() without arguments handles multiple spaces, tabs, newlines, etc.
    for word in text.split():
        # Skip empty strings
        if not word.strip():
            continue
        normalized = normalize_word(word)
        # Only add tokens with valid normalized words
        if normalized:
            variants = generate_variants(normalized) if len(normalized) >= 2 else set()
            tokens.append({
                'original': word,
                'normalized': normalized,
                'variants': variants
            })
    return tokens

def find_matching_indices_jaccard(tokens1, tokens2):
    """
    Jaccard-style highlighting: Only exact word matches (case-insensitive, no punctuation).
    Simple and strict - a word must appear in both texts exactly to be highlighted.
    Handles any amount of whitespace and punctuation differences.
    """
    highlight1 = set()
    highlight2 = set()
    
    # Build lookup for exact normalized words (all valid words)
    norm_lookup2 = defaultdict(list)
    for idx2, token2 in enumerate(tokens2):
        norm = token2['normalized']
        # Include all non-empty normalized words
        if norm:
            norm_lookup2[norm].append(idx2)
    
    # Find exact matches - any word in text1 that appears anywhere in text2
    for idx1, token1 in enumerate(tokens1):
        norm1 = token1['normalized']
        # Skip if no valid normalized word
        if not norm1:
            continue
        
        # Check if this normalized word exists in text2
        candidates = norm_lookup2.get(norm1, [])
        if candidates:
            # Highlight this word in text1
            highlight1.add(idx1)
            # Highlight all occurrences of this word in text2
            for idx2 in candidates:
                highlight2.add(idx2)
    
    return highlight1, highlight2

def find_matching_indices_tfidf(tokens1, tokens2, text1, text2):
    """
    TF-IDF-based highlighting: Highlights words with significant TF-IDF importance.
    Words that are important in both texts get highlighted.
    """
    highlight1 = set()
    highlight2 = set()
    
    if not text1.strip() or not text2.strip():
        return highlight1, highlight2
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r'\b\w+\b')
    
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        feature_names = vectorizer.get_feature_names_out()
        
        # Get TF-IDF scores for each text
        tfidf1 = dict(zip(feature_names, tfidf_matrix[0].toarray()[0]))
        tfidf2 = dict(zip(feature_names, tfidf_matrix[1].toarray()[0]))
        
        # Threshold: words with TF-IDF > 0.1 in both texts
        important_words = set()
        for word in feature_names:
            if tfidf1.get(word, 0) > 0.05 and tfidf2.get(word, 0) > 0.05:
                important_words.add(word.lower())
        
        # Highlight tokens that match important words
        for idx1, token1 in enumerate(tokens1):
            if token1['normalized'] in important_words:
                highlight1.add(idx1)
        
        for idx2, token2 in enumerate(tokens2):
            if token2['normalized'] in important_words:
                highlight2.add(idx2)
    except:
        # Fallback to exact matching if TF-IDF fails
        return find_matching_indices_jaccard(tokens1, tokens2)
    
    return highlight1, highlight2

def find_matching_indices_sequence(tokens1, tokens2, text1, text2):
    """
    Sequence Matcher highlighting: Highlights matching sequences/blocks of text.
    Uses difflib to find contiguous matching blocks at the word level.
    """
    highlight1 = set()
    highlight2 = set()
    
    # Create normalized word sequences for matching
    words1 = [token['normalized'] for token in tokens1]
    words2 = [token['normalized'] for token in tokens2]
    
    # Use SequenceMatcher on word sequences (much more accurate than character-level)
    matcher = difflib.SequenceMatcher(None, words1, words2)
    matching_blocks = matcher.get_matching_blocks()
    
    # Highlight words that are part of matching blocks
    for block in matching_blocks:
        i, j, size = block
        if size == 0:  # Skip empty blocks
            continue
        
        # Add all word indices in this matching block
        for word_idx in range(i, i + size):
            highlight1.add(word_idx)
        
        for word_idx in range(j, j + size):
            highlight2.add(word_idx)
    
    return highlight1, highlight2

def find_matching_indices_comprehensive(tokens1, tokens2):
    """
    Comprehensive matching: Combines exact, variant, and fuzzy matching.
    This is the most thorough approach, used for "Average (All)" mode.
    """
    highlight1 = set()
    highlight2 = set()
    
    # Build a comprehensive lookup for all normalized words in text2
    norm_lookup2 = defaultdict(list)
    for idx2, token2 in enumerate(tokens2):
        norm = token2['normalized']
        if norm and len(norm) >= 1:
            norm_lookup2[norm].append(idx2)
    
    # First pass: Exact matches
    for idx1, token1 in enumerate(tokens1):
        norm1 = token1['normalized']
        if not norm1:
            continue
        
        candidates = norm_lookup2.get(norm1, [])
        if candidates:
            for idx2 in candidates:
                if idx2 not in highlight2:
                    highlight1.add(idx1)
                    highlight2.add(idx2)
                    break
            else:
                if candidates:
                    highlight1.add(idx1)
                    highlight2.add(candidates[0])
    
    # Second pass: Variant matches
    variant_lookup2 = defaultdict(list)
    for idx2, token2 in enumerate(tokens2):
        for variant in token2['variants']:
            variant_lookup2[variant].append(idx2)
    
    for idx1, token1 in enumerate(tokens1):
        if idx1 in highlight1:
            continue
        norm1 = token1['normalized']
        if not norm1 or len(norm1) < 3:
            continue
        
        candidate_indices = []
        for variant in token1['variants']:
            candidate_indices.extend(variant_lookup2.get(variant, []))
        
        if candidate_indices:
            for idx2 in candidate_indices:
                if idx2 not in highlight2:
                    highlight1.add(idx1)
                    highlight2.add(idx2)
                    break
        
        # Third pass: Fuzzy matching
        if idx1 not in highlight1:
            candidate_indices = []
            for idx2, token2 in enumerate(tokens2):
                if idx2 in highlight2:
                    continue
                norm2 = token2['normalized']
                if not norm2 or len(norm2) < 3:
                    continue
                
                if norm1[0] != norm2[0]:
                    continue
                
                length_diff = abs(len(norm1) - len(norm2))
                max_len = max(len(norm1), len(norm2))
                if max_len and length_diff >= max_len * 0.5 and length_diff >= 3:
                    continue
                
                score = difflib.SequenceMatcher(None, norm1, norm2).ratio()
                if score >= 0.75:
                    candidate_indices.append((score, idx2))
            
            if candidate_indices:
                candidate_indices.sort(reverse=True)
                best_score, best_idx2 = candidate_indices[0]
                highlight1.add(idx1)
                highlight2.add(best_idx2)
    
    return highlight1, highlight2

def find_matching_indices_character(tokens1, tokens2):
    """
    Character-based highlighting: Highlights words that share significant character overlap.
    """
    highlight1 = set()
    highlight2 = set()
    
    # For each token, find tokens in the other text with high character overlap
    for idx1, token1 in enumerate(tokens1):
        norm1 = token1['normalized']
        if not norm1 or len(norm1) < 2:
            continue
        
        chars1 = set(norm1)
        best_overlap = 0
        best_idx2 = None
        
        for idx2, token2 in enumerate(tokens2):
            norm2 = token2['normalized']
            if not norm2 or len(norm2) < 2:
                continue
            
            chars2 = set(norm2)
            intersection = len(chars1 & chars2)
            union = len(chars1 | chars2)
            
            if union > 0:
                overlap = intersection / union
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx2 = idx2
        
        # Threshold: at least 60% character overlap
        if best_overlap >= 0.6 and best_idx2 is not None:
            highlight1.add(idx1)
            highlight2.add(best_idx2)
    
    return highlight1, highlight2

def find_matching_indices(tokens1, tokens2, algorithm="Average (All)", text1="", text2=""):
    """
    Main function that routes to the appropriate highlighting algorithm.
    """
    if algorithm == "Jaccard (Words)":
        return find_matching_indices_jaccard(tokens1, tokens2)
    elif algorithm == "TF-IDF Cosine":
        return find_matching_indices_tfidf(tokens1, tokens2, text1, text2)
    elif algorithm == "Sequence Matcher":
        return find_matching_indices_sequence(tokens1, tokens2, text1, text2)
    elif algorithm == "Character Overlap":
        return find_matching_indices_character(tokens1, tokens2)
    else:  # "Average (All)" or default
        return find_matching_indices_comprehensive(tokens1, tokens2)

def render_highlighted_text(tokens, highlighted_indices):
    """
    Build HTML snippet with highlighted tokens based on matched indices.
    Normalizes spacing for clean display while preserving word content.
    """
    html_parts = [
        "<div style='padding: 20px; border: 1px solid #ccc; border-radius: 8px; min-height: 300px; max-height: 450px; overflow-y: auto; line-height: 2.2; font-size: 14px; background-color: white; color: #333; word-spacing: normal;'>"
    ]
    
    for idx, token in enumerate(tokens):
        # Use the original word but strip extra whitespace for clean display
        word_display = token['original'].strip()
        word_html = escape_html(word_display)
        
        if idx in highlighted_indices:
            html_parts.append(
                f"<span style='background-color: #fff9c4; color: #333; padding: 3px 6px; margin: 0 2px; border-radius: 4px; font-weight: 500; display: inline-block; white-space: nowrap;'>{word_html}</span> "
            )
        else:
            html_parts.append(f"<span style='color: #333; margin: 0 2px; display: inline-block; white-space: nowrap;'>{word_html}</span> ")
    
    html_parts.append("</div>")
    return "".join(html_parts)

# Create two columns for text input
col1, col2 = st.columns(2)

# Pre-compute highlighting if the analysis is done and differences are shown
show_highlight = st.session_state.analyzed and st.session_state.show_diff
highlight_html1 = highlight_html2 = None

if show_highlight:
    tokens_text1 = tokenize_text(st.session_state.text1)
    tokens_text2 = tokenize_text(st.session_state.text2)
    # Use word-by-word exact matching (ignores punctuation, works across languages)
    highlighted1, highlighted2 = find_matching_indices_jaccard(tokens_text1, tokens_text2)
    highlight_html1 = render_highlighted_text(tokens_text1, highlighted1)
    highlight_html2 = render_highlighted_text(tokens_text2, highlighted2)

with col1:
    st.subheader("Text 1")
    if not show_highlight:
        text1 = st.text_area(
            "Enter first text",
            height=300,
            placeholder="Paste or type your first text here...",
            label_visibility="collapsed",
            key="text1_input",
            value=st.session_state.text1
        )
        st.session_state.text1 = text1
    else:
        st.markdown(highlight_html1, unsafe_allow_html=True)
    
    st.caption(f"Characters: {len(st.session_state.text1)} | Words: {len(st.session_state.text1.split())}")

with col2:
    st.subheader("Text 2")
    if not show_highlight:
        text2 = st.text_area(
            "Enter second text",
            height=300,
            placeholder="Paste or type your second text here...",
            label_visibility="collapsed",
            key="text2_input",
            value=st.session_state.text2
        )
        st.session_state.text2 = text2
    else:
        st.markdown(highlight_html2, unsafe_allow_html=True)
    
    st.caption(f"Characters: {len(st.session_state.text2)} | Words: {len(st.session_state.text2.split())}")

# Add some spacing
st.markdown("---")

# Analysis options
st.subheader("⚙️ Analysis Options")

show_diff = st.checkbox("Show text differences", value=False)

# Calculate button and Reset button
button_col1, button_col2 = st.columns([3, 1])
with button_col1:
    analyze_clicked = st.button("🔍 Analyze Similarity", type="primary", use_container_width=True)
with button_col2:
    if st.session_state.analyzed:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.analyzed = False
            st.rerun()

if analyze_clicked:
    if not st.session_state.text1.strip() or not st.session_state.text2.strip():
        st.error("⚠️ Please enter both texts to compare.")
    else:
        # Calculate similarity using word-by-word matching (Jaccard)
        text1 = st.session_state.text1
        text2 = st.session_state.text2
        
        # Primary method: Jaccard similarity (word overlap, ignores punctuation)
        jaccard_sim = jaccard_similarity(text1, text2)
        
        # Also calculate other metrics for detailed view
        cosine_sim = cosine_similarity_tfidf(text1, text2)
        sequence_sim = sequence_similarity(text1, text2)
        char_sim = character_similarity(text1, text2)
        average_sim = (jaccard_sim + cosine_sim + sequence_sim + char_sim) / 4
        
        # Store results in session state
        st.session_state.analysis_results = {
            'jaccard_sim': jaccard_sim,
            'cosine_sim': cosine_sim,
            'sequence_sim': sequence_sim,
            'char_sim': char_sim,
            'average_sim': average_sim
        }
        st.session_state.show_diff = show_diff
        st.session_state.analyzed = True
        st.rerun()

# Display results if analysis has been performed
if st.session_state.analyzed and st.session_state.analysis_results:
    results = st.session_state.analysis_results
    jaccard_sim = results['jaccard_sim']
    cosine_sim = results['cosine_sim']
    sequence_sim = results['sequence_sim']
    char_sim = results['char_sim']
    average_sim = results['average_sim']
    text1 = st.session_state.text1
    text2 = st.session_state.text2
    
    # Display results
    st.markdown("---")
    st.subheader("📊 Similarity Results")
    
    # Primary similarity score (word-by-word matching)
    st.markdown("### Word-by-Word Similarity")
    st.metric(
        label="Similarity Score (Word Overlap)",
        value=f"{jaccard_sim:.2f}%",
        help="Percentage of words that appear in both texts (ignores punctuation and case)"
    )
    
    # Progress bar for visual representation
    st.progress(jaccard_sim / 100)
    
    # Show differences if requested
    show_diff = st.session_state.show_diff
    if show_diff:
        st.markdown("---")
        st.subheader("🔍 Text Differences")
        
        if text1 == text2:
            st.success("✅ The texts are identical!")
        else:
            st.info("📝 **Similar words are highlighted in the boxes above**")
            st.markdown("""
            - <span style='background-color: #fff9c4; color: #333; padding: 2px 6px; border-radius: 3px; font-weight: 500;'>Yellow highlight</span> = Words that appear in both texts (ignores punctuation and case)
            - <span style='color: #333;'>No highlight</span> = Words that are different or unique to each text
            
            **Note:** The word matching works across all languages and ignores punctuation marks.
            """, unsafe_allow_html=True)
    
    # Interpretation
    st.markdown("---")
    st.subheader("💡 Interpretation")
    
    if jaccard_sim >= 90:
        st.success("🟢 **Very High Similarity**: The texts share almost all the same words.")
    elif jaccard_sim >= 70:
        st.info("🔵 **High Similarity**: The texts share most of their words.")
    elif jaccard_sim >= 50:
        st.warning("🟡 **Moderate Similarity**: The texts have some words in common but also notable differences.")
    elif jaccard_sim >= 30:
        st.warning("🟠 **Low Similarity**: The texts have limited word overlap.")
    else:
        st.error("🔴 **Very Low Similarity**: The texts share very few words in common.")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    Simple, accurate text comparison tool that works with any language.
    
    ### 📖 How to Use
    1. **Paste** your texts into the two boxes
    2. **Check** "Show text differences" (optional)
    3. **Click** "Analyze Similarity"
    4. **View** results and highlighted differences
    
    ---
    
    ### 🔍 How It Works
    
    **Word-by-Word Matching**
    
    The app compares texts by:
    - Breaking each text into individual words
    - Removing punctuation (periods, commas, quotes, etc.)
    - Converting to lowercase
    - Finding words that appear in both texts
    - Calculating similarity percentage
    
    **Formula:** Jaccard Similarity
    ```
    Similarity = (Common Words) / (Total Unique Words) × 100
    ```
    
    ---
    
    ### 🌍 Language Support
    
    Works with **all languages**:
    - ✅ Latin scripts (English, Spanish, French...)
    - ✅ Arabic (العربية)
    - ✅ Chinese (中文)
    - ✅ Japanese (日本語)
    - ✅ Korean (한국어)
    - ✅ Cyrillic (Русский)
    - ✅ Greek (Ελληνικά)
    - ✅ Hebrew (עברית)
    - ✅ And more!
    
    ---
    
    ### 💡 Use Cases
    
    Perfect for:
    - 📄 Comparing document versions
    - ✏️ Checking text revisions
    - 🔄 Validating translations
    - 📚 Educational content comparison
    - 📝 Detecting plagiarism
    - 🔍 Finding text duplicates
    
    ---
    
    ### 📊 Understanding Results
    
    **Similarity Score:**
    - **90-100%** = Nearly identical
    - **70-89%** = Very similar
    - **50-69%** = Moderately similar
    - **30-49%** = Somewhat similar
    - **0-29%** = Very different
    
    **Highlighting:**
    - 🟨 Yellow = Words in both texts
    - ⬜ No color = Unique words
    
    ---
    
    ### 💻 Technical Details
    
    **Algorithm:** Jaccard Similarity Index
    - Industry-standard metric
    - Fast and efficient
    - Language-agnostic
    - Punctuation-insensitive
    
    ---
    
    """)
    
    st.markdown("Built with [Streamlit](https://streamlit.io) 🎈")
