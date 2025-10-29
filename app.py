import streamlit as st
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import defaultdict, deque
import numpy as np
from sentence_transformers import SentenceTransformer

# Page configuration
st.set_page_config(
    page_title="Text Similarity Analyzer",
    page_icon="📝",
    layout="wide"
)

# Title and description
st.title("📝 Advanced Text Similarity Analyzer")
st.markdown("""
Compare two texts using **6 different algorithms** including AI-powered semantic analysis.
From exact word matching to deep meaning understanding - get comprehensive similarity insights.
Works with all languages. Perfect for plagiarism detection, content analysis, document comparison, and paraphrase detection.
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
        
        # Also add sub-parts if word contains separators (for compound words)
        if '/' in word or ',' in word:
            parts = re.split(r'[/,]+', word)
            for part in parts:
                part_normalized = normalize_word(part)
                if part_normalized and len(part_normalized) >= 2:
                    normalized_words1.add(part_normalized)
    
    normalized_words2 = set()
    for word in text2.split():
        normalized = normalize_word(word)
        if normalized:  # Only add non-empty normalized words
            normalized_words2.add(normalized)
        
        # Also add sub-parts if word contains separators (for compound words)
        if '/' in word or ',' in word:
            parts = re.split(r'[/,]+', word)
            for part in parts:
                part_normalized = normalize_word(part)
                if part_normalized and len(part_normalized) >= 2:
                    normalized_words2.add(part_normalized)
    
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

# Load Sentence-BERT model (cached to avoid reloading)
@st.cache_resource
def load_sbert_model():
    """Load and cache the Sentence-BERT model"""
    # Using all-mpnet-base-v2 - best performance according to MTEB benchmarks
    return SentenceTransformer('all-mpnet-base-v2')

# Function to calculate Sentence-BERT similarity
def sbert_similarity(text1, text2):
    """Calculate semantic similarity using Sentence-BERT embeddings"""
    if not text1.strip() or not text2.strip():
        return 0.0
    
    try:
        model = load_sbert_model()
        
        # Generate embeddings for both texts
        embeddings = model.encode([text1, text2], convert_to_numpy=True)
        
        # Calculate cosine similarity between embeddings
        embedding1 = embeddings[0]
        embedding2 = embeddings[1]
        
        # Cosine similarity: dot product / (norm1 * norm2)
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        
        return similarity * 100
    except Exception as e:
        # Fallback to cosine similarity if SBERT fails
        st.warning(f"SBERT calculation failed, using TF-IDF instead: {str(e)}")
        return cosine_similarity_tfidf(text1, text2)

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
    Also splits compound words separated by slashes, commas, etc. for better matching.
    """
    tokens = []
    # split() without arguments handles multiple spaces, tabs, newlines, etc.
    for word in text.split():
        # Skip empty strings
        if not word.strip():
            continue
        
        # First, try to normalize the whole word
        normalized = normalize_word(word)
        
        # Only add tokens with valid normalized words
        if normalized:
            variants = generate_variants(normalized) if len(normalized) >= 2 else set()
            tokens.append({
                'original': word,
                'normalized': normalized,
                'variants': variants
            })
            
            # ADDITIONALLY: If the word contains slashes or other separators,
            # also tokenize the individual parts for better matching
            # This helps with cases like "(Carbonos/Kohlenstoffe/碳原子)"
            if '/' in word or ',' in word:
                # Split by common separators
                parts = re.split(r'[/,]+', word)
                for part in parts:
                    part_normalized = normalize_word(part)
                    if part_normalized and len(part_normalized) >= 2:
                        # Add these as hidden tokens (they help with matching but don't display separately)
                        # We'll mark them as part of the original compound word
                        part_variants = generate_variants(part_normalized) if len(part_normalized) >= 2 else set()
                        # Use a special marker to identify these as sub-tokens
                        tokens.append({
                            'original': word,  # Keep the original compound word for display
                            'normalized': part_normalized,
                            'variants': part_variants,
                            'is_subtoken': True  # Mark this as a sub-token
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

def find_matching_indices_sbert(tokens1, tokens2, text1, text2):
    """
    SBERT-based highlighting: Highlights semantically similar words/phrases using embeddings.
    Groups tokens into phrases for better semantic matching.
    """
    highlight1 = set()
    highlight2 = set()
    
    if not text1.strip() or not text2.strip():
        return highlight1, highlight2
    
    try:
        model = load_sbert_model()
        
        # Extract normalized words (non-subtokens only)
        words1 = [token['normalized'] for token in tokens1 if not token.get('is_subtoken', False) and token['normalized']]
        words2 = [token['normalized'] for token in tokens2 if not token.get('is_subtoken', False) and token['normalized']]
        
        if not words1 or not words2:
            return highlight1, highlight2
        
        # Generate embeddings for individual words
        embeddings1 = model.encode(words1, convert_to_numpy=True)
        embeddings2 = model.encode(words2, convert_to_numpy=True)
        
        # Calculate similarity matrix
        similarity_matrix = np.dot(embeddings1, embeddings2.T) / (
            np.linalg.norm(embeddings1, axis=1)[:, np.newaxis] * 
            np.linalg.norm(embeddings2, axis=1)[np.newaxis, :]
        )
        
        # Threshold for semantic similarity (0.6 = moderate semantic similarity)
        threshold = 0.6
        
        # Find matches above threshold
        word_idx1 = 0
        for token_idx1, token1 in enumerate(tokens1):
            if token1.get('is_subtoken', False) or not token1['normalized']:
                continue
            
            word_idx2 = 0
            for token_idx2, token2 in enumerate(tokens2):
                if token2.get('is_subtoken', False) or not token2['normalized']:
                    continue
                
                if similarity_matrix[word_idx1, word_idx2] >= threshold:
                    highlight1.add(token_idx1)
                    highlight2.add(token_idx2)
                
                word_idx2 += 1
            
            word_idx1 += 1
        
    except Exception as e:
        # Fallback to exact matching if SBERT fails
        return find_matching_indices_jaccard(tokens1, tokens2)
    
    return highlight1, highlight2

def find_matching_indices(tokens1, tokens2, algorithm="Average (All)", text1="", text2=""):
    """
    Main function that routes to the appropriate highlighting algorithm.
    """
    if algorithm == "Jaccard (Words)":
        return find_matching_indices_jaccard(tokens1, tokens2)
    elif algorithm == "TF-IDF Cosine":
        return find_matching_indices_tfidf(tokens1, tokens2, text1, text2)
    elif algorithm == "Sentence-BERT (Semantic)":
        return find_matching_indices_sbert(tokens1, tokens2, text1, text2)
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
    Handles sub-tokens (compound words split by slashes) properly.
    """
    html_parts = [
        "<div style='padding: 20px; border: 1px solid #ccc; border-radius: 8px; min-height: 300px; max-height: 450px; overflow-y: auto; line-height: 2.2; font-size: 14px; background-color: white; color: #333; word-spacing: normal;'>"
    ]
    
    # Keep track of already rendered words to avoid duplicates from sub-tokens
    rendered_originals = set()
    
    for idx, token in enumerate(tokens):
        # Skip sub-tokens for display (they're only for matching)
        # But use them to highlight the main token
        is_subtoken = token.get('is_subtoken', False)
        
        # Get the original word
        original_word = token['original'].strip()
        
        # Skip if this is a sub-token and we've already rendered this original word
        if is_subtoken and original_word in rendered_originals:
            continue
        
        # Check if this token or any of its sub-tokens are highlighted
        should_highlight = idx in highlighted_indices
        
        # If this is not a sub-token, check if any following sub-tokens with same original are highlighted
        if not is_subtoken:
            # Look ahead for sub-tokens with the same original
            for future_idx in range(idx + 1, len(tokens)):
                future_token = tokens[future_idx]
                if (future_token.get('is_subtoken', False) and 
                    future_token['original'].strip() == original_word and
                    future_idx in highlighted_indices):
                    should_highlight = True
                    break
                # Stop looking if we hit a different word
                if not future_token.get('is_subtoken', False):
                    break
        
        # Only render if this is not a sub-token
        if not is_subtoken:
            word_html = escape_html(original_word)
            
            if should_highlight:
                html_parts.append(
                    f"<span style='background-color: #fff9c4; color: #333; padding: 3px 6px; margin: 0 2px; border-radius: 4px; font-weight: 500; display: inline-block; white-space: nowrap;'>{word_html}</span> "
                )
            else:
                html_parts.append(f"<span style='color: #333; margin: 0 2px; display: inline-block; white-space: nowrap;'>{word_html}</span> ")
            
            rendered_originals.add(original_word)
    
    html_parts.append("</div>")
    return "".join(html_parts)

# Create two columns for text input
col1, col2 = st.columns(2)

# Input text areas (always editable - highlighting shown in results section below)
with col1:
    st.subheader("Text 1")
    text1 = st.text_area(
        "Enter first text",
        height=300,
        placeholder="Paste or type your first text here...",
        label_visibility="collapsed",
        key="text1_input",
        value=st.session_state.text1
    )
    st.session_state.text1 = text1
    st.caption(f"Characters: {len(st.session_state.text1)} | Words: {len(st.session_state.text1.split())}")

with col2:
    st.subheader("Text 2")
    text2 = st.text_area(
        "Enter second text",
        height=300,
        placeholder="Paste or type your second text here...",
        label_visibility="collapsed",
        key="text2_input",
        value=st.session_state.text2
    )
    st.session_state.text2 = text2
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
        # Calculate similarity using all available methods
        text1 = st.session_state.text1
        text2 = st.session_state.text2
        
        # Show progress indicator while calculating (especially for SBERT)
        with st.spinner('Calculating similarities...'):
            # Jaccard similarity (word overlap, ignores punctuation)
            jaccard_sim = jaccard_similarity(text1, text2)
            
            # TF-IDF Cosine similarity (statistical, weighted by importance)
            cosine_sim = cosine_similarity_tfidf(text1, text2)
            
            # Sentence-BERT semantic similarity (understands meaning)
            sbert_sim = sbert_similarity(text1, text2)
            
            # Sequence matcher (order-aware)
            sequence_sim = sequence_similarity(text1, text2)
            
            # Character-level similarity
            char_sim = character_similarity(text1, text2)
            
            # Average of all metrics
            average_sim = (jaccard_sim + cosine_sim + sbert_sim + sequence_sim + char_sim) / 5
        
        # Store results in session state
        st.session_state.analysis_results = {
            'jaccard_sim': jaccard_sim,
            'cosine_sim': cosine_sim,
            'sbert_sim': sbert_sim,
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
    sbert_sim = results['sbert_sim']
    sequence_sim = results['sequence_sim']
    char_sim = results['char_sim']
    average_sim = results['average_sim']
    text1 = st.session_state.text1
    text2 = st.session_state.text2
    
    # Display results
    st.markdown("---")
    st.subheader("📊 Similarity Results")
    
    # Create columns for different metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🔤 Jaccard (Word Overlap)",
            value=f"{jaccard_sim:.1f}%",
            help="Exact word matching - ignores punctuation and case. Good for duplicate detection."
        )
        st.progress(float(jaccard_sim) / 100.0)
    
    with col2:
        st.metric(
            label="📊 TF-IDF Cosine",
            value=f"{cosine_sim:.1f}%",
            help="Statistical similarity - weights important words. Good for document comparison."
        )
        st.progress(float(cosine_sim) / 100.0)
    
    with col3:
        st.metric(
            label="🧠 Sentence-BERT (Semantic)",
            value=f"{sbert_sim:.1f}%",
            help="AI-powered semantic understanding - captures meaning and context. Best for paraphrases."
        )
        st.progress(float(sbert_sim) / 100.0)
    
    # Second row of metrics
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.metric(
            label="📝 Sequence Matcher",
            value=f"{sequence_sim:.1f}%",
            help="Order-aware matching - finds contiguous matching blocks. Good for revisions."
        )
        st.progress(float(sequence_sim) / 100.0)
    
    with col5:
        st.metric(
            label="🔡 Character Overlap",
            value=f"{char_sim:.1f}%",
            help="Character-level similarity - good for typos and minor variations."
        )
        st.progress(float(char_sim) / 100.0)
    
    with col6:
        st.metric(
            label="⭐ Average (All Methods)",
            value=f"{average_sim:.1f}%",
            help="Average of all similarity metrics for a balanced view."
        )
        st.progress(float(average_sim) / 100.0)
    
    # Show differences if requested
    show_diff = st.session_state.show_diff
    if show_diff:
        st.markdown("---")
        st.subheader("🔍 Text Differences (Visual Highlighting)")
        
        if text1 == text2:
            st.success("✅ The texts are identical!")
        else:
            # Allow user to select highlighting algorithm
            highlight_algo = st.selectbox(
                "Select highlighting algorithm:",
                ["Jaccard (Words)", "TF-IDF Cosine", "Sentence-BERT (Semantic)", "Sequence Matcher", "Character Overlap", "Average (All)"],
                help="Choose how to identify and highlight similar content"
            )
            
            st.info("📝 **Similar words/content are highlighted in yellow below**")
            st.markdown(f"""
            - <span style='background-color: #fff9c4; color: #333; padding: 2px 6px; border-radius: 3px; font-weight: 500;'>Yellow highlight</span> = Similar content (using {highlight_algo})
            - <span style='color: #333;'>No highlight</span> = Unique content
            
            **Note:** Different algorithms highlight different types of similarity - try them all!
            """, unsafe_allow_html=True)
            
            # Render highlighted text with selected algorithm
            tokens_text1 = tokenize_text(text1)
            tokens_text2 = tokenize_text(text2)
            highlighted1, highlighted2 = find_matching_indices(tokens_text1, tokens_text2, highlight_algo, text1, text2)
            highlight_html1 = render_highlighted_text(tokens_text1, highlighted1)
            highlight_html2 = render_highlighted_text(tokens_text2, highlighted2)
            
            col_h1, col_h2 = st.columns(2)
            with col_h1:
                st.markdown("**Text 1 (Highlighted)**")
                st.markdown(highlight_html1, unsafe_allow_html=True)
            with col_h2:
                st.markdown("**Text 2 (Highlighted)**")
                st.markdown(highlight_html2, unsafe_allow_html=True)
    
    # Interpretation
    st.markdown("---")
    st.subheader("💡 Interpretation")
    
    # Provide interpretation based on SBERT (semantic) and Jaccard (exact matching)
    st.markdown("### Overall Assessment")
    
    # Primary interpretation based on SBERT (semantic understanding)
    if sbert_sim >= 80:
        st.success("🟢 **Semantically Very Similar**: The texts convey highly similar meanings, even if different words are used.")
    elif sbert_sim >= 60:
        st.info("🔵 **Semantically Similar**: The texts express related ideas with moderate semantic overlap.")
    elif sbert_sim >= 40:
        st.warning("🟡 **Somewhat Related**: The texts share some semantic connections but have significant differences.")
    else:
        st.error("🔴 **Semantically Different**: The texts discuss different topics or convey different meanings.")
    
    # Secondary interpretation based on Jaccard (exact word matching)
    st.markdown("### Word-Level Analysis")
    if jaccard_sim >= 70:
        st.write("✅ **Very High Word Overlap (≥70%)**: Likely duplicates or near-duplicates with exact word matching.")
    elif jaccard_sim >= 50:
        st.write("✅ **High Word Overlap (50-69%)**: Strong vocabulary overlap, possibly related versions or paraphrases.")
    elif jaccard_sim >= 25:
        st.write("⚠️ **Moderate Word Overlap (25-49%)**: Noticeable shared vocabulary but also substantial differences.")
    else:
        st.write("❌ **Low Word Overlap (<25%)**: Limited shared vocabulary, texts are largely distinct at word level.")
    
    # Highlight discrepancies between semantic and exact matching
    semantic_vs_exact_diff = abs(sbert_sim - jaccard_sim)
    if semantic_vs_exact_diff > 30:
        st.markdown("### 🔍 Key Finding")
        if sbert_sim > jaccard_sim:
            st.info("**📚 Paraphrased Content Detected**: Semantic similarity is much higher than word overlap, suggesting the texts convey similar meanings using different vocabulary (e.g., synonyms, rephrasing).")
        else:
            st.warning("**⚠️ Coincidental Word Overlap**: Word overlap is high but semantic similarity is lower, suggesting shared common words without similar meaning.")
    
    # Add explanation of thresholds and algorithms
    with st.expander("📖 Understanding Similarity Algorithms & Thresholds"):
        st.markdown("""
        **Algorithm Overview:**
        
        This tool uses multiple algorithms, each suited for different tasks:
        
        **1. 🧠 Sentence-BERT (Semantic) - BEST FOR MEANING**
        - **What it does**: Uses AI to understand the actual meaning of text
        - **Strengths**: Detects paraphrases, synonyms, and semantic relationships
        - **Best for**: Content analysis, paraphrase detection, understanding context
        - **Thresholds**: ≥80% (very similar meaning), 60-79% (related), 40-59% (somewhat related), <40% (different)
        - **Research**: 15-30% better than statistical methods on semantic tasks
        
        **2. 🔤 Jaccard (Word Overlap) - BEST FOR EXACT MATCHING**
        - **What it does**: Counts words that appear in both texts (case-insensitive)
        - **Strengths**: Fast, simple, good for duplicate detection
        - **Best for**: Finding copies, plagiarism detection, exact duplicates
        - **Thresholds**: ≥70% (near-duplicate), 50-69% (high overlap), 25-49% (moderate), <25% (low)
        - **Limitation**: Cannot understand meaning - "car" and "automobile" treated as different
        
        **3. 📊 TF-IDF Cosine - BEST FOR DOCUMENT COMPARISON**
        - **What it does**: Weights words by importance, emphasizes rare/meaningful terms
        - **Strengths**: Handles document-level comparison, accounts for word frequency
        - **Best for**: Document similarity, topic matching, content categorization
        - **Thresholds**: ≥70% (very similar), 50-69% (similar), 30-49% (related), <30% (different)
        
        **4. 📝 Sequence Matcher - BEST FOR REVISIONS**
        - **What it does**: Finds longest matching sequences, respects word order
        - **Strengths**: Good for finding edited sections, tracks changes
        - **Best for**: Comparing document versions, tracking edits, revision analysis
        
        **5. 🔡 Character Overlap - BEST FOR TYPOS**
        - **What it does**: Compares character-level similarity
        - **Strengths**: Tolerant to spelling variations and typos
        - **Best for**: Fuzzy matching, handling misspellings, name matching
        
        ---
        
        **Research-Based Threshold Ranges:**
        
        Based on peer-reviewed research (2015-2025):
        - **Near-duplicate detection**: Use Jaccard with threshold 0.7-0.9
        - **Semantic similarity**: Use SBERT with threshold 0.4-0.6
        - **Document comparison**: Use TF-IDF with threshold 0.5-0.7
        - **Paraphrase detection**: Use SBERT (Jaccard fails here)
        
        **⚠️ Important**: The commonly assumed 0.5 threshold lacks empirical support. 
        Research shows optimal thresholds vary by domain from 0.2 to 0.9.
        
        *Thresholds based on: PLOS ONE (Bettembourg et al., 2015), EMNLP (Reimers & Gurevych, 2019), 
        ACL Anthology studies, and MTEB benchmark (2023).*
        """)

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Advanced multi-algorithm text similarity analyzer** with AI-powered semantic understanding.
    
    ### 📖 How to Use
    1. **Paste** your texts into the two boxes
    2. **Check** "Show text differences" to visualize (optional)
    3. **Click** "Analyze Similarity"
    4. **View** 6 different similarity metrics
    5. **Select** highlighting algorithm to visualize matches
    
    ---
    
    ### 🧠 Algorithms Available
    
    **1. Sentence-BERT (Semantic) ⭐ NEW**
    - AI-powered meaning understanding
    - Detects paraphrases & synonyms
    - Best for: Content analysis, semantic similarity
    - 13,000× faster than standard BERT
    
    **2. Jaccard (Word Overlap)**
    - Exact word matching (case-insensitive)
    - Best for: Duplicate detection, plagiarism
    - Fast and efficient
    
    **3. TF-IDF Cosine**
    - Statistical similarity with word weighting
    - Best for: Document comparison
    - Emphasizes important terms
    
    **4. Sequence Matcher**
    - Order-aware matching
    - Best for: Tracking revisions, edits
    - Finds contiguous blocks
    
    **5. Character Overlap**
    - Character-level similarity
    - Best for: Typos, fuzzy matching
    - Tolerant to spelling errors
    
    **6. Average (All Methods)**
    - Balanced comprehensive view
    - Combines all algorithms
    
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
    
    **By Algorithm:**
    
    🧠 **SBERT** (Semantic):
    - Paraphrase detection
    - Content analysis
    - Semantic matching
    - Understanding context
    
    🔤 **Jaccard**:
    - Plagiarism detection (threshold: 0.7-0.9)
    - Duplicate detection
    - Exact copy finding
    
    📊 **TF-IDF**:
    - Document categorization
    - Topic similarity
    - Content recommendation
    
    📝 **Sequence Matcher**:
    - Version comparison
    - Change tracking
    - Revision analysis
    
    ---
    
    ### 📊 Research-Based Thresholds
    
    Based on peer-reviewed literature (2015-2025):
    
    **SBERT Semantic:**
    - ≥ 80% = Very similar meaning
    - 60-79% = Semantically related
    - 40-59% = Somewhat related
    - < 40% = Different topics
    
    **Jaccard Word Overlap:**
    - ≥ 70% = Near-duplicate
    - 50-69% = High overlap
    - 25-49% = Moderate overlap
    - < 25% = Low overlap
    
    **Key Insight:** Research shows SBERT achieves 
    15-30% better performance on semantic tasks 
    compared to statistical methods.
    
    ---
    
    ### 💻 Technical Details
    
    **Powered by:**
    - Sentence-BERT (all-mpnet-base-v2)
    - scikit-learn TF-IDF
    - Python difflib
    - Streamlit UI
    
    **Research sources:**
    - EMNLP 2019 (Reimers & Gurevych)
    - PLOS ONE (Bettembourg et al., 2015)
    - MTEB Benchmark (2023)
    - ACL Anthology studies
    
    ---
    
    """)
    
    st.markdown("Built with [Streamlit](https://streamlit.io) 🎈 | Powered by 🧠 AI")
