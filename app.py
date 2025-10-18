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
Compare two texts and calculate their similarity percentage using multiple algorithms.
Each algorithm provides a different perspective on how similar the texts are.
""")

# Function to calculate Jaccard similarity
def jaccard_similarity(text1, text2):
    """Calculate Jaccard similarity between two texts"""
    # Convert to lowercase and split into words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calculate intersection and union
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
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
    1. Remove ALL punctuation and special characters
    2. Convert to lowercase
    3. Strip whitespace
    4. Keep only alphanumeric characters (including Unicode like Chinese)
    
    Returns the normalized word, or empty string if nothing remains
    """
    if not word:
        return ""
    
    # Remove all punctuation and special characters but keep alphanumeric and unicode
    # This handles: quotes, parentheses, colons, slashes, etc.
    normalized = re.sub(r'[^\w]', '', word, flags=re.UNICODE)
    
    # Convert to lowercase and strip
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
    """
    tokens = []
    for word in text.split():
        normalized = normalize_word(word)
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
    """
    highlight1 = set()
    highlight2 = set()
    
    # Build lookup for exact normalized words
    norm_lookup2 = defaultdict(list)
    for idx2, token2 in enumerate(tokens2):
        norm = token2['normalized']
        if norm and len(norm) >= 1:
            norm_lookup2[norm].append(idx2)
    
    # Only exact matches
    for idx1, token1 in enumerate(tokens1):
        norm1 = token1['normalized']
        if not norm1:
            continue
        
        candidates = norm_lookup2.get(norm1, [])
        if candidates:
            highlight1.add(idx1)
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
    Uses difflib to find contiguous matching blocks.
    """
    highlight1 = set()
    highlight2 = set()
    
    # Use SequenceMatcher on the full text
    matcher = difflib.SequenceMatcher(None, text1, text2)
    matching_blocks = matcher.get_matching_blocks()
    
    # Convert character positions to word indices
    # First, build character-to-word-index mappings
    char_to_word1 = {}
    char_to_word2 = {}
    
    pos1 = 0
    for idx1, token1 in enumerate(tokens1):
        word = token1['original']
        for i in range(len(word)):
            char_to_word1[pos1 + i] = idx1
        pos1 += len(word) + 1  # +1 for space
    
    pos2 = 0
    for idx2, token2 in enumerate(tokens2):
        word = token2['original']
        for i in range(len(word)):
            char_to_word2[pos2 + i] = idx2
        pos2 += len(word) + 1  # +1 for space
    
    # Highlight words that are part of matching blocks
    for block in matching_blocks:
        i, j, size = block
        if size < 3:  # Ignore very short matches
            continue
        
        # Add all word indices in this block
        for char_pos in range(i, i + size):
            if char_pos in char_to_word1:
                highlight1.add(char_to_word1[char_pos])
        
        for char_pos in range(j, j + size):
            if char_pos in char_to_word2:
                highlight2.add(char_to_word2[char_pos])
    
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
    """
    html_parts = [
        "<div style='padding: 20px; border: 1px solid #ccc; border-radius: 8px; min-height: 300px; max-height: 450px; overflow-y: auto; line-height: 2.2; font-size: 14px; background-color: white; color: #333;'>"
    ]
    
    for idx, token in enumerate(tokens):
        word_html = escape_html(token['original'])
        if idx in highlighted_indices:
            html_parts.append(
                f"<span style='background-color: #fff9c4; color: #333; padding: 3px 6px; margin: 0 1px; border-radius: 4px; font-weight: 500; display: inline-block;'>{word_html}</span> "
            )
        else:
            html_parts.append(f"<span style='color: #333;'>{word_html}</span> ")
    
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
    # Get the algorithm that was used during analysis
    algorithm = st.session_state.analysis_results.get('primary_algorithm', 'Average (All)')
    highlighted1, highlighted2 = find_matching_indices(
        tokens_text1, tokens_text2, 
        algorithm=algorithm,
        text1=st.session_state.text1,
        text2=st.session_state.text2
    )
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
col_opt1, col_opt2, col_opt3 = st.columns(3)

with col_opt1:
    show_details = st.checkbox("Show detailed breakdown", value=True)

with col_opt2:
    show_diff = st.checkbox("Show text differences", value=False)

with col_opt3:
    primary_algorithm = st.selectbox(
        "Primary Algorithm",
        ["Average (All)", "TF-IDF Cosine", "Sequence Matcher", "Jaccard (Words)", "Character Overlap"],
        help="Choose which algorithm to use for similarity scoring AND text highlighting (when 'Show text differences' is enabled)"
    )

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
        # Calculate all similarity metrics
        text1 = st.session_state.text1
        text2 = st.session_state.text2
        
        jaccard_sim = jaccard_similarity(text1, text2)
        cosine_sim = cosine_similarity_tfidf(text1, text2)
        sequence_sim = sequence_similarity(text1, text2)
        char_sim = character_similarity(text1, text2)
        average_sim = (jaccard_sim + cosine_sim + sequence_sim + char_sim) / 4
        
        # Determine primary score based on selected algorithm
        if primary_algorithm == "TF-IDF Cosine":
            primary_score = cosine_sim
        elif primary_algorithm == "Sequence Matcher":
            primary_score = sequence_sim
        elif primary_algorithm == "Jaccard (Words)":
            primary_score = jaccard_sim
        elif primary_algorithm == "Character Overlap":
            primary_score = char_sim
        else:  # Average (All)
            primary_score = average_sim
        
        # Store results in session state
        st.session_state.analysis_results = {
            'jaccard_sim': jaccard_sim,
            'cosine_sim': cosine_sim,
            'sequence_sim': sequence_sim,
            'char_sim': char_sim,
            'average_sim': average_sim,
            'primary_score': primary_score,
            'primary_algorithm': primary_algorithm
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
    primary_score = results['primary_score']
    primary_algorithm = results['primary_algorithm']
    text1 = st.session_state.text1
    text2 = st.session_state.text2
    
    # Display results
    st.markdown("---")
    st.subheader("📊 Similarity Results")
    
    # Primary similarity score (large display)
    st.markdown(f"### {primary_algorithm} Score")
    st.metric(
        label=f"Similarity using {primary_algorithm}",
        value=f"{primary_score:.2f}%",
        help=f"Similarity score calculated using {primary_algorithm}"
    )
    
    # Progress bar for visual representation
    st.progress(primary_score / 100)
    
    st.markdown("---")
    
    if show_details:
        st.markdown("### Detailed Breakdown")
        st.markdown("""
        Different algorithms measure similarity in different ways:
        - **TF-IDF Cosine Similarity**: Measures semantic similarity based on term frequency and importance
        - **Sequence Matcher**: Measures character-by-character similarity (best for detecting edits)
        - **Jaccard Similarity**: Measures word overlap (unique words in common)
        - **Character Similarity**: Measures character set overlap
        """)
        
        # Create columns for detailed metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric(
                "TF-IDF Cosine",
                f"{cosine_sim:.2f}%",
                help="Semantic similarity using TF-IDF vectors"
            )
        
        with metric_col2:
            st.metric(
                "Sequence Match",
                f"{sequence_sim:.2f}%",
                help="Character-by-character sequence similarity"
            )
        
        with metric_col3:
            st.metric(
                "Jaccard (Words)",
                f"{jaccard_sim:.2f}%",
                help="Word overlap similarity"
            )
        
        with metric_col4:
            st.metric(
                "Character Overlap",
                f"{char_sim:.2f}%",
                help="Character set similarity"
            )
        
        # Visual comparison bars
        st.markdown("### Visual Comparison")
        st.markdown("**TF-IDF Cosine Similarity**")
        st.progress(cosine_sim / 100)
        
        st.markdown("**Sequence Matcher**")
        st.progress(sequence_sim / 100)
        
        st.markdown("**Jaccard Similarity (Words)**")
        st.progress(jaccard_sim / 100)
        
        st.markdown("**Character Overlap**")
        st.progress(char_sim / 100)
    
    # Show differences if requested
    show_diff = st.session_state.show_diff
    if show_diff:
        st.markdown("---")
        st.subheader("🔍 Text Differences")
        
        if text1 == text2:
            st.success("✅ The texts are identical!")
        else:
            # Show which algorithm is being used for highlighting
            algorithm_descriptions = {
                "Jaccard (Words)": "exact word matches only",
                "TF-IDF Cosine": "semantically important words that appear in both texts",
                "Sequence Matcher": "words within matching text sequences/blocks",
                "Character Overlap": "words with significant character overlap",
                "Average (All)": "comprehensive matching (exact, variants, and fuzzy)"
            }
            algo_desc = algorithm_descriptions.get(primary_algorithm, "matched words")
            
            st.info(f"📝 **Similar words are highlighted in the boxes above using {primary_algorithm} algorithm**")
            st.markdown(f"""
            - <span style='background-color: #fff9c4; color: #333; padding: 2px 6px; border-radius: 3px; font-weight: 500;'>Yellow highlight</span> = {algo_desc.capitalize()}
            - <span style='color: #333;'>No highlight</span> = Words that are different or unique to each text
            
            **Note:** Different algorithms highlight different types of similarities. Try changing the algorithm to see different highlighting patterns!
            """, unsafe_allow_html=True)
            
            # Word-level comparison for statistics
            words1 = text1.split()
            words2 = text2.split()
            
            # Get unique words in each text
            set1 = set(words1)
            set2 = set(words2)
            
            only_in_text1 = set1 - set2
            only_in_text2 = set2 - set1
            common_words = set1 & set2
            
            # Display statistics
            st.markdown("### 📊 Summary Statistics")
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            
            with stat_col1:
                st.metric("✅ Common Words", len(common_words))
            with stat_col2:
                st.metric("➖ Only in Text 1", len(only_in_text1))
            with stat_col3:
                st.metric("➕ Only in Text 2", len(only_in_text2))
            
            # Show unique words if there are any
            if only_in_text1 or only_in_text2:
                st.markdown("---")
                st.markdown("### 🔤 Unique Words Analysis")
                
                unique_col1, unique_col2 = st.columns(2)
                
                with unique_col1:
                    st.markdown("**Words only in Text 1:**")
                    if only_in_text1:
                        sorted_words1 = sorted(list(only_in_text1))
                        words_display = " · ".join([f"**{word}**" for word in sorted_words1[:30]])
                        st.markdown(words_display)
                        if len(only_in_text1) > 30:
                            st.caption(f"... and {len(only_in_text1) - 30} more words")
                    else:
                        st.caption("None")
                
                with unique_col2:
                    st.markdown("**Words only in Text 2:**")
                    if only_in_text2:
                        sorted_words2 = sorted(list(only_in_text2))
                        words_display = " · ".join([f"**{word}**" for word in sorted_words2[:30]])
                        st.markdown(words_display)
                        if len(only_in_text2) > 30:
                            st.caption(f"... and {len(only_in_text2) - 30} more words")
                    else:
                        st.caption("None")
    
    # Interpretation
    st.markdown("---")
    st.subheader("💡 Interpretation")
    
    # Use the primary score for interpretation
    score_to_interpret = primary_score
    
    if score_to_interpret >= 90:
        st.success("🟢 **Very High Similarity**: The texts are nearly identical or very closely related.")
    elif score_to_interpret >= 70:
        st.info("🔵 **High Similarity**: The texts share substantial content and meaning.")
    elif score_to_interpret >= 50:
        st.warning("🟡 **Moderate Similarity**: The texts have some common elements but also notable differences.")
    elif score_to_interpret >= 30:
        st.warning("🟠 **Low Similarity**: The texts have limited overlap.")
    else:
        st.error("🔴 **Very Low Similarity**: The texts are quite different from each other.")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app analyzes text similarity using multiple algorithms to give you a comprehensive view of how similar two texts are.
    
    ### How to Use
    1. Paste or type your texts in the two text areas
    2. Choose your analysis options
    3. Click "Analyze Similarity"
    4. Review the results
    
    ### Algorithms Used
    
    **TF-IDF Cosine Similarity**
    - Best for: Semantic similarity
    - Considers word importance and frequency
    - Range: 0-100%
    
    **Sequence Matcher**
    - Best for: Detecting edits and changes
    - Character-by-character comparison
    - Range: 0-100%
    
    **Jaccard Similarity**
    - Best for: Word overlap
    - Compares unique words
    - Range: 0-100%
    
    **Character Overlap**
    - Best for: Character-level comparison
    - Compares unique characters
    - Range: 0-100%
    
    ### Tips
    - Use multiple algorithms for a complete picture
    - Higher percentages indicate greater similarity
    - Different algorithms may give different results
    - **NEW:** Each algorithm now affects both the similarity score AND the text highlighting pattern when "Show text differences" is enabled
    - Try switching algorithms to see different highlighting behaviors
    """)
    
    st.markdown("---")
    st.markdown("Built with [Streamlit](https://streamlit.io)")
