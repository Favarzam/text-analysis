import streamlit as st
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

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

# Create two columns for text input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Text 1")
    text1 = st.text_area(
        "Enter first text",
        height=300,
        placeholder="Paste or type your first text here...",
        label_visibility="collapsed"
    )
    st.caption(f"Characters: {len(text1)} | Words: {len(text1.split())}")

with col2:
    st.subheader("Text 2")
    text2 = st.text_area(
        "Enter second text",
        height=300,
        placeholder="Paste or type your second text here...",
        label_visibility="collapsed"
    )
    st.caption(f"Characters: {len(text2)} | Words: {len(text2.split())}")

# Add some spacing
st.markdown("---")

# Analysis options
st.subheader("⚙️ Analysis Options")
col_opt1, col_opt2 = st.columns(2)

with col_opt1:
    show_details = st.checkbox("Show detailed breakdown", value=True)

with col_opt2:
    show_diff = st.checkbox("Show text differences", value=False)

# Calculate button
if st.button("🔍 Analyze Similarity", type="primary", use_container_width=True):
    if not text1.strip() or not text2.strip():
        st.error("⚠️ Please enter both texts to compare.")
    else:
        with st.spinner("Analyzing texts..."):
            # Calculate all similarity metrics
            jaccard_sim = jaccard_similarity(text1, text2)
            cosine_sim = cosine_similarity_tfidf(text1, text2)
            sequence_sim = sequence_similarity(text1, text2)
            char_sim = character_similarity(text1, text2)
            
            # Calculate average similarity
            average_sim = (jaccard_sim + cosine_sim + sequence_sim + char_sim) / 4
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Similarity Results")
            
            # Overall similarity (large display)
            st.markdown("### Overall Similarity")
            st.metric(
                label="Average Similarity Score",
                value=f"{average_sim:.2f}%",
                help="Average of all similarity algorithms"
            )
            
            # Progress bar for visual representation
            st.progress(average_sim / 100)
            
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
            if show_diff:
                st.markdown("---")
                st.subheader("🔍 Text Differences")
                
                # Generate diff
                diff = difflib.unified_diff(
                    text1.splitlines(keepends=True),
                    text2.splitlines(keepends=True),
                    fromfile='Text 1',
                    tofile='Text 2',
                    lineterm=''
                )
                
                diff_text = ''.join(diff)
                
                if diff_text:
                    st.code(diff_text, language="diff")
                else:
                    st.success("✅ The texts are identical!")
            
            # Interpretation
            st.markdown("---")
            st.subheader("💡 Interpretation")
            
            if average_sim >= 90:
                st.success("🟢 **Very High Similarity**: The texts are nearly identical or very closely related.")
            elif average_sim >= 70:
                st.info("🔵 **High Similarity**: The texts share substantial content and meaning.")
            elif average_sim >= 50:
                st.warning("🟡 **Moderate Similarity**: The texts have some common elements but also notable differences.")
            elif average_sim >= 30:
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
    """)
    
    st.markdown("---")
    st.markdown("Built with [Streamlit](https://streamlit.io)")

