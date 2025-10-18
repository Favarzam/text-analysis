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

# Create two columns for text input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Text 1")
    if not st.session_state.analyzed or not st.session_state.show_diff:
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
        # Show highlighted similarities
        text1 = st.session_state.text1
        words1_list = text1.split()
        words2_list = st.session_state.text2.split()
        
        # Get set of words in text2 for comparison
        words2_set = set([w.lower() for w in words2_list])
        
        html1 = """
        <div style='padding: 20px; border: 1px solid #ccc; border-radius: 8px; min-height: 300px; max-height: 450px; overflow-y: auto; line-height: 2.2; font-size: 14px; background-color: white; color: #333;'>
        """
        for word in words1_list:
            # Check if this word exists in text2 (case-insensitive)
            if word.lower() in words2_set:
                # Highlight matching words
                html1 += f"<span style='background-color: #fff9c4; color: #333; padding: 3px 6px; margin: 0 1px; border-radius: 4px; font-weight: 500; display: inline-block;'>{escape_html(word)}</span> "
            else:
                # Leave different words unhighlighted
                html1 += f"<span style='color: #333;'>{escape_html(word)}</span> "
        html1 += "</div>"
        st.markdown(html1, unsafe_allow_html=True)
    
    st.caption(f"Characters: {len(st.session_state.text1)} | Words: {len(st.session_state.text1.split())}")

with col2:
    st.subheader("Text 2")
    if not st.session_state.analyzed or not st.session_state.show_diff:
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
        # Show highlighted similarities
        text2 = st.session_state.text2
        words1_list = st.session_state.text1.split()
        words2_list = text2.split()
        
        # Get set of words in text1 for comparison
        words1_set = set([w.lower() for w in words1_list])
        
        html2 = """
        <div style='padding: 20px; border: 1px solid #ccc; border-radius: 8px; min-height: 300px; max-height: 450px; overflow-y: auto; line-height: 2.2; font-size: 14px; background-color: white; color: #333;'>
        """
        for word in words2_list:
            # Check if this word exists in text1 (case-insensitive)
            if word.lower() in words1_set:
                # Highlight matching words
                html2 += f"<span style='background-color: #fff9c4; color: #333; padding: 3px 6px; margin: 0 1px; border-radius: 4px; font-weight: 500; display: inline-block;'>{escape_html(word)}</span> "
            else:
                # Leave different words unhighlighted
                html2 += f"<span style='color: #333;'>{escape_html(word)}</span> "
        html2 += "</div>"
        st.markdown(html2, unsafe_allow_html=True)
    
    st.caption(f"Characters: {len(st.session_state.text2)} | Words: {len(st.session_state.text2.split())}")

# Add some spacing
st.markdown("---")

# Analysis options
st.subheader("⚙️ Analysis Options")
col_opt1, col_opt2 = st.columns(2)

with col_opt1:
    show_details = st.checkbox("Show detailed breakdown", value=True)

with col_opt2:
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
        # Calculate all similarity metrics
        text1 = st.session_state.text1
        text2 = st.session_state.text2
        
        jaccard_sim = jaccard_similarity(text1, text2)
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
    show_diff = st.session_state.show_diff
    if show_diff:
        st.markdown("---")
        st.subheader("🔍 Text Differences")
        
        if text1 == text2:
            st.success("✅ The texts are identical!")
        else:
            st.info("📝 **Similar words are highlighted in the boxes above**")
            st.markdown("""
            - <span style='background-color: #fff9c4; color: #333; padding: 2px 6px; border-radius: 3px; font-weight: 500;'>Yellow highlight</span> = Words that appear in both texts
            - <span style='color: #333;'>No highlight</span> = Words that are different or unique to each text
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

