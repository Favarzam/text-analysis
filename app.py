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
                
                if text1 == text2:
                    st.success("✅ The texts are identical!")
                else:
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
                    
                    st.markdown("---")
                    
                    # Comparison mode selector
                    st.markdown("### 🎨 Visual Comparison")
                    comparison_mode = st.radio(
                        "Choose comparison mode:",
                        ["Line-by-Line", "Word-by-Word", "Character-Level"],
                        horizontal=True,
                        help="Different modes show different levels of detail"
                    )
                    
                    # Helper function to escape HTML
                    def escape_html(text):
                        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    
                    if comparison_mode == "Line-by-Line":
                        # Line-by-line comparison (cleanest)
                        lines1 = text1.splitlines()
                        lines2 = text2.splitlines()
                        
                        matcher = difflib.SequenceMatcher(None, lines1, lines2)
                        opcodes = matcher.get_opcodes()
                        
                        st.info("📝 **Line-by-Line Comparison** - Shows which lines were added, removed, or changed")
                        
                        diff_col1, diff_col2 = st.columns(2)
                        
                        with diff_col1:
                            st.markdown("**📄 Text 1**")
                            html1 = "<div style='padding: 20px; background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; max-height: 500px; overflow-y: auto;'>"
                            for tag, i1, i2, j1, j2 in opcodes:
                                if tag == 'equal':
                                    for line in lines1[i1:i2]:
                                        html1 += f"<div style='padding: 4px 8px; margin: 2px 0; line-height: 1.6;'>{escape_html(line) if line else '&nbsp;'}</div>"
                                elif tag == 'delete':
                                    for line in lines1[i1:i2]:
                                        html1 += f"<div style='padding: 4px 8px; margin: 2px 0; background-color: #ffe6e6; border-left: 4px solid #ff4444; line-height: 1.6;'>➖ {escape_html(line) if line else '&nbsp;'}</div>"
                                elif tag == 'replace':
                                    for line in lines1[i1:i2]:
                                        html1 += f"<div style='padding: 4px 8px; margin: 2px 0; background-color: #fff3cd; border-left: 4px solid #ff9800; line-height: 1.6;'>🔄 {escape_html(line) if line else '&nbsp;'}</div>"
                            html1 += "</div>"
                            st.markdown(html1, unsafe_allow_html=True)
                        
                        with diff_col2:
                            st.markdown("**📄 Text 2**")
                            html2 = "<div style='padding: 20px; background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; max-height: 500px; overflow-y: auto;'>"
                            for tag, i1, i2, j1, j2 in opcodes:
                                if tag == 'equal':
                                    for line in lines2[j1:j2]:
                                        html2 += f"<div style='padding: 4px 8px; margin: 2px 0; line-height: 1.6;'>{escape_html(line) if line else '&nbsp;'}</div>"
                                elif tag == 'insert':
                                    for line in lines2[j1:j2]:
                                        html2 += f"<div style='padding: 4px 8px; margin: 2px 0; background-color: #e6ffe6; border-left: 4px solid #44ff44; line-height: 1.6;'>➕ {escape_html(line) if line else '&nbsp;'}</div>"
                                elif tag == 'replace':
                                    for line in lines2[j1:j2]:
                                        html2 += f"<div style='padding: 4px 8px; margin: 2px 0; background-color: #fff3cd; border-left: 4px solid #ff9800; line-height: 1.6;'>🔄 {escape_html(line) if line else '&nbsp;'}</div>"
                            html2 += "</div>"
                            st.markdown(html2, unsafe_allow_html=True)
                        
                        st.caption("✅ No highlight = Unchanged | 🔴 Red = Removed | 🟢 Green = Added | 🟡 Yellow = Modified")
                    
                    elif comparison_mode == "Word-by-Word":
                        # Word-by-word comparison
                        st.info("📝 **Word-by-Word Comparison** - Highlights individual word differences")
                        
                        words1_list = text1.split()
                        words2_list = text2.split()
                        
                        matcher = difflib.SequenceMatcher(None, words1_list, words2_list)
                        opcodes = matcher.get_opcodes()
                        
                        diff_col1, diff_col2 = st.columns(2)
                        
                        with diff_col1:
                            st.markdown("**📄 Text 1**")
                            html1 = "<div style='padding: 20px; background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; max-height: 500px; overflow-y: auto; line-height: 2.2;'>"
                            for tag, i1, i2, j1, j2 in opcodes:
                                if tag == 'equal':
                                    html1 += ' '.join([escape_html(w) for w in words1_list[i1:i2]]) + ' '
                                elif tag == 'delete':
                                    for word in words1_list[i1:i2]:
                                        html1 += f"<span style='background-color: #ffcccc; padding: 2px 6px; margin: 0 2px; border-radius: 4px; text-decoration: line-through;'>{escape_html(word)}</span> "
                                elif tag == 'replace':
                                    for word in words1_list[i1:i2]:
                                        html1 += f"<span style='background-color: #ffe4b3; padding: 2px 6px; margin: 0 2px; border-radius: 4px;'>{escape_html(word)}</span> "
                            html1 += "</div>"
                            st.markdown(html1, unsafe_allow_html=True)
                        
                        with diff_col2:
                            st.markdown("**📄 Text 2**")
                            html2 = "<div style='padding: 20px; background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; max-height: 500px; overflow-y: auto; line-height: 2.2;'>"
                            for tag, i1, i2, j1, j2 in opcodes:
                                if tag == 'equal':
                                    html2 += ' '.join([escape_html(w) for w in words2_list[j1:j2]]) + ' '
                                elif tag == 'insert':
                                    for word in words2_list[j1:j2]:
                                        html2 += f"<span style='background-color: #ccffcc; padding: 2px 6px; margin: 0 2px; border-radius: 4px; font-weight: 500;'>{escape_html(word)}</span> "
                                elif tag == 'replace':
                                    for word in words2_list[j1:j2]:
                                        html2 += f"<span style='background-color: #ffe4b3; padding: 2px 6px; margin: 0 2px; border-radius: 4px;'>{escape_html(word)}</span> "
                            html2 += "</div>"
                            st.markdown(html2, unsafe_allow_html=True)
                        
                        st.caption("✅ No highlight = Unchanged | 🔴 Red strikethrough = Removed | 🟢 Green = Added | 🟡 Orange = Modified")
                    
                    else:  # Character-Level
                        # Character-level comparison (most detailed)
                        st.info("📝 **Character-Level Comparison** - Shows exact character differences (most detailed)")
                        st.warning("⚠️ This view may show many highlights for texts with structural differences")
                        
                        matcher = difflib.SequenceMatcher(None, text1, text2)
                        opcodes = matcher.get_opcodes()
                        
                        diff_col1, diff_col2 = st.columns(2)
                        
                        with diff_col1:
                            st.markdown("**📄 Text 1**")
                            html1 = "<div style='padding: 20px; background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; max-height: 500px; overflow-y: auto; white-space: pre-wrap; line-height: 1.8;'>"
                            for tag, i1, i2, j1, j2 in opcodes:
                                if tag == 'equal':
                                    html1 += escape_html(text1[i1:i2])
                                elif tag in ('delete', 'replace'):
                                    html1 += f"<span style='background-color: #ffcccc; padding: 1px 2px;'>{escape_html(text1[i1:i2])}</span>"
                            html1 += "</div>"
                            st.markdown(html1, unsafe_allow_html=True)
                        
                        with diff_col2:
                            st.markdown("**📄 Text 2**")
                            html2 = "<div style='padding: 20px; background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; max-height: 500px; overflow-y: auto; white-space: pre-wrap; line-height: 1.8;'>"
                            for tag, i1, i2, j1, j2 in opcodes:
                                if tag == 'equal':
                                    html2 += escape_html(text2[j1:j2])
                                elif tag in ('insert', 'replace'):
                                    html2 += f"<span style='background-color: #ccffcc; padding: 1px 2px;'>{escape_html(text2[j1:j2])}</span>"
                            html2 += "</div>"
                            st.markdown(html2, unsafe_allow_html=True)
                        
                        st.caption("✅ No highlight = Unchanged | 🔴 Red = Removed | 🟢 Green = Added")
                    
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

