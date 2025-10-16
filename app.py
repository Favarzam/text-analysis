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
                        ["Unified Diff View", "Side-by-Side", "Word Highlights"],
                        horizontal=True,
                        help="Select how you want to view the differences"
                    )
                    
                    # Helper function to escape HTML
                    def escape_html(text):
                        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    
                    if comparison_mode == "Unified Diff View":
                        # Intelligent analysis view with explanations and visual map
                        st.info("📝 **Smart Analysis** - AI-powered difference analysis with explanations and visual map")
                        
                        lines1 = text1.splitlines()
                        lines2 = text2.splitlines()
                        
                        # Analyze differences
                        matcher = difflib.SequenceMatcher(None, lines1, lines2)
                        opcodes = matcher.get_opcodes()
                        
                        # Collect difference information
                        additions = []
                        deletions = []
                        modifications = []
                        total_lines1 = len(lines1)
                        total_lines2 = len(lines2)
                        unchanged_lines = 0
                        
                        for tag, i1, i2, j1, j2 in opcodes:
                            if tag == 'equal':
                                unchanged_lines += (i2 - i1)
                            elif tag == 'delete':
                                for idx in range(i1, i2):
                                    deletions.append((idx, lines1[idx]))
                            elif tag == 'insert':
                                for idx in range(j1, j2):
                                    additions.append((idx, lines2[idx]))
                            elif tag == 'replace':
                                for idx in range(i1, i2):
                                    modifications.append(('removed', idx, lines1[idx]))
                                for idx in range(j1, j2):
                                    modifications.append(('added', idx, lines2[idx]))
                        
                        # Summary explanation
                        st.markdown("### 📋 Difference Summary")
                        
                        summary_html = f"""
                        <div style='background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%); 
                                    padding: 24px; border-radius: 12px; border-left: 5px solid #6366f1;'>
                            <h4 style='margin-top: 0; color: #6366f1;'>📊 Analysis Overview</h4>
                            <p style='font-size: 16px; line-height: 1.8; margin: 12px 0;'>
                                <strong>Text 1</strong> contains <strong>{total_lines1} lines</strong> while 
                                <strong>Text 2</strong> contains <strong>{total_lines2} lines</strong>.
                            </p>
                            <ul style='font-size: 15px; line-height: 2;'>
                                <li><strong style='color: #28a745;'>{len(additions)} lines added</strong> in Text 2</li>
                                <li><strong style='color: #dc3545;'>{len(deletions)} lines removed</strong> from Text 1</li>
                                <li><strong style='color: #fd7e14;'>{len(modifications)//2 if len(modifications) > 0 else 0} lines modified</strong></li>
                                <li><strong style='color: #6c757d;'>{unchanged_lines} lines unchanged</strong></li>
                            </ul>
                        </div>
                        """
                        st.markdown(summary_html, unsafe_allow_html=True)
                        
                        # Visual difference map
                        st.markdown("### 🗺️ Visual Difference Map")
                        st.caption("Each segment represents a section of the text. Color indicates the type of change.")
                        
                        # Create visual map data
                        map_segments = []
                        for tag, i1, i2, j1, j2 in opcodes:
                            size = max(i2 - i1, j2 - j1)
                            if tag == 'equal':
                                map_segments.append(('unchanged', size))
                            elif tag == 'delete':
                                map_segments.append(('removed', i2 - i1))
                            elif tag == 'insert':
                                map_segments.append(('added', j2 - j1))
                            elif tag == 'replace':
                                map_segments.append(('modified', size))
                        
                        # Generate visual map HTML
                        total_segments = sum(s[1] for s in map_segments)
                        map_html = "<div style='display: flex; width: 100%; height: 60px; border-radius: 8px; overflow: hidden; border: 2px solid #ccc;'>"
                        
                        for seg_type, seg_size in map_segments:
                            width_pct = (seg_size / total_segments * 100) if total_segments > 0 else 0
                            if seg_type == 'unchanged':
                                color = '#e9ecef'
                                tooltip = 'Unchanged'
                            elif seg_type == 'added':
                                color = '#28a745'
                                tooltip = 'Added'
                            elif seg_type == 'removed':
                                color = '#dc3545'
                                tooltip = 'Removed'
                            else:  # modified
                                color = '#fd7e14'
                                tooltip = 'Modified'
                            
                            if width_pct > 0:
                                map_html += f"<div style='width: {width_pct}%; background-color: {color}; height: 100%; border-right: 1px solid white;' title='{tooltip}'></div>"
                        
                        map_html += "</div>"
                        
                        # Legend for map
                        map_html += """
                        <div style='display: flex; justify-content: center; gap: 24px; margin-top: 16px; flex-wrap: wrap;'>
                            <div style='display: flex; align-items: center; gap: 8px;'>
                                <div style='width: 20px; height: 20px; background-color: #28a745; border-radius: 4px;'></div>
                                <span>Added</span>
                            </div>
                            <div style='display: flex; align-items: center; gap: 8px;'>
                                <div style='width: 20px; height: 20px; background-color: #dc3545; border-radius: 4px;'></div>
                                <span>Removed</span>
                            </div>
                            <div style='display: flex; align-items: center; gap: 8px;'>
                                <div style='width: 20px; height: 20px; background-color: #fd7e14; border-radius: 4px;'></div>
                                <span>Modified</span>
                            </div>
                            <div style='display: flex; align-items: center; gap: 8px;'>
                                <div style='width: 20px; height: 20px; background-color: #e9ecef; border-radius: 4px;'></div>
                                <span>Unchanged</span>
                            </div>
                        </div>
                        """
                        
                        st.markdown(map_html, unsafe_allow_html=True)
                        
                        # Show specific examples
                        st.markdown("---")
                        st.markdown("### 📝 Difference Examples")
                        
                        # Show additions
                        if additions:
                            with st.expander(f"➕ **Added Content** ({len(additions)} lines)", expanded=len(additions) <= 5):
                                st.markdown("*These lines appear in Text 2 but not in Text 1:*")
                                examples_to_show = min(10, len(additions))
                                for i, (line_num, line) in enumerate(additions[:examples_to_show]):
                                    excerpt = line[:150] + "..." if len(line) > 150 else line
                                    st.markdown(f"""
                                    <div style='background-color: rgba(40, 167, 69, 0.1); padding: 12px; margin: 8px 0; border-left: 4px solid #28a745; border-radius: 4px;'>
                                        <div style='color: #28a745; font-weight: bold; margin-bottom: 6px;'>📍 Line {line_num + 1} in Text 2</div>
                                        <code style='color: #28a745;'>{escape_html(excerpt)}</code>
                                    </div>
                                    """, unsafe_allow_html=True)
                                if len(additions) > examples_to_show:
                                    st.caption(f"... and {len(additions) - examples_to_show} more lines")
                        
                        # Show deletions
                        if deletions:
                            with st.expander(f"➖ **Removed Content** ({len(deletions)} lines)", expanded=len(deletions) <= 5):
                                st.markdown("*These lines appear in Text 1 but not in Text 2:*")
                                examples_to_show = min(10, len(deletions))
                                for i, (line_num, line) in enumerate(deletions[:examples_to_show]):
                                    excerpt = line[:150] + "..." if len(line) > 150 else line
                                    st.markdown(f"""
                                    <div style='background-color: rgba(220, 53, 69, 0.1); padding: 12px; margin: 8px 0; border-left: 4px solid #dc3545; border-radius: 4px;'>
                                        <div style='color: #dc3545; font-weight: bold; margin-bottom: 6px;'>📍 Line {line_num + 1} in Text 1</div>
                                        <code style='color: #dc3545;'>{escape_html(excerpt)}</code>
                                    </div>
                                    """, unsafe_allow_html=True)
                                if len(deletions) > examples_to_show:
                                    st.caption(f"... and {len(deletions) - examples_to_show} more lines")
                        
                        # Show modifications
                        if modifications:
                            with st.expander(f"🔄 **Modified Content** ({len(modifications)//2} changes)", expanded=len(modifications) <= 10):
                                st.markdown("*These sections were changed between the texts:*")
                                examples_to_show = min(10, len(modifications)//2)
                                
                                # Group modifications
                                i = 0
                                shown = 0
                                while i < len(modifications) and shown < examples_to_show:
                                    if i + 1 < len(modifications):
                                        mod_type1, line_num1, line1 = modifications[i]
                                        mod_type2, line_num2, line2 = modifications[i + 1]
                                        
                                        excerpt1 = line1[:150] + "..." if len(line1) > 150 else line1
                                        excerpt2 = line2[:150] + "..." if len(line2) > 150 else line2
                                        
                                        st.markdown(f"""
                                        <div style='background-color: rgba(255, 193, 7, 0.1); padding: 12px; margin: 8px 0; border-left: 4px solid #fd7e14; border-radius: 4px;'>
                                            <div style='color: #fd7e14; font-weight: bold; margin-bottom: 8px;'>🔄 Modification #{shown + 1}</div>
                                            <div style='margin-bottom: 8px;'>
                                                <span style='color: #dc3545; font-weight: bold;'>Before (Line {line_num1 + 1}):</span><br>
                                                <code style='color: #dc3545;'>{escape_html(excerpt1)}</code>
                                            </div>
                                            <div>
                                                <span style='color: #28a745; font-weight: bold;'>After (Line {line_num2 + 1}):</span><br>
                                                <code style='color: #28a745;'>{escape_html(excerpt2)}</code>
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        i += 2
                                        shown += 1
                                    else:
                                        i += 1
                                
                                if len(modifications)//2 > examples_to_show:
                                    st.caption(f"... and {len(modifications)//2 - examples_to_show} more modifications")
                        
                        # If no differences
                        if not additions and not deletions and not modifications:
                            st.success("✅ No differences found! The texts are identical.")
                    
                    elif comparison_mode == "Side-by-Side":
                        # Side-by-side comparison
                        st.info("📝 **Side-by-Side Comparison** - Compare texts in parallel columns")
                        
                        lines1 = text1.splitlines()
                        lines2 = text2.splitlines()
                        
                        matcher = difflib.SequenceMatcher(None, lines1, lines2)
                        opcodes = matcher.get_opcodes()
                        
                        diff_col1, diff_col2 = st.columns(2)
                        
                        with diff_col1:
                            st.markdown("**📄 Text 1 (Original)**")
                            html1 = """
                            <div style='font-family: "Courier New", monospace; font-size: 13px; padding: 16px; border: 2px solid var(--text-color, #333); border-radius: 8px; max-height: 600px; overflow-y: auto;'>
                            """
                            for tag, i1, i2, j1, j2 in opcodes:
                                if tag == 'equal':
                                    for line in lines1[i1:i2]:
                                        html1 += f"<div style='padding: 6px 10px; margin: 2px 0; color: var(--text-color, #333); opacity: 0.7;'>{escape_html(line) if line else '&nbsp;'}</div>"
                                elif tag == 'delete':
                                    for line in lines1[i1:i2]:
                                        html1 += f"""
                                        <div style='padding: 6px 10px; margin: 2px 0; background: linear-gradient(90deg, rgba(220, 53, 69, 0.2) 0%, rgba(220, 53, 69, 0.05) 100%); border-left: 4px solid #dc3545; color: #dc3545; font-weight: 500;'>
                                            {escape_html(line) if line else '&nbsp;'}
                                        </div>
                                        """
                                elif tag == 'replace':
                                    for line in lines1[i1:i2]:
                                        html1 += f"""
                                        <div style='padding: 6px 10px; margin: 2px 0; background: linear-gradient(90deg, rgba(255, 193, 7, 0.2) 0%, rgba(255, 193, 7, 0.05) 100%); border-left: 4px solid #ffc107; color: #e65100; font-weight: 500;'>
                                            {escape_html(line) if line else '&nbsp;'}
                                        </div>
                                        """
                            html1 += "</div>"
                            st.markdown(html1, unsafe_allow_html=True)
                        
                        with diff_col2:
                            st.markdown("**📄 Text 2 (Modified)**")
                            html2 = """
                            <div style='font-family: "Courier New", monospace; font-size: 13px; padding: 16px; border: 2px solid var(--text-color, #333); border-radius: 8px; max-height: 600px; overflow-y: auto;'>
                            """
                            for tag, i1, i2, j1, j2 in opcodes:
                                if tag == 'equal':
                                    for line in lines2[j1:j2]:
                                        html2 += f"<div style='padding: 6px 10px; margin: 2px 0; color: var(--text-color, #333); opacity: 0.7;'>{escape_html(line) if line else '&nbsp;'}</div>"
                                elif tag == 'insert':
                                    for line in lines2[j1:j2]:
                                        html2 += f"""
                                        <div style='padding: 6px 10px; margin: 2px 0; background: linear-gradient(90deg, rgba(40, 167, 69, 0.2) 0%, rgba(40, 167, 69, 0.05) 100%); border-left: 4px solid #28a745; color: #28a745; font-weight: 500;'>
                                            {escape_html(line) if line else '&nbsp;'}
                                        </div>
                                        """
                                elif tag == 'replace':
                                    for line in lines2[j1:j2]:
                                        html2 += f"""
                                        <div style='padding: 6px 10px; margin: 2px 0; background: linear-gradient(90deg, rgba(255, 193, 7, 0.2) 0%, rgba(255, 193, 7, 0.05) 100%); border-left: 4px solid #ffc107; color: #e65100; font-weight: 500;'>
                                            {escape_html(line) if line else '&nbsp;'}
                                        </div>
                                        """
                            html2 += "</div>"
                            st.markdown(html2, unsafe_allow_html=True)
                        
                        # Legend
                        st.markdown("""
                        <div style='margin-top: 16px; padding: 12px; background-color: rgba(128, 128, 128, 0.05); border-radius: 6px; text-align: center;'>
                            <span style='color: #28a745; font-weight: bold;'>🟢 Green</span> = Added &nbsp;&nbsp;|&nbsp;&nbsp; 
                            <span style='color: #dc3545; font-weight: bold;'>🔴 Red</span> = Removed &nbsp;&nbsp;|&nbsp;&nbsp; 
                            <span style='color: #e65100; font-weight: bold;'>🟡 Orange</span> = Modified &nbsp;&nbsp;|&nbsp;&nbsp; 
                            <span style='opacity: 0.6;'>Gray</span> = Unchanged
                        </div>
                        """, unsafe_allow_html=True)
                    
                    else:  # Word Highlights
                        # Word-by-word comparison with inline highlighting
                        st.info("📝 **Word Highlights** - Shows word-level differences with inline highlighting")
                        
                        words1_list = text1.split()
                        words2_list = text2.split()
                        
                        matcher = difflib.SequenceMatcher(None, words1_list, words2_list)
                        opcodes = matcher.get_opcodes()
                        
                        diff_col1, diff_col2 = st.columns(2)
                        
                        with diff_col1:
                            st.markdown("**📄 Text 1 (Original)**")
                            html1 = """
                            <div style='padding: 20px; border: 2px solid var(--text-color, #333); border-radius: 8px; max-height: 600px; overflow-y: auto; line-height: 2.4; font-size: 14px;'>
                            """
                            for tag, i1, i2, j1, j2 in opcodes:
                                if tag == 'equal':
                                    html1 += ' '.join([f"<span style='color: var(--text-color, #333); opacity: 0.8;'>{escape_html(w)}</span>" for w in words1_list[i1:i2]]) + ' '
                                elif tag == 'delete':
                                    for word in words1_list[i1:i2]:
                                        html1 += f"<span style='background-color: #dc3545; color: white; padding: 4px 8px; margin: 0 2px; border-radius: 6px; font-weight: 600; display: inline-block;'>{escape_html(word)}</span> "
                                elif tag == 'replace':
                                    for word in words1_list[i1:i2]:
                                        html1 += f"<span style='background-color: #fd7e14; color: white; padding: 4px 8px; margin: 0 2px; border-radius: 6px; font-weight: 600; display: inline-block;'>{escape_html(word)}</span> "
                            html1 += "</div>"
                            st.markdown(html1, unsafe_allow_html=True)
                        
                        with diff_col2:
                            st.markdown("**📄 Text 2 (Modified)**")
                            html2 = """
                            <div style='padding: 20px; border: 2px solid var(--text-color, #333); border-radius: 8px; max-height: 600px; overflow-y: auto; line-height: 2.4; font-size: 14px;'>
                            """
                            for tag, i1, i2, j1, j2 in opcodes:
                                if tag == 'equal':
                                    html2 += ' '.join([f"<span style='color: var(--text-color, #333); opacity: 0.8;'>{escape_html(w)}</span>" for w in words2_list[j1:j2]]) + ' '
                                elif tag == 'insert':
                                    for word in words2_list[j1:j2]:
                                        html2 += f"<span style='background-color: #28a745; color: white; padding: 4px 8px; margin: 0 2px; border-radius: 6px; font-weight: 600; display: inline-block;'>{escape_html(word)}</span> "
                                elif tag == 'replace':
                                    for word in words2_list[j1:j2]:
                                        html2 += f"<span style='background-color: #fd7e14; color: white; padding: 4px 8px; margin: 0 2px; border-radius: 6px; font-weight: 600; display: inline-block;'>{escape_html(word)}</span> "
                            html2 += "</div>"
                            st.markdown(html2, unsafe_allow_html=True)
                        
                        # Legend
                        st.markdown("""
                        <div style='margin-top: 16px; padding: 12px; background-color: rgba(128, 128, 128, 0.05); border-radius: 6px; text-align: center;'>
                            <span style='background-color: #28a745; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold;'>🟢 Green</span> = Added words &nbsp;&nbsp;
                            <span style='background-color: #dc3545; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold;'>🔴 Red</span> = Removed words &nbsp;&nbsp;
                            <span style='background-color: #fd7e14; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold;'>🟠 Orange</span> = Modified words
                        </div>
                        """, unsafe_allow_html=True)
                    
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

