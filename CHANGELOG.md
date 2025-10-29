# Changelog

## Version 2.0.0 - Major Update (2025)

### 🎉 Major Features Added

#### 1. Sentence-BERT (SBERT) Integration ⭐
- **AI-Powered Semantic Analysis**: Added state-of-the-art Sentence-BERT for deep meaning understanding
- **Model**: Using `all-mpnet-base-v2` (best performing model per MTEB benchmarks)
- **Performance**: 13,000× faster than standard BERT, 15-30% better accuracy on semantic tasks
- **Features**:
  - Detects paraphrases and synonyms
  - Understands context and meaning
  - Recognizes semantic relationships
  - Works across all languages

#### 2. Enhanced UI/UX
- **6 Metrics Display**: All similarity algorithms now prominently displayed
- **Grid Layout**: Clean 2x3 grid showing all metrics simultaneously with progress bars
- **Interactive Highlighting**: Select any algorithm to visualize text similarities
- **Real-time Comparison**: Compare different algorithm results instantly

#### 3. TF-IDF Cosine Visibility
- **Previously Hidden**: TF-IDF was calculated but not shown in UI
- **Now Prominent**: Displayed alongside all other metrics
- **Enhanced Documentation**: Clear explanations of when to use TF-IDF

#### 4. Smart Interpretation
- **Paraphrase Detection**: Automatically detects when semantic similarity exceeds word overlap by 30%+
- **Multi-Metric Analysis**: Interprets results based on both SBERT and Jaccard
- **Contextual Insights**: Explains discrepancies between different algorithms

#### 5. Research-Based Thresholds
- **Evidence-Based**: All thresholds based on peer-reviewed research (2015-2025)
- **Citations Included**: References to EMNLP 2019, PLOS ONE 2015, MTEB 2023, ACL studies
- **Domain-Specific**: Different thresholds for different use cases
- **Educational**: Built-in explanations of why thresholds matter

### 📝 Technical Changes

#### Dependencies Added
- `sentence-transformers >= 2.2.0` - For SBERT functionality
- `torch >= 2.0.0` - Deep learning backend
- `numpy` - Already present, now used for embeddings

#### New Functions
- `load_sbert_model()` - Cached model loading
- `sbert_similarity()` - Semantic similarity calculation
- `find_matching_indices_sbert()` - Semantic-based highlighting

#### UI Improvements
- Multi-column layout for 6 metrics
- Algorithm selector for highlighting
- Expandable research explanations
- Enhanced sidebar documentation

### 🔍 Algorithm Comparison

| Algorithm | Type | Best For | Threshold (High) |
|-----------|------|----------|------------------|
| Sentence-BERT | Semantic | Paraphrases, meaning | ≥80% |
| Jaccard | Exact | Duplicates, plagiarism | ≥70% |
| TF-IDF Cosine | Statistical | Documents, topics | ≥70% |
| Sequence Matcher | Order-aware | Revisions, edits | ≥70% |
| Character Overlap | Character | Typos, fuzzy match | ≥60% |
| Average | Combined | General comparison | ≥70% |

### 📚 Documentation Updates

#### README.md
- Complete rewrite with all 6 algorithms
- Research citations and foundations
- Use cases by algorithm
- Installation instructions with model download info
- Quick start guide

#### Sidebar (In-App)
- Algorithm explanations
- Research-based thresholds
- Use case recommendations
- Multi-language support info

### 🎯 Use Cases Now Supported

#### New with SBERT:
- ✅ Paraphrase detection
- ✅ Content analysis across writing styles
- ✅ Semantic matching for resumes/job descriptions
- ✅ Advanced plagiarism detection (with rewording)

#### Enhanced:
- ✅ Document comparison (TF-IDF now visible)
- ✅ Multi-algorithm validation
- ✅ Comprehensive similarity analysis

### ⚙️ Performance Notes

- **First Run**: Downloads ~420MB SBERT model (one-time)
- **Subsequent Runs**: Model cached, instant loading
- **SBERT Calculation**: 1-3 seconds for typical texts
- **Other Algorithms**: Near-instant (<100ms)

### 🌍 Language Support

All algorithms now tested and working with:
- Latin scripts (English, Spanish, French, German, etc.)
- Arabic (العربية)
- Chinese (中文)
- Japanese (日本語)
- Korean (한국어)
- Cyrillic (Русский)
- Greek (Ελληνικά)
- Hebrew (עברית)

### 🔬 Research Foundation

Implementations based on:
1. **Reimers & Gurevych (EMNLP 2019)**: Sentence-BERT architecture
2. **Bettembourg et al. (PLOS ONE 2015)**: Threshold determination methodology
3. **Muennighoff et al. (EACL 2023)**: MTEB comprehensive benchmarks
4. **Multiple ACL studies (2020-2025)**: Jaccard and TF-IDF thresholds

### 🐛 Bug Fixes

- Fixed highlighting to be in results section (not input boxes)
- Improved text area handling for better UX
- Enhanced error handling for SBERT failures

### 📈 Metrics

Before:
- 4 algorithms (Jaccard prominent, others hidden)
- Single interpretation method
- Basic thresholds

After:
- 6 algorithms (all visible)
- Multi-algorithm interpretation with paraphrase detection
- Research-based domain-specific thresholds
- Interactive highlighting with algorithm selection

---

## Migration Notes

### For Existing Users

No breaking changes - all existing functionality preserved. New features are additive:
- All previous metrics still calculated
- Jaccard still works exactly the same
- Text input/output unchanged
- Just install new dependencies: `pip install -r requirements.txt`

### For Developers

New functions available:
```python
# Load SBERT model
model = load_sbert_model()

# Calculate semantic similarity
similarity = sbert_similarity(text1, text2)

# Semantic highlighting
highlights = find_matching_indices_sbert(tokens1, tokens2, text1, text2)
```

---

**Total Lines Changed**: ~400 lines added/modified
**New Files**: CHANGELOG.md (this file)
**Updated Files**: app.py, requirements.txt, README.md

