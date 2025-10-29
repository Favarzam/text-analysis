# Advanced Text Similarity Analyzer

A powerful Streamlit web application that analyzes and compares two texts using **6 different algorithms** including AI-powered semantic understanding with Sentence-BERT.

## ✨ Key Features

- 🧠 **AI-Powered Semantic Analysis**: State-of-the-art Sentence-BERT for deep meaning understanding
- 📊 **6 Similarity Algorithms**:
  - **Sentence-BERT (Semantic)** - AI-powered meaning detection, paraphrase recognition ⭐ NEW
  - **TF-IDF Cosine** - Statistical similarity with word importance weighting
  - **Jaccard (Word Overlap)** - Exact word matching for duplicate detection
  - **Sequence Matcher** - Order-aware matching for revision tracking
  - **Character Overlap** - Character-level similarity for typo tolerance
  - **Average (All Methods)** - Comprehensive balanced view

- 🎨 **Modern UI**: Clean, intuitive interface with side-by-side text comparison
- 📈 **Comprehensive Results**: 6 metrics displayed simultaneously with visual progress bars
- 🔍 **Interactive Highlighting**: Choose any algorithm to visualize similar content
- 💡 **Smart Interpretation**: Research-based thresholds with automatic paraphrase detection
- 🌍 **Multi-Language Support**: Works with all languages (English, Arabic, Chinese, Japanese, etc.)
- 📖 **Educational**: Built-in explanations based on peer-reviewed research (2015-2025)

## Installation

1. Clone this repository or download the files

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## How It Works

### Algorithms Explained

#### 1. 🧠 Sentence-BERT (Semantic) ⭐ NEW
- **Technology**: Transformer-based neural network (all-mpnet-base-v2 model)
- **How it works**: Generates deep embeddings that capture semantic meaning
- **Strengths**: 
  - Detects paraphrases ("car" ≈ "automobile")
  - Understands context and meaning
  - 15-30% better than statistical methods on semantic tasks
  - 13,000× faster than standard BERT
- **Best for**: Paraphrase detection, content analysis, semantic matching
- **Research**: Based on EMNLP 2019 (Reimers & Gurevych)

#### 2. 🔤 Jaccard (Word Overlap)
- **How it works**: Counts unique words appearing in both texts
- **Formula**: `(Common Words) / (Total Unique Words)`
- **Strengths**: Fast, simple, intuitive
- **Best for**: Duplicate detection, plagiarism (threshold: 0.7-0.9), exact matching
- **Limitation**: Cannot understand meaning - treats "car" and "automobile" as different

#### 3. 📊 TF-IDF Cosine
- **How it works**: Weights words by importance (rare words = more important)
- **Strengths**: Accounts for word frequency and document length
- **Best for**: Document comparison, topic similarity, content recommendation
- **Thresholds**: ≥70% (very similar), 50-69% (similar), 30-49% (related)

#### 4. 📝 Sequence Matcher
- **How it works**: Finds longest contiguous matching sequences
- **Strengths**: Order-aware, good for tracking changes
- **Best for**: Version comparison, revision tracking, change detection

#### 5. 🔡 Character Overlap
- **How it works**: Compares character-level Jaccard similarity
- **Strengths**: Tolerant to typos and spelling variations
- **Best for**: Fuzzy matching, spell checking, name matching

#### 6. ⭐ Average (All Methods)
- **How it works**: Combines all 5 algorithms for balanced view
- **Best for**: General-purpose comparison when unsure which metric to trust

### Research-Based Thresholds

#### Sentence-BERT (Semantic):
- **≥80%**: Semantically very similar (highly related meanings)
- **60-79%**: Semantically similar (related ideas)
- **40-59%**: Somewhat related (some semantic connection)
- **<40%**: Semantically different (different topics)

#### Jaccard (Word Overlap):
- **≥70%**: Near-duplicate (likely copy/plagiarism)
- **50-69%**: High overlap (related versions)
- **25-49%**: Moderate overlap (shared topic)
- **<25%**: Low overlap (distinct texts)

### Paraphrase Detection
The tool automatically detects when **semantic similarity > word overlap by 30%+**, indicating paraphrased content (same meaning, different words).

## 💼 Example Use Cases

### By Algorithm

**🧠 Sentence-BERT (Semantic):**
- Detect paraphrased content or plagiarism with rewording
- Analyze content similarity across different writing styles
- Match job descriptions with resumes (semantic understanding)
- Find semantically similar articles or documents
- Content deduplication when exact wording differs

**🔤 Jaccard (Word Overlap):**
- Plagiarism detection with high threshold (0.7-0.9)
- Find exact or near-exact duplicates
- Mirror site detection (>90% similarity)
- Copyright violation detection

**📊 TF-IDF Cosine:**
- Document categorization and clustering
- Content recommendation systems
- Topic similarity analysis
- Academic paper similarity

**📝 Sequence Matcher:**
- Track document revisions and edits
- Version comparison (before/after)
- Change detection in legal documents
- Quality control for content updates

**🔡 Character Overlap:**
- Fuzzy name matching with typos
- Data deduplication with spelling errors
- OCR error tolerance
- User input validation

### General Use Cases
- ✅ Academic integrity checking
- ✅ Content quality assurance
- ✅ Translation validation
- ✅ SEO duplicate content detection
- ✅ Legal document comparison
- ✅ News article deduplication

## 🛠 Technologies Used

- **Sentence-BERT (sentence-transformers)**: State-of-the-art semantic embeddings
- **PyTorch**: Deep learning backend for SBERT
- **scikit-learn**: TF-IDF vectorization and cosine similarity
- **Python difflib**: Sequence matching algorithms
- **NumPy**: Numerical computations and vector operations
- **Streamlit**: Modern web application framework

## 📋 Requirements

- Python 3.8+
- streamlit >= 1.28.0
- scikit-learn >= 1.3.0
- numpy >= 1.24.0
- sentence-transformers >= 2.2.0 ⭐ NEW
- torch >= 2.0.0 ⭐ NEW

### Installation Size
Note: The first run will download the `all-mpnet-base-v2` model (~420MB). Subsequent runs will use the cached model.

## 🔬 Research Foundation

This tool implements algorithms and thresholds based on peer-reviewed research:

- **Sentence-BERT**: Reimers & Gurevych (EMNLP 2019) - "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
- **Threshold Optimization**: Bettembourg et al. (PLOS ONE 2015) - Systematic threshold determination methodology
- **MTEB Benchmark**: Muennighoff et al. (EACL 2023) - Comprehensive evaluation across 58 datasets
- **Jaccard Analysis**: Multiple studies (2015-2025) on text similarity and plagiarism detection
- **Comparative Studies**: Research showing SBERT achieves 15-30% better performance on semantic tasks

The tool's interpretation and thresholds are based on empirical findings, not arbitrary defaults.

## 🚀 Quick Start

```bash
# Clone the repository
git clone <your-repo-url>
cd text-analysis

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### First Run
On the first run, the app will download the Sentence-BERT model (~420MB). This is a one-time download and subsequent runs will be instant.

## 📸 Screenshots

The tool provides:
- **6 similarity metrics** displayed simultaneously
- **Interactive highlighting** with algorithm selection
- **Automatic paraphrase detection** when semantic similarity exceeds word overlap
- **Research-based interpretations** with threshold explanations

## ⚠️ Important Notes

1. **Threshold Selection**: Research shows the commonly assumed 0.5 threshold lacks empirical support. This tool uses domain-appropriate thresholds validated by peer-reviewed studies.

2. **Algorithm Selection**: 
   - Use **SBERT** for semantic understanding and paraphrase detection
   - Use **Jaccard** for exact duplicate detection (threshold: 0.7-0.9)
   - Use **TF-IDF** for statistical document comparison
   - Compare all metrics for comprehensive analysis

3. **Performance**: First SBERT calculation may take 1-3 seconds. Results are not cached, so re-analyzing requires recalculation.

## 🆕 What's New

### Latest Version
- ✨ **Sentence-BERT Integration**: AI-powered semantic similarity
- 📊 **All algorithms now visible**: TF-IDF Cosine prominently displayed
- 🎨 **Improved UI**: 6 metrics in clean grid layout
- 🔍 **Interactive Highlighting**: Choose algorithm for visual comparison
- 💡 **Smart Interpretation**: Automatic paraphrase detection
- 📖 **Educational Content**: Research-based explanations and citations
- 🌐 **Multi-language**: Works with all languages including Arabic, Chinese, Japanese

## 📝 License

MIT License - Feel free to use and modify as needed!

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

### Areas for Contribution
- Additional similarity algorithms (e.g., Levenshtein, Jaro-Winkler)
- Domain-specific BERT models (SciBERT, Legal-BERT, etc.)
- Performance optimizations
- UI/UX improvements
- Additional language support testing

