# Text Similarity Analyzer

A powerful Streamlit web application that analyzes and compares two texts to calculate their similarity percentage using multiple algorithms.

## Features

- 📊 **Multiple Similarity Algorithms**:
  - TF-IDF Cosine Similarity (semantic similarity)
  - Sequence Matcher (character-by-character comparison)
  - Jaccard Similarity (word overlap)
  - Character Overlap (character-level comparison)

- 🎨 **Beautiful UI**: Modern, clean interface with intuitive design
- 📈 **Visual Results**: Progress bars and metrics for easy interpretation
- 🔍 **Smart Text Analysis**: Intelligent difference analysis with:
  - 📋 Written summary of all changes
  - 🗺️ Visual difference map showing change distribution
  - 📝 Specific examples with excerpts from both texts
  - Multiple comparison modes (Smart Analysis, Side-by-Side, Word Highlights)
- 💡 **Smart Interpretation**: Automatic interpretation of similarity scores

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

### Algorithms

1. **TF-IDF Cosine Similarity**
   - Converts texts to TF-IDF vectors
   - Calculates cosine similarity between vectors
   - Best for: Understanding semantic similarity

2. **Sequence Matcher**
   - Uses Python's difflib library
   - Compares texts character by character
   - Best for: Detecting edits and changes

3. **Jaccard Similarity**
   - Compares unique words in both texts
   - Calculates intersection over union
   - Best for: Understanding word overlap

4. **Character Overlap**
   - Compares unique characters
   - Calculates character set similarity
   - Best for: Character-level analysis

### Interpretation Guide

- **90-100%**: Very High Similarity - Texts are nearly identical
- **70-89%**: High Similarity - Substantial content overlap
- **50-69%**: Moderate Similarity - Some common elements
- **30-49%**: Low Similarity - Limited overlap
- **0-29%**: Very Low Similarity - Quite different

## Example Use Cases

- Compare different versions of documents
- Detect plagiarism or content similarity
- Analyze text revisions and changes
- Compare translations or paraphrases
- Quality control for content rewriting

## Technologies Used

- **Streamlit**: Web application framework
- **scikit-learn**: TF-IDF vectorization and cosine similarity
- **Python difflib**: Sequence matching
- **NumPy**: Numerical computations

## Requirements

- Python 3.7+
- streamlit >= 1.28.0
- scikit-learn >= 1.3.0
- numpy >= 1.24.0

## License

MIT License - Feel free to use and modify as needed!

## Contributing

Contributions, issues, and feature requests are welcome!

