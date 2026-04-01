## Title
The Potential of Machine Annotation to Replace Human 
Annotation in Manglish Sentiment Analysis: Opportunities and 
Challenges 

## Description
This repository contains a complete sentiment analysis framework for **Manglish (Malay-English code-mixed) Twitter data**.  
The project compares multiple annotation strategies, including:

- Human annotation
- Rule-based sentiment analysis using TextBlob
- Multilingual BERT (M-BERT)
- Hybrid BERT + rule-based sentiment annotation

The repository also includes additional analysis pipelines for:

- Sentiment distribution comparison
- Entropy-based intrinsic evaluation
- Word cloud visualization
- Confusion matrix analysis
- TF-IDF feature extraction
- Traditional machine learning benchmarking (SVM and Random Forest)

This work supports research on **code-mixed language sentiment analysis**, particularly for **Manglish social media datasets**.

## Dataset Information
The repository includes the following datasets:

### 1. `TweeterManglishDS.xlsx`
Primary Manglish Twitter dataset containing:

- `id`
- `post/keyword`
- `comment/tweet` → tweet text
- `username`
- `like count`
- `replied`
- `replied count`
- `2nd level comment/ reply`
- `time created`
- `majority_sent` → gold-standard human sentiment labels
  - Labels:
    - `positive`
    - `negative`
    - `neutral` 
- `majority_sarc`
- `lang_id`

### 2. `sentiment_analysis_output.xlsx`
Generated output dataset after automated annotation containing:

- comment/tweet
- majority_sent(Human labels)
- textblob_label (TextBlob predictions)
- mbert_label (M-BERT predictions)
- hybrid_label (Hybrid predictions)

---

## Code Information
The repository contains two main Python scripts:

### 1. `sentiment_analysis_bert_cpu_optimized.py`
Main annotation and intrinsic analysis pipeline.

### Features
- Dataset loading and cleaning
- Human label preprocessing
- TextBlob sentiment annotation
- Multilingual BERT sentiment inference
- Hybrid rule-enhanced BERT annotation
- Entropy computation
- Stacked sentiment distribution chart
- Radar chart for entropy comparison

---

### 2. `sentiment_analysis_bert_cpu_optimized_more_analysis.py`
Extended exploratory analysis and benchmarking pipeline.

### Features
- Word cloud generation
- Top frequent words extraction
- Confusion matrix plotting
- Accuracy, precision, recall, and F1-score evaluation
- TF-IDF vectorization
- SVM classifier benchmarking
- Random Forest benchmarking
- Radar chart model comparison
- Summary performance table

---

## Usage Instructions

### Step 1: Clone the repository

git clone https://github.com/yourusername/manglish-sentiment-analysis.git
cd manglish-sentiment-analysis

### Step 2: Install dependencies
- pandas
- numpy
- matplotlib
- seaborn
- torch
- transformers
- textblob
- openpyxl
- scikit-learn
- nltk
- wordcloud

### Step 3: place dataset files
- TweeterManglishDS.xlsx
- sentiment_analysis_output.xlsx
### Step 4: run the main annotation pipeline
- python sentiment_analysis_bert_cpu_optimized.py
  
### Step 5: Run extended analysis
- python sentiment_analysis_bert_cpu_optimized_more_analysis.py

## Requirements
- pandas
- numpy
- matplotlib
- seaborn
- torch
- transformers
- textblob
- openpyxl
- scikit-learn
- nltk
- wordcloud

## Methodology

The project follows this workflow:

1. Data Loading and Cleaning
  - Remove null tweets
  - Standardize sentiment labels
2. Human Annotation Processing
  - Convert categorical labels into numerical classes
3. Automated Annotation
  - TextBlob polarity-based labeling
  - M-BERT multilingual sentiment classification
  - Hybrid lexicon-enhanced M-BERT refinement
4. Intrinsic Evaluation
  - Sentiment distribution comparison
  - Entropy analysis
5. Extrinsic Evaluation
  - Compare annotations against human labels
  - Train SVM and Random Forest using TF-IDF features
  - Evaluate using Accuracy, Precision, Recall, and F1-score
6. Visualization
  - Stacked bar plots
  - Radar charts
  - Word clouds
  - Confusion matrices

## Citation
- M. S. Md Suhaimin, M. H. Ahmad Hijazi, and E. G. Moung, “Annotated dataset for sentiment 
analysis and sarcasm detection: Bilingual code-mixed English-Malay social media data in the public 
security domain,” Data in Brief, vol. 55, p. 110663, Aug. 2024, doi: 10.1016/j.dib.2024.110663.
