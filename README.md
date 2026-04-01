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
