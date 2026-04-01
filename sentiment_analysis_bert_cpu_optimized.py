import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from collections import Counter
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_path = 'PeerJ\TweeterManglishDS.xlsx'

# Load and preprocess the dataset
try:
    df = pd.read_excel(dataset_path, engine='openpyxl')
    df['comment/tweet'] = df['comment/tweet'].fillna('').astype(str).replace('nan', '')
    df = df[df['majority_sent'].notna() & (df['majority_sent'].str.strip() != '')]
    print(f"Loaded dataset with {len(df)} samples after removing blank labels.")
except FileNotFoundError:
    print(f"File '{dataset_path}' not found. Please ensure the file is in the specified directory in your Google Drive.")
    df = pd.DataFrame()

# Map sentiment labels to numerical values
def map_sentiment_label(label):
    label = str(label).lower().strip()
    if label == 'positive':
        return 2
    elif label == 'negative':
        return 0
    elif label == 'neutral':
        return 1
    else:
        return None

if not df.empty:
    df['actual_label'] = df['majority_sent'].apply(map_sentiment_label)

# Calculate sentiment counts for actual labels
if not df.empty:
    actual_counts = Counter(df['actual_label'])
    actual_sentiment_counts = {'# Positive': actual_counts.get(2, 0), '# Negative': actual_counts.get(0, 0), '# Neutral': actual_counts.get(1, 0)}
    print("\n=== Actual Sentiment Counts ===")
    print(pd.DataFrame([actual_sentiment_counts]))
else:
    actual_sentiment_counts = {'# Positive': 0, '# Negative': 0, '# Neutral': 0}

# Annotate sentiments using TextBlob
def get_textblob_sentiment(texts):
    labels = []
    for text in texts:
        text = str(text) if not pd.isna(text) else ''
        if not text.strip():
            labels.append(1)
            continue
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0:
            label = 2
        elif polarity < 0:
            label = 0
        else:
            label = 1
        labels.append(label)
    return labels

if not df.empty:
    df['textblob_label'] = get_textblob_sentiment(df['comment/tweet'].values)
    textblob_counts = Counter(df['textblob_label'])
    textblob_sentiment_counts = {'# Positive': textblob_counts.get(2, 0), '# Negative': textblob_counts.get(0, 0), '# Neutral': textblob_counts.get(1, 0)}
    print("\n=== TextBlob Sentiment Counts ===")
    print(pd.DataFrame([textblob_sentiment_counts]))
else:
    textblob_sentiment_counts = {'# Positive': 0, '# Negative': 0, '# Neutral': 0}

# Load Multilingual BERT tokenizer and model with CPU support
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
sentiment_model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
sentiment_model.to(device)
sentiment_model.eval()

# Annotate sentiment labels using Multilingual BERT with CPU optimization
def get_bert_sentiment_probs(texts, batch_size=16):  # Reduced batch size for memory efficiency
    probs_list = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_texts = [str(text) if not pd.isna(text) else '' for text in batch_texts]
        inputs = tokenizer(batch_texts, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        mapped_probs = []
        for p in probs:
            negative = p[0] + p[1]  # 1-2 stars
            neutral = p[2]           # 3 stars
            positive = p[3] + p[4]   # 4-5 stars
            mapped_probs.append([negative, neutral, positive])
        probs_list.extend(mapped_probs)
    return np.array(probs_list)

def get_bert_sentiment_labels(probs):
    labels = []
    for p in probs:
        if p[2] > p[0] and p[2] > p[1]:
            labels.append(2)  # Positive
        elif p[1] > p[0] and p[1] > p[2]:
            labels.append(1)  # Neutral
        else:
            labels.append(0)  # Negative
    return labels

if not df.empty:
    probs = get_bert_sentiment_probs(df['comment/tweet'].iloc[:].tolist())
    df['mbert_label'] = get_bert_sentiment_labels(probs)
    mbert_counts = Counter(df['mbert_label'])
    mbert_sentiment_counts = {'# Positive': mbert_counts.get(2, 0), '# Negative': mbert_counts.get(0, 0), '# Neutral': mbert_counts.get(1, 0)}
    print("\n=== M-BERT Sentiment Counts ===")
    print(pd.DataFrame([mbert_sentiment_counts]))

# Hybrid BERT + Rule-Based Approach
def hybrid_sentiment(text, mbert_pred):
    text = str(text).lower()
    manglish_positive_words = ['nice', 'good', 'happy', 'awesome', 'like', 'cantik', 'baik', 'gembira', 'terbaik', 'suka']
    manglish_negative_words = ['sucks', 'bad', 'sad', 'angry', 'hate', 'buruk', 'jahat', 'sedih', 'marah', 'benci']
    if any(word in text for word in manglish_positive_words):
        return 2  # Positive
    elif any(word in text for word in manglish_negative_words):
        return 0  # Negative
    return mbert_pred  # Use M-BERT prediction as fallback

if not df.empty:
    df['hybrid_label'] = [hybrid_sentiment(text, pred) for text, pred in zip(df['comment/tweet'], df['mbert_label'])]
    hybrid_counts = Counter(df['hybrid_label'])
    hybrid_sentiment_counts = {'# Positive': hybrid_counts.get(2, 0), '# Negative': hybrid_counts.get(0, 0), '# Neutral': hybrid_counts.get(1, 0)}
    print("\n=== Hybrid BERT + Rule-Based Sentiment Counts ===")
    print(pd.DataFrame([hybrid_sentiment_counts]))

# # Export to Excel
# if not df.empty:
#     output_df = df[['comment/tweet', 'majority_sent', 'textblob_label', 'mbert_label', 'hybrid_label']].copy()
#     output_df['textblob_label'] = output_df['textblob_label'].map({2: 'positive', 0: 'negative', 1: 'neutral'})
#     output_df['mbert_label'] = output_df['mbert_label'].map({2: 'positive', 0: 'negative', 1: 'neutral'})
#     output_df['hybrid_label'] = output_df['hybrid_label'].map({2: 'positive', 0: 'negative', 1: 'neutral'})
#     output_df.to_excel('/content/sentiment_analysis_output.xlsx', index=False, engine='openpyxl')
#     from google.colab import files
#     print("\nExported results to 'sentiment_analysis_output.xlsx'")
#     files.download('/content/sentiment_analysis_output.xlsx')
# else:
#     print("\nNo data to export.")

# Compute entropy
def compute_entropy(labels):
    label_counts = Counter(labels)
    total = len(labels)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in label_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy

if not df.empty:
    entropies = {
        'Human Annotation': compute_entropy(df['actual_label']),
        'TextBlob Annotation': compute_entropy(df['textblob_label']),
        'M-BERT Annotation': compute_entropy(df['mbert_label']),
        'Hybrid Annotation': compute_entropy(df['hybrid_label'])
    }
    print("\n=== Intrinsic Metrics Summary ===")
    for method, entropy in entropies.items():
        print(f"Entropy ({method}): {entropy:.3f}")

# Create stacked bar graph comparing all annotations
if not df.empty:
    total_samples = len(df)
    actual_positive = actual_sentiment_counts['# Positive'] / total_samples * 100
    actual_negative = actual_sentiment_counts['# Negative'] / total_samples * 100
    actual_neutral = actual_sentiment_counts['# Neutral'] / total_samples * 100
    textblob_positive = textblob_sentiment_counts['# Positive'] / total_samples * 100
    textblob_negative = textblob_sentiment_counts['# Negative'] / total_samples * 100
    textblob_neutral = textblob_sentiment_counts['# Neutral'] / total_samples * 100
    mbert_positive = mbert_sentiment_counts['# Positive'] / total_samples * 100
    mbert_negative = mbert_sentiment_counts['# Negative'] / total_samples * 100
    mbert_neutral = mbert_sentiment_counts['# Neutral'] / total_samples * 100
    hybrid_positive = hybrid_sentiment_counts['# Positive'] / total_samples * 100
    hybrid_negative = hybrid_sentiment_counts['# Negative'] / total_samples * 100
    hybrid_neutral = hybrid_sentiment_counts['# Neutral'] / total_samples * 100

    labels = ['Human Annotation', 'TextBlob Annotation', 'M-BERT Annotation', 'Hybrid Annotation']
    positive = [actual_positive, textblob_positive, mbert_positive, hybrid_positive]
    negative = [actual_negative, textblob_negative, mbert_negative, hybrid_negative]
    neutral = [actual_neutral, textblob_neutral, mbert_neutral, hybrid_neutral]

    plt.figure(figsize=(10, 6))
    bar_width = 0.5
    x = np.arange(len(labels))

    plt.bar(x - bar_width, positive, bar_width, label='Positive', color='red')
    plt.bar(x - bar_width, negative, bar_width, bottom=positive, label='Negative', color='grey')
    plt.bar(x - bar_width, neutral, bar_width, bottom=np.array(positive) + np.array(negative), label='Neutral', color='blue')

    plt.xlabel('Annotation Method')
    plt.ylabel('Percentage of Instances (%)')
    plt.title('Sentiment Distribution Across Annotation Methods')
    plt.xticks(x, labels, rotation=45)
    plt.legend()

    for i in range(len(labels)):
        plt.text(i - bar_width, positive[i]/2, f'{positive[i]:.1f}%', ha='center', va='center', color='white')
        plt.text(i - bar_width, positive[i] + negative[i]/2, f'{negative[i]:.1f}%', ha='center', va='center', color='white')
        plt.text(i - bar_width, positive[i] + negative[i] + neutral[i]/2, f'{neutral[i]:.1f}%', ha='center', va='center', color='white')

    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()

# Create radar chart for entropy comparison
if not df.empty:
    labels = list(entropies.keys())
    values = list(entropies.values())
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='blue', alpha=0.25)
    ax.plot(angles, values, color='blue', linewidth=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0, max(values) * 1.2)
    plt.title('Entropy Comparison Across Annotation Methods')
    plt.tight_layout()
    plt.show()
else:
    print("\nNo data to plot.")