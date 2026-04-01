# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from wordcloud import WordCloud
import seaborn as sns
from textblob import TextBlob
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')

# Correct file path
excel_path = 'PeerJ\sentiment_analysis_output.xlsx'

# Load dataset
try:
    df = pd.read_excel(excel_path, engine='openpyxl')
    df = df.dropna(subset=['comment/tweet'])
    print(f"Loaded file with {len(df)} valid entries.")
except FileNotFoundError:
    print(f"Excel file not found at: {excel_path}")
    df = pd.DataFrame()

# WordCloud function
def generate_word_cloud(ax, texts, title):
    wc = WordCloud(width=400, height=200, background_color='white', min_font_size=10).generate(' '.join(texts))
    ax.imshow(wc, interpolation='bilinear')
    ax.set_title(title, pad=20)  # Added space between title and word cloud
    ax.axis('off')

# Top word function
def get_top_words(texts, n=5):
    words = [w.lower() for text in texts for w in word_tokenize(text) if w.isalpha() and not pd.isna(text)]
    return Counter(words).most_common(n)

# Evaluation function for model training
def evaluate_performance(X, y_true, model_name, clf):
    X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"{model_name}: Acc={accuracy:.3f}, Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}")
    return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1}

# Evaluation function for annotation comparison
def evaluate_annotation_performance(y_true, y_pred, method_pair):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    print(f"{method_pair}: Acc={accuracy:.3f}, Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}")
    return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1}, cm

# Annotation methods
annotation_methods = {
    'Human Annotation': 'majority_sent',
    'TextBlob Annotation': 'textblob_label',
    'M-BERT Annotation': 'mbert_label',
    'Hybrid Annotation': 'hybrid_label'
}
sentiment_map = {'positive': 2, 'negative': 0, 'neutral': 1}

# Proceed if data exists
if not df.empty:
    # WordCloud Grid
    fig, axes = plt.subplots(4, 3, figsize=(15, 20), gridspec_kw={'hspace': 0.4, 'wspace': 0.2})
    for idx, (method, column) in enumerate(annotation_methods.items()):
        for j, sentiment in enumerate(['positive', 'negative', 'neutral']):
            texts = df[df[column] == sentiment]['comment/tweet'].dropna().tolist()
            if texts:
                generate_word_cloud(axes[idx, j], texts, f"{method} - {sentiment.capitalize()}")
            else:
                axes[idx, j].text(0.5, 0.5, 'No data', ha='center', va='center')
    plt.tight_layout()
    plt.show()

    # Top word bar chart (setup only, no plotting)
    plt.figure(figsize=(15, 10))
    for idx, (method, column) in enumerate(annotation_methods.items(), 1):
        for sentiment in ['positive', 'negative', 'neutral']:
            texts = df[df[column] == sentiment]['comment/tweet'].dropna().tolist()
            if texts:
                top_words = dict(get_top_words(texts))
                words, counts = zip(*top_words.items())
                # plt.subplot(4, 3, idx + (['positive', 'negative', 'neutral'].index(sentiment) * 4))
                # plt.bar(words, counts)
                # plt.title(f"{method} - {sentiment.capitalize()}")
                # plt.xticks(rotation=45)
                # plt.tight_layout()
    # plt.show()  # Commented out to avoid empty plot

    # Annotation Comparison
    comparisons = [
        ('Human vs TextBlob', 'majority_sent', 'textblob_label'),
        ('Human vs M-BERT', 'majority_sent', 'mbert_label'),
        ('Human vs Hybrid', 'majority_sent', 'hybrid_label')
    ]
    all_results = {}
    all_cms = {}

    for name, col1, col2 in comparisons:
        y_true = df[col1].map(sentiment_map).values
        y_pred = df[col2].map(sentiment_map).values
        results, cm = evaluate_annotation_performance(y_true, y_pred, name)
        all_results[name] = results
        all_cms[name] = cm

    # Plot Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, (name, cm) in enumerate(all_cms.items()):
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], cmap='Blues')
        axes[idx].set_title(f'Confusion Matrix - {name}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')
    plt.tight_layout()
    plt.show()

    # Plot Bar Graphs for Metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, (name, results) in enumerate(all_results.items()):
        values = [results[m] for m in metrics]
        axes[idx].bar(metrics, values)
        axes[idx].set_title(f'{name} Metrics')
        axes[idx].set_ylim(0, 1.2)
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.05, f'{v:.3f}', ha='center')
    plt.tight_layout()
    plt.show()

    # TF-IDF and Model Evaluation
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    X = tfidf.fit_transform(df['comment/tweet']).toarray()

    classifiers = {
        'SVM': SVC(kernel='linear', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    for method, column in annotation_methods.items():
        y = df[column].map(sentiment_map).values
        results[method] = {}
        for clf_name, clf in classifiers.items():
            results[method][clf_name] = evaluate_performance(X, y, f"{method} with {clf_name}", clf)

    # Radar chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    angles = [n / float(len(metrics)) * 2 * np.pi for n in range(len(metrics))] + [0]

    plt.figure(figsize=(15, 7))
    for i, model_type in enumerate(['SVM', 'Random Forest']):
        ax = plt.subplot(1, 2, i + 1, polar=True)
        for method in annotation_methods:
            vals = [results[method][model_type][m] for m in metrics] + [results[method][model_type][metrics[0]]]
            ax.plot(angles, vals, label=method)
            ax.fill(angles, vals, alpha=0.1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1.2)
        ax.set_title(f"{model_type} Performance")
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    plt.show()

    # Summary table
    summary_data = []
    for method in annotation_methods:
        row = [method]
        for clf in classifiers:
            row += [results[method][clf][m] for m in metrics]
        summary_data.append(row)

    cols = ['Method'] + [f'{clf} {m}' for clf in classifiers for m in metrics]
    summary_df = pd.DataFrame(summary_data, columns=cols)
    print("\n=== Summary Table ===")
    print(summary_df.round(3))
else:
    print("No data to process.")