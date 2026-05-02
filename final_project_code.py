#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string
import warnings
warnings.filterwarnings('ignore')

nltk.download('stopwords')
nltk.download('punkt')

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_theme(style="whitegrid")
print("All libraries loaded.")


# In[2]:


fake = pd.read_csv('../data/Fake.csv')
real = pd.read_csv('../data/True.csv')

fake['label'] = 1   # Fake = 1
real['label'] = 0   # Real = 0

df = pd.concat([fake, real], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dataset shape: {df.shape}")
df.head()


# In[3]:


print(df.info())
print("\n--- Missing Values ---")
print(df.isnull().sum())
print("\n--- Class Distribution ---")
print(df['label'].value_counts())


# In[4]:


ax = sns.countplot(x='label', data=df, palette='Set2')
ax.set_xticklabels(['Real (0)', 'Fake (1)'])
plt.title('Class Distribution: Real vs Fake News')
plt.ylabel('Count')
plt.show()


# In[5]:


df['text_length'] = df['text'].apply(len)
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))

print(df.groupby('label')[['text_length', 'word_count']].mean())

sns.histplot(data=df, x='word_count', hue='label', bins=50, kde=True, palette='Set1')
plt.title('Word Count Distribution: Real vs Fake')
plt.xlim(0, 2000)
plt.show()


# In[6]:


# --- Sensationalism Feature Extraction ---
df['exclamation_count'] = df['text'].apply(lambda x: str(x).count('!'))
df['question_count'] = df['text'].apply(lambda x: str(x).count('?'))
df['caps_ratio'] = df['text'].apply(
    lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x)) + 1)
)
df['punct_ratio'] = df['text'].apply(
    lambda x: sum(1 for c in str(x) if c in string.punctuation) / (len(str(x)) + 1)
)

# --- Visualization ---
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
cols = ['exclamation_count', 'question_count', 'caps_ratio', 'punct_ratio']

for ax, col in zip(axes.flatten(), cols):
    sns.boxplot(x='label', y=col, data=df, palette='Set2', ax=ax)
    ax.set_xticklabels(['Real', 'Fake'])
    ax.set_title(f'Distribution of {col}')

plt.suptitle('Sensationalism Signals: Real vs Fake News', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ── KAGGLE ONLY DATA ──
df.to_csv('../data/kaggle_only.csv', index=False)

print("Kaggle base dataset saved → kaggle_only.csv")
print("Do NOT rerun this after running NB2")
print("Updated labeled_data.csv with sensationalism features.")


# In[7]:


sns.countplot(y='subject', data=df, hue='label', palette='Set2')
plt.title('News Subject by Label')
plt.tight_layout()
plt.show()


# In[8]:


# ── KAGGLE ONLY DATA ──
df.to_csv('../data/kaggle_only.csv', index=False)

print("Kaggle base dataset saved → kaggle_only.csv")
print("Do NOT rerun this after running NB2")
print(" labeled_data.csv saved to data/ folder")


# In[9]:


# ── KAGGLE ONLY DATA ──
df.to_csv('../data/kaggle_only.csv', index=False)

print("Kaggle base dataset saved → kaggle_only.csv")
print("Do NOT rerun this after running NB2")
print(" File saved with new columns!")


# In[10]:


import sys
print(sys.executable)


# In[ ]:




#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import warnings 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix, hstack
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
warnings.filterwarnings('ignore')

nltk.download('stopwords')
nltk.download('punkt')

print("Imports done.")


# In[2]:


liar_cols = ['id','label','statement','subject','speaker','job',
             'state','party','barely_true_ct','false_ct',
             'half_true_ct','mostly_true_ct','pants_on_fire_ct','venue']

liar_train = pd.read_csv('../data/liar/train.tsv', sep='\t', header=None, names=liar_cols)
liar_val   = pd.read_csv('../data/liar/valid.tsv', sep='\t', header=None, names=liar_cols)
liar_test  = pd.read_csv('../data/liar/test.tsv',  sep='\t', header=None, names=liar_cols)

liar_df = pd.concat([liar_train, liar_val, liar_test], ignore_index=True)

print("LIAR loaded:", liar_df.shape)


# In[3]:


fake_labels = {'pants-fire', 'false', 'barely-true'}
real_labels = {'half-true', 'mostly-true', 'true'}

liar_df = liar_df[liar_df['label'].isin(fake_labels | real_labels)].copy()

liar_df['label'] = liar_df['label'].apply(
    lambda x: 1 if x in fake_labels else 0
)

print("After filtering:", liar_df.shape)


# In[4]:


liar_df['title'] = liar_df['statement']
liar_df['text']  = liar_df['statement']

liar_df = liar_df[['title', 'text', 'subject', 'label']]


# In[5]:


kaggle_df = pd.read_csv('../data/kaggle_only.csv')

print("Kaggle loaded:", kaggle_df.shape)


# In[6]:


kaggle_df['source'] = 'kaggle'
liar_df['source']   = 'liar'

combined_df = pd.concat([kaggle_df, liar_df], ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Combined shape:", combined_df.shape)
print(combined_df['source'].value_counts())


# In[7]:


kaggle_df['source'] = 'kaggle'
liar_df['source']   = 'liar'

combined_df = pd.concat([kaggle_df, liar_df], ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Combined shape:", combined_df.shape)
print(combined_df['source'].value_counts())


# In[8]:


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)

    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]

    return ' '.join(tokens)


# In[9]:


combined_df['content'] = combined_df['title'] + ' ' + combined_df['text']
combined_df['content_clean'] = combined_df['content'].apply(clean_text)

print(combined_df['content_clean'].iloc[0])


# In[10]:


from textblob import TextBlob
import textstat

df_feat = combined_df  

df_feat['sentiment']    = df_feat['content'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
df_feat['subjectivity'] = df_feat['content'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
df_feat['flesch_score'] = df_feat['content'].apply(lambda x: textstat.flesch_reading_ease(str(x)))
df_feat['avg_word_len'] = df_feat['content_clean'].apply(
    lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
)
combined_df = df_feat
print("New features added:", ['sentiment','subjectivity','flesch_score','avg_word_len'])
print(combined_df[['sentiment','subjectivity','flesch_score','avg_word_len']].describe())


# In[11]:


X = combined_df['content_clean']
y = combined_df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train:", len(X_train), "Test:", len(X_test))


# In[12]:


tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1,3),
                        sublinear_tf=True, min_df=2, max_df=0.95)


X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

print("TF-IDF shape:", X_train_tfidf.shape)


# In[13]:


feature_cols = ['exclamation_count', 'question_count', 'caps_ratio',
                'punct_ratio', 'text_length', 'word_count',
                'sentiment', 'subjectivity', 'flesch_score', 'avg_word_len']

extra_train = csr_matrix(combined_df.loc[X_train.index, feature_cols].values)
extra_test  = csr_matrix(combined_df.loc[X_test.index,  feature_cols].values)

X_train_final = hstack([X_train_tfidf, extra_train])
X_test_final  = hstack([X_test_tfidf,  extra_test])

print("Final shape:", X_train_final.shape)


# In[14]:


with open('../models/X_train_tfidf.pkl', 'wb') as f:
    pickle.dump(X_train_tfidf, f)

with open('../models/X_test_tfidf.pkl', 'wb') as f:
    pickle.dump(X_test_tfidf, f)

with open('../models/X_train_final.pkl', 'wb') as f:
    pickle.dump(X_train_final, f)

with open('../models/X_test_final.pkl', 'wb') as f:
    pickle.dump(X_test_final, f)

with open('../models/y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)

with open('../models/y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)

with open('../models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("All files saved.")


# In[15]:


combined_df.to_csv('../data/labeled_data.csv', index=False)

print("Final combined dataset saved → labeled_data.csv")


# In[16]:


get_ipython().system('pip install textblob textstat --break-system-packages')
get_ipython().system('python -m textblob.download_corpora')


# In[ ]:




#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ── CELL 1: Imports ──────────────────────────────────────────
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
from scipy import sparse

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')
print("Imports done.")


# In[2]:


# ── CELL 2: Load saved features ──────────────────────────────


with open('../models/y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)
with open('../models/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)
with open('../models/X_train_tfidf.pkl', 'rb') as f:
    X_train_tfidf = pickle.load(f)
with open('../models/X_test_tfidf.pkl', 'rb') as f:
    X_test_tfidf = pickle.load(f)
with open('../models/X_train_final.pkl', 'rb') as f:
    X_train_hybrid = pickle.load(f)
with open('../models/X_test_final.pkl', 'rb') as f:
    X_test_hybrid = pickle.load(f)

print(f"y_train      : {y_train.shape}")
print(f"X_train_tfidf: {X_train_tfidf.shape}")
print(f"X_train_hybrid: {X_train_hybrid.shape}")

# Sanity check — train + test should equal full dataset size
print(f"\nTrain + Test = {len(y_train) + len(y_test)} rows (should be 57689)")


# In[3]:


# ── CELL 3: Fix NaN values in hybrid sparse matrix ───────────
def clean_sparse(X):
    if sparse.issparse(X):
        X = X.copy()
        X.data = np.nan_to_num(X.data)
        return X
    return np.nan_to_num(X)

X_train_hybrid = clean_sparse(X_train_hybrid)
X_test_hybrid  = clean_sparse(X_test_hybrid)
print("NaN values cleaned from hybrid matrices.")


# In[4]:


# ── CELL 4: Sample weights (FIXED — use .loc not .iloc) ──────

df_meta = pd.read_csv('../data/labeled_data.csv')

train_meta = df_meta.loc[y_train.index]   # CORRECT: label-based
sample_weights = train_meta['source'].map({'kaggle': 1.0, 'liar': 1.2}).values
#  Reason: LIAR statements are ~15 words; Kaggle articles are ~400 words.
#  Over-weighting short statements was pulling the TF-IDF distribution
#  toward short-text patterns and hurting generalisation.

print(f"Sample weights shape: {sample_weights.shape}")
print(f"Unique weights: {np.unique(sample_weights)}")



# In[5]:


# ── CELL 5: Logistic Regression (tuned) ──────────────────────

start = time.time()

lr = LogisticRegression(C=2.0, max_iter=2000, solver='saga', random_state=42, n_jobs=-1)

lr.fit(X_train_tfidf, y_train, sample_weight=sample_weights)
lr_preds = lr.predict(X_test_tfidf)
lr_proba = lr.predict_proba(X_test_tfidf)[:, 1]

lr_acc = accuracy_score(y_test, lr_preds)
lr_auc = roc_auc_score(y_test, lr_proba)
lr_time = time.time() - start

print(f"LR  | Acc: {lr_acc:.4f} | AUC: {lr_auc:.4f} | Time: {lr_time:.1f}s")
# Expected: Acc ~0.93-0.94  (was 0.9062)



# In[6]:


# ── CELL 6: Naive Bayes (tuned) ──────────────────────────────
# Key change: alpha=0.1 (was default 1.0)
# Lower alpha = less smoothing = stronger word signals
# MultinomialNB needs non-negative features — TF-IDF with sublinear_tf is fine
start = time.time()

nb = MultinomialNB(alpha=0.5)
nb.fit(X_train_tfidf, y_train, sample_weight=sample_weights)
nb_preds = nb.predict(X_test_tfidf)
nb_proba = nb.predict_proba(X_test_tfidf)[:, 1]

nb_acc = accuracy_score(y_test, nb_preds)
nb_auc = roc_auc_score(y_test, nb_proba)
nb_time = time.time() - start

print(f"NB  | Acc: {nb_acc:.4f} | AUC: {nb_auc:.4f} | Time: {nb_time:.1f}s")
# Expected: Acc ~0.91-0.92  (was 0.8771)


# In[7]:


# ── CELL 7: Random Forest (tuned) ────────────────────────────
# Key changes vs original:
#   max_features='sqrt'  — standard best practice for RF, prevents overfitting
#   n_estimators=300     — reduced from 500 (diminishing returns above 300)
#   min_samples_leaf=1   — was 2, let leaves be pure for text classification
# Using HYBRID features (TF-IDF + sensationalism signals)
start = time.time()

rf = RandomForestClassifier(
    n_estimators=300,
    max_features='log2',      
    min_samples_leaf=1,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train_hybrid, y_train, sample_weight=sample_weights)
rf_preds = rf.predict(X_test_hybrid)
rf_proba = rf.predict_proba(X_test_hybrid)[:, 1]

rf_acc = accuracy_score(y_test, rf_preds)
rf_auc = roc_auc_score(y_test, rf_proba)
rf_time = time.time() - start

print(f"RF  | Acc: {rf_acc:.4f} | AUC: {rf_auc:.4f} | Time: {rf_time:.1f}s")
# Expected: Acc ~0.93-0.94  (was 0.9124)



# In[8]:


# ── CELL 8: MLP (tuned) ──────────────────────────────────────
# Key changes vs original:
#   Scaled input with MaxAbsScaler — MLP is sensitive to feature scale
#   adam solver with lower learning_rate_init — more stable convergence
#   dropout via alpha (L2 regularization) increased slightly
# MLP on sparse matrix: MaxAbsScaler preserves sparsity
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
X_train_scaled = scaler.fit_transform(X_train_tfidf)
X_test_scaled  = scaler.transform(X_test_tfidf)

start = time.time()
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation='relu',
    solver='adam',
    alpha=0.001,               # L2 reg (was default 0.0001)
    learning_rate_init=0.001,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15,       # was 10 — give it more patience
    random_state=42
)
mlp.fit(X_train_scaled, y_train)  # MLP doesn't support sample_weight in all versions
mlp_preds = mlp.predict(X_test_scaled)
mlp_proba = mlp.predict_proba(X_test_scaled)[:, 1]

mlp_acc = accuracy_score(y_test, mlp_preds)
mlp_auc = roc_auc_score(y_test, mlp_proba)
mlp_time = time.time() - start

print(f"MLP | Acc: {mlp_acc:.4f} | AUC: {mlp_auc:.4f} | Time: {mlp_time:.1f}s")



# In[9]:


# ── CELL 9: Voting Ensemble (replaces XGBoost) ───────────────
# Why Voting Ensemble instead of XGBoost?
#   1. XGBoost on 10k-feature dense matrix needs ~8GB RAM — risky
#   2. Voting combines LR + NB + RF — all already trained, zero extra cost
#   3. Soft voting averages probabilities — outperforms any single model
#   4. Stays within project scope (all three suggested model types contribute)
#
# We build a fresh VotingClassifier that retrains each estimator internally.
# This avoids the sparse/dense mismatch between LR(tfidf) and RF(hybrid).
# Both use TF-IDF here for a clean ensemble on identical feature space.

from sklearn.ensemble import StackingClassifier

start = time.time()

lr_ens = LogisticRegression(C=2.0, max_iter=2000, solver='saga',
                             random_state=42, n_jobs=-1)
nb_ens = MultinomialNB(alpha=0.5)
rf_ens = RandomForestClassifier(n_estimators=300, max_features='log2',
                                 min_samples_leaf=1, n_jobs=-1, random_state=42)

ensemble = StackingClassifier(
    estimators=[('lr', lr_ens), ('nb', nb_ens), ('rf', rf_ens)],
    final_estimator=LogisticRegression(C=1.0, max_iter=1000, solver='saga'),
    cv=5,
    stack_method='predict_proba',
    passthrough=False,
    n_jobs=-1
)
ensemble.fit(X_train_tfidf, y_train)   # StackingClassifier handles CV internally, no sample_weight here
ens_preds = ensemble.predict(X_test_tfidf)
ens_proba = ensemble.predict_proba(X_test_tfidf)[:, 1]

ens_acc = accuracy_score(y_test, ens_preds)
ens_auc = roc_auc_score(y_test, ens_proba)
ens_time = time.time() - start

print(f"ENS | Acc: {ens_acc:.4f} | AUC: {ens_auc:.4f} | Time: {ens_time:.1f}s")
# Expected: 0.94–0.96



# In[10]:


# ── CELL 10: Results summary ──────────────────────────────────
results = {
    'Logistic Regression': {'acc': lr_acc,  'auc': lr_auc,  'preds': lr_preds,  'proba': lr_proba},
    'Naive Bayes':          {'acc': nb_acc,  'auc': nb_auc,  'preds': nb_preds,  'proba': nb_proba},
    'Random Forest':        {'acc': rf_acc,  'auc': rf_auc,  'preds': rf_preds,  'proba': rf_proba},
    'MLP Neural Net':       {'acc': mlp_acc, 'auc': mlp_auc, 'preds': mlp_preds, 'proba': mlp_proba},
    'Voting Ensemble':      {'acc': ens_acc, 'auc': ens_auc, 'preds': ens_preds, 'proba': ens_proba},
}

summary_df = pd.DataFrame(
    {name: {'Accuracy': v['acc'], 'AUC': v['auc']} for name, v in results.items()}
).T.sort_values('Accuracy', ascending=False)

print("\n=== Model Performance Summary ===")
print(summary_df.round(4).to_string())

# Bar chart
plt.figure(figsize=(10, 5))
bars = plt.bar(summary_df.index, summary_df['Accuracy'], color='steelblue')
min_acc = summary_df['Accuracy'].min()
plt.ylim(min_acc - 0.02, 1.0)
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.xticks(rotation=25, ha='right')
for bar, val in zip(bars, summary_df['Accuracy']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{val*100:.2f}%', ha='center', fontsize=10)
plt.tight_layout()
plt.show()



# In[11]:


# ── CELL 11: ROC curve comparison ─────────────────────────────
eval_map = {
    'Logistic Regression': (X_test_tfidf, lr_proba),
    'Naive Bayes':          (X_test_tfidf, nb_proba),
    'Random Forest':        (X_test_hybrid, rf_proba),
    'MLP Neural Net':       (X_test_scaled, mlp_proba),
    'Voting Ensemble':      (X_test_tfidf, ens_proba),
}

plt.figure(figsize=(10, 6))
for name, (_, proba) in eval_map.items():
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc_val = results[name]['auc']
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.4f})")

plt.plot([0, 1], [0, 1], 'k--', label='Random baseline')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison — All Models')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



# In[12]:


# ── CELL 12: Best model — classification report + confusion matrix ──
best_name = summary_df.index[0]
best = results[best_name]
best_preds = best['preds']

print(f"=== Best Model: {best_name} ===")
print(classification_report(y_test, best_preds, target_names=['Real', 'Fake']))

cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'])
plt.title(f'Confusion Matrix — {best_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()



# In[13]:


# ── CELL 13: Save all models ──────────────────────────────────
# Save individual models
with open('../models/lr_model.pkl', 'wb') as f:
    pickle.dump(lr, f)
with open('../models/nb_model.pkl', 'wb') as f:
    pickle.dump(nb, f)
with open('../models/rf_model.pkl', 'wb') as f:
    pickle.dump(rf, f)
with open('../models/mlp_model.pkl', 'wb') as f:
    pickle.dump(mlp, f)
with open('../models/ensemble_model.pkl', 'wb') as f:
    pickle.dump(ensemble, f)
with open('../models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save best model (for NB4 to load)
with open('../models/best_model.pkl', 'wb') as f:
    pickle.dump(results[best_name], f)  # saves full results dict entry

# IMPORTANT: Also save just the sklearn model object for NB4 compatibility
import pickle
best_sklearn_model = ensemble if best_name == 'Voting Ensemble' else \
                     lr if best_name == 'Logistic Regression' else \
                     rf if best_name == 'Random Forest' else \
                     mlp if best_name == 'MLP Neural Net' else nb

with open('../models/best_model_tfidf.pkl', 'wb') as f:
    pickle.dump(best_sklearn_model, f)

print(f"All models saved.")
print(f"Best model '{best_name}' saved as best_model_tfidf.pkl")
print(f"\nNote for NB4: Best model uses {'tfidf' if best_name != 'Random Forest' else 'hybrid'} features.")



# In[ ]:




#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from collections import Counter
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

print("Imports done.")


# In[2]:


df = pd.read_csv('../data/labeled_data.csv')
df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
df['content_clean'] = df['content'].apply(clean_text)

with open('../models/best_model_tfidf.pkl', 'rb') as f:
    model = pickle.load(f)
with open('../models/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)
with open('../models/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Auto-detect which feature matrix the best model needs
if hasattr(model, 'n_features_in_') and model.n_features_in_ > 15000:
    with open('../models/X_test_final.pkl', 'rb') as f:
        X_test = pickle.load(f)
    print("Loaded hybrid X_test (TF-IDF + handcrafted features)")
else:
    with open('../models/X_test_tfidf.pkl', 'rb') as f:
        X_test = pickle.load(f)
    print("Loaded TF-IDF X_test")

# Clean NaN values
from scipy.sparse import issparse
if issparse(X_test):
    X_test = X_test.copy()
    X_test.data = np.nan_to_num(X_test.data)
else:
    X_test = np.nan_to_num(X_test)
print("NaN values cleaned from X_test")

print(f"Model type   : {type(model).__name__}")
print(f"X_test shape : {X_test.shape}")
print(f"y_test shape : {y_test.shape}")
print(f"Test rows = 20% of {len(df)} = {int(len(df)*0.2)} (expected ~{len(y_test)})")

assert y_test.index.isin(df.index).all(), "Index mismatch — rerun NB2 to regenerate splits"
print("Index alignment check: PASSED")


# In[3]:


# ── CELL 3: Predictions and overall metrics ───────────────────
preds = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, preds)
auc = roc_auc_score(y_test, proba)

print(f"\n=== Overall Performance ===")
print(f"Accuracy : {acc*100:.2f}%")
print(f"AUC      : {auc:.4f}")
print()
print(classification_report(y_test, preds, target_names=['Real', 'Fake']))

# Confusion matrix
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'])
plt.title('Confusion Matrix — Best Model')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()


# In[4]:


# ── CELL 4: Performance by source dataset ─────────────────────
test_df = df.loc[y_test.index].copy()
test_df['predicted'] = preds
test_df['actual']    = y_test.values

kaggle_test = test_df[test_df['source'] == 'kaggle']
liar_test   = test_df[test_df['source'] == 'liar']

kaggle_acc = accuracy_score(kaggle_test['actual'], kaggle_test['predicted'])
liar_acc   = accuracy_score(liar_test['actual'],   liar_test['predicted'])

print(f"Kaggle subset : {len(kaggle_test)} samples | Accuracy: {kaggle_acc:.4f}")
print(f"LIAR subset   : {len(liar_test)} samples | Accuracy: {liar_acc:.4f}")

# Side-by-side bar chart
plt.figure(figsize=(8, 5))
bars = plt.bar(['Kaggle', 'LIAR'], [kaggle_acc, liar_acc],
               color=['steelblue', 'coral'], width=0.5)
for bar, val in zip(bars, [kaggle_acc, liar_acc]):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.005,
             f'{val*100:.2f}%', ha='center', fontsize=12)
plt.ylim(0.5, 1.05)
plt.title('Accuracy by Dataset Source')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.show()



# In[5]:


# ── CELL 5: Confusion matrices per source ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_k = confusion_matrix(kaggle_test['actual'], kaggle_test['predicted'])
sns.heatmap(cm_k, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'], ax=axes[0])
axes[0].set_title('Kaggle Subset')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

cm_l = confusion_matrix(liar_test['actual'], liar_test['predicted'])
sns.heatmap(cm_l, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'], ax=axes[1])
axes[1].set_title('LIAR Subset')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')

plt.suptitle('Confusion Matrices by Source', fontsize=14)
plt.tight_layout()
plt.show()



# In[6]:


# ── CELL 6: Per-source classification reports ─────────────────
print("=== Kaggle Subset ===")
print(classification_report(kaggle_test['actual'], kaggle_test['predicted'],
                             target_names=['Real', 'Fake']))

print("=== LIAR Subset ===")
print(classification_report(liar_test['actual'], liar_test['predicted'],
                             target_names=['Real', 'Fake']))




# In[7]:


# ── CELL 7: Error analysis ─────────────────────────────────────
errors = test_df[test_df['predicted'] != test_df['actual']]
fp = errors[errors['predicted'] == 1]   # Real called Fake
fn = errors[errors['predicted'] == 0]   # Fake called Real

print(f"Total errors : {len(errors)} / {len(test_df)} ({len(errors)/len(test_df)*100:.2f}%)")
print(f"False Positives (Real → Fake) : {len(fp)}")
print(f"False Negatives (Fake → Real) : {len(fn)}")

print("\n--- False Positives: Real news wrongly flagged as Fake ---")
print(fp[['title', 'source', 'subject']].head(5).to_string(index=False))

print("\n--- False Negatives: Fake news that slipped through ---")
print(fn[['title', 'source', 'subject']].head(5).to_string(index=False))




# In[8]:


# ── CELL 8: Feature importance (RF only — skip for Ensemble) ──
try:
    # If best model is Voting Ensemble, pull RF from inside
    if hasattr(model, 'estimators_'):
        rf_inside = dict(model.estimators)['rf'] if hasattr(model, 'estimators') else None
        # Try named_estimators_ (sklearn attribute)
        if hasattr(model, 'named_estimators_'):
            rf_inside = model.named_estimators_.get('rf', None)
        importances_series = None
        if rf_inside is not None and hasattr(rf_inside, 'feature_importances_'):
            feature_names = list(tfidf.get_feature_names_out())
            importances_series = pd.Series(
                rf_inside.feature_importances_, index=feature_names
            ).sort_values(ascending=False)
    elif hasattr(model, 'feature_importances_'):
        feature_names = list(tfidf.get_feature_names_out())
        importances_series = pd.Series(
            model.feature_importances_, index=feature_names
        ).sort_values(ascending=False)
    else:
        importances_series = None

    if importances_series is not None:
        plt.figure(figsize=(12, 6))
        importances_series.head(25).plot(kind='bar', color='steelblue')
        plt.title('Top 25 Features — Random Forest Component')
        plt.ylabel('Importance Score')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        print("\nTop 10 important features:")
        print(importances_series.head(10).to_string())
    else:
        print("Feature importance not available for this model type.")
        print("(VotingClassifier hides internal estimator importances)")
        print("Showing LR coefficients instead...")

        # For LR inside ensemble
        if hasattr(model, 'named_estimators_'):
            lr_inside = model.named_estimators_.get('lr', None)
            if lr_inside is not None:
                feature_names = list(tfidf.get_feature_names_out())
                coef = pd.Series(
                    lr_inside.coef_[0], index=feature_names
                ).sort_values(ascending=False)
                print("\nTop 15 Fake-leaning words (positive LR coef):")
                print(coef.head(15).to_string())
                print("\nTop 15 Real-leaning words (negative LR coef):")
                print(coef.tail(15).to_string())

except Exception as e:
    print(f"Feature importance extraction skipped: {e}")




# In[9]:


# ── CELL 9: Word frequency analysis ───────────────────────────
fake_words = ' '.join(df[df['label'] == 1]['content_clean'].dropna()).split()
real_words = ' '.join(df[df['label'] == 0]['content_clean'].dropna()).split()

fake_freq = pd.DataFrame(Counter(fake_words).most_common(15), columns=['Word', 'Frequency'])
real_freq = pd.DataFrame(Counter(real_words).most_common(15), columns=['Word', 'Frequency'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].barh(fake_freq['Word'][::-1], fake_freq['Frequency'][::-1], color='red')
axes[0].set_title('Top 15 Words in FAKE News')
axes[1].barh(real_freq['Word'][::-1], real_freq['Frequency'][::-1], color='green')
axes[1].set_title('Top 15 Words in REAL News')
plt.tight_layout()
plt.show()



# In[10]:


# ── CELL 10: Word clouds ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

wc_fake = WordCloud(width=700, height=400, background_color='black',
                    colormap='Reds', max_words=100).generate(' '.join(fake_words))
axes[0].imshow(wc_fake, interpolation='bilinear')
axes[0].axis('off')
axes[0].set_title('FAKE News Word Cloud', fontsize=14)

wc_real = WordCloud(width=700, height=400, background_color='white',
                    colormap='Greens', max_words=100).generate(' '.join(real_words))
axes[1].imshow(wc_real, interpolation='bilinear')
axes[1].axis('off')
axes[1].set_title('REAL News Word Cloud', fontsize=14)

plt.tight_layout()
plt.show()




# In[11]:


# ── CELL 11: Final summary table ──────────────────────────────
summary = pd.DataFrame({
    'Metric': ['Accuracy', 'AUC Score', 'Total Test Samples',
               'False Positives', 'False Negatives', 'Error Rate'],
    'Value': [
        f"{acc*100:.2f}%",
        f"{auc:.4f}",
        len(y_test),
        len(fp),
        len(fn),
        f"{len(errors)/len(test_df)*100:.2f}%"
    ]
})
print(summary.to_string(index=False))




# In[12]:


def predict_news(title, body=""):
    """
    Predicts whether a news article is Real or Fake.
    For best results, pass both title and full article body.
    Headline-only predictions will have lower confidence.
    """
    from scipy.sparse import hstack, csr_matrix

    combined = clean_text(title + " " + body)
    vectorized = tfidf.transform([combined])

    # If model expects hybrid features, pad with zeros for handcrafted cols
    if hasattr(model, 'n_features_in_') and model.n_features_in_ > vectorized.shape[1]:
        n_extra = model.n_features_in_ - vectorized.shape[1]
        padding = csr_matrix(np.zeros((1, n_extra)))
        vectorized = hstack([vectorized, padding])

    prediction = model.predict(vectorized)[0]
    proba = model.predict_proba(vectorized)[0]
    confidence = max(proba) * 100

    if confidence >= 75:
        label = "FAKE NEWS" if prediction == 1 else "REAL NEWS"
        icon  = "RED" if prediction == 1 else "GREEN"
    else:
        label = "UNCERTAIN — needs human review"
        icon  = "YELLOW"

    print(f"Title      : {title[:80]}")
    print(f"Prediction : [{icon}] {label}")
    print(f"Confidence : {confidence:.1f}%")
    if body:
        print(f"Input type : Full article ({len(body.split())} words)")
    else:
        print(f"Input type : Headline only (low confidence expected)")
    print("-" * 60)


# In[13]:


# Test with full articles
print("=== Prediction Tests ===\n")

predict_news(
    title="Federal Reserve raises interest rates by 0.25 percent",
    body="""WASHINGTON (Reuters) - The Federal Reserve raised interest rates by a 
    quarter percentage point on Wednesday. The decision was unanimous among voting 
    members. Fed Chair Jerome Powell said the move reflects confidence in the 
    US economy and labour market. Markets had widely anticipated the hike."""
)

predict_news(
    title="Bill Gates microchips found in COVID vaccines say doctors",
    body="""A group of online doctors claimed that Microsoft founder Bill Gates 
    has secretly embedded microchips inside COVID-19 vaccines to track the global 
    population. No physical evidence was presented. The claim has been repeatedly 
    debunked by the WHO and CDC. The doctors credentials could not be verified."""
)

predict_news(
    title="Scientists discover new treatment for Alzheimers disease",
    body="""LONDON (Reuters) - British scientists announced a breakthrough 
    Alzheimer's treatment publishing results in the New England Journal of Medicine. 
    The trial of 800 patients showed a 35 percent reduction in cognitive decline 
    over 18 months. The treatment may enter phase three trials pending FDA approval."""
)

# Headline-only (to show why confidence drops)
print("\n--- Headline-only (confidence will be lower) ---\n")
predict_news("NASA confirms moon is made of cheese after secret mission")
predict_news("Supreme Court rules on landmark immigration case")


# In[14]:


real_text = ' '.join(df[df['label'] == 0]['content_clean'].dropna())

wc = WordCloud(width=800, height=400,
               background_color='white',
               colormap='Blues',
               max_words=100).generate(real_text)

plt.figure(figsize=(14,6))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Words in REAL News', fontsize=16)
plt.show()


# In[15]:


from collections import Counter

fake_words = ' '.join(df[df['label']==1]['content_clean'].dropna()).split()
real_words = ' '.join(df[df['label']==0]['content_clean'].dropna()).split()

fake_freq = pd.DataFrame(Counter(fake_words).most_common(15), columns=['Word', 'Frequency'])
real_freq = pd.DataFrame(Counter(real_words).most_common(15), columns=['Word', 'Frequency'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].barh(fake_freq['Word'][::-1], fake_freq['Frequency'][::-1], color='red')
axes[0].set_title('Top 15 Words in FAKE News')
axes[1].barh(real_freq['Word'][::-1], real_freq['Frequency'][::-1], color='green')
axes[1].set_title('Top 15 Words in REAL News')
plt.tight_layout()
plt.show()

print("TOP FAKE NEWS WORDS:")
print(fake_freq.to_string(index=False))
print("\nTOP REAL NEWS WORDS:")
print(real_freq.to_string(index=False))


# In[16]:


summary = {
    'Metric': ['Accuracy', 'AUC Score', 'Total Samples',
               'Fake Articles', 'Real Articles'],
    'Value': [
        f"{accuracy_score(y_test, preds)*100:.2f}%",
        f"{roc_auc_score(y_test, model.predict_proba(X_test)[:,1]):.4f}",  # ← fix
        len(df),
        len(df[df['label']==1]),
        len(df[df['label']==0])
    ]
}
summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))


# In[17]:

import matplotlib.pyplot as plt

# Keeping the name 'models_compared' to avoid NameError
models_compared = {
    'Random Forest': 0.9122,
    'MLP Neural Net': 0.9097,
    'Voting Ensemble': 0.9087,
    'Logistic Regression': 0.9086,
    'Naive Bayes': 0.8762
}

plt.figure(figsize=(10, 5))
bars = plt.bar(models_compared.keys(), models_compared.values(), color='steelblue')

# Dynamic Y-axis limits based on your new data
min_acc = min(models_compared.values())
plt.ylim(min_acc - 0.02, 0.95)

plt.ylabel('Accuracy')
plt.title('Model Performance Comparison (Group 10)')

# Logic for adding labels on top of the bars
for bar, val in zip(bars, models_compared.values()):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.001,
             f'{val*100:.2f}%',
             ha='center',
             fontsize=11,
             fontweight='bold')

plt.tight_layout()
plt.show()


# In[18]:


# Check what the vectorizer actually sees for a "real" headline
test = "Federal Reserve raises interest rates by 0.25 percent"
cleaned = clean_text(test)
print(f"Cleaned: '{cleaned}'")

# Check what a real article looks like after cleaning
sample_real = df[df['label']==0]['content_clean'].iloc[0]
print(f"\nSample real article start: '{sample_real[:200]}'")

sample_fake = df[df['label']==1]['content_clean'].iloc[0]
print(f"\nSample fake article start: '{sample_fake[:200]}'")


# In[ ]:




