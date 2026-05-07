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

df.to_csv('../data/kaggle_only.csv', index=False)
print("Kaggle base dataset saved → kaggle_only.csv")
print("NOTE: Sensationalism features here are for EDA only.")
print("NB2 will recompute them for ALL rows (Kaggle + LIAR) after merge.")


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




