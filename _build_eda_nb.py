import json
import uuid
from pathlib import Path

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14.3"},
    },
    "cells": [],
}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def code(src):
    lines = [line + "\n" for line in src.splitlines()]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


cells_src = [
    ("md", "Imports"),
    (
        "code",
        """import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path""",
    ),
    ("md", "Create Filtered CSV For EDA"),
    (
        "code",
        r"""# Notebook-friendly paths (no __file__ in Jupyter)
ROOT = Path.cwd()
DATA_DIR = ROOT / 'data'
TWEETS_PATH = DATA_DIR / 'tweets.txt'
EMOJI_PATH = DATA_DIR / 'emoji.txt'
OUT_CSV = DATA_DIR / 'filtered_tweets.csv'

# Remove all mentions
MENTION_RE = re.compile(r'@\w+')
URL_RE = re.compile(r'https?://\S+|www\.\S+', re.I)
# Remote Retweet syntax
RT_PREFIX_RE = re.compile(r'^\s*rt\s*:\s*', re.I)
SPACE_RE = re.compile(r'\s+')


def clean_tweet(text: str) -> str:
    text = URL_RE.sub('', text)
    text = MENTION_RE.sub('', text)
    text = SPACE_RE.sub(' ', text).strip()
    text = RT_PREFIX_RE.sub('', text)
    return text.strip()


rows = []
with open(TWEETS_PATH, encoding='utf-8', errors='replace') as tf, open(
    EMOJI_PATH, encoding='utf-8', errors='replace'
) as ef:
    for line_no, (tweet_line, emoji_line) in enumerate(zip(tf, ef), start=1):
        raw = tweet_line.rstrip('\n\r')
        emoji_label = emoji_line.strip()
        mention_count = raw.count('@')
        is_retweet = raw.lstrip().lower().startswith('rt @')
        tweet_body = clean_tweet(raw)
        char_count = len(tweet_body)
        has_question_mark = '?' in tweet_body
        has_exclamation_mark = '!' in tweet_body
        has_hash = '#' in tweet_body
        capital_letter_count = sum(1 for c in tweet_body if c.isupper())
        rows.append(
            {
                'LINE_NUM': line_no,
                'TWEET': tweet_body,
                'EMOJI': emoji_label,
                'CHAR_COUNT': char_count,
                'CAPITAL_LETTER_COUNT': capital_letter_count,
                'MENTION_COUNT': mention_count,
                'HAS_QUESTION_MARK': has_question_mark,
                'HAS_EXCLAMATION_MARK': has_exclamation_mark,
                'HAS_HASH': has_hash,
                'IS_RETWEET': is_retweet,
            }
        )

df = pd.DataFrame(rows)

# Remove dupe tweets
df = df[df['TWEET'].duplicated(keep=False)]
df.to_csv(OUT_CSV, index=False)

print(f'Wrote {len(df)} rows to {OUT_CSV}')""",
    ),
    (
        "code",
        """ROOT = Path.cwd()
DATA_DIR = ROOT / 'data'
TRAIN_CSV = DATA_DIR / 'filtered_tweets.csv'

BOOL_COLS = [
    'HAS_QUESTION_MARK',
    'HAS_EXCLAMATION_MARK',
    'HAS_HASH',
    'IS_RETWEET',
]
df = pd.read_csv(TRAIN_CSV)
for col in BOOL_COLS:
    df[col] = df[col].map(
        lambda x: x is True or (isinstance(x, str) and x.lower() == 'true')
    )""",
    ),
    (
        "md",
        "## EDA (`filtered_tweets.csv`)\n\n"
        "**Subset:** The build cell keeps only rows where `TWEET` appears more than once "
        "(`duplicated(keep=False)`). Everything below describes that duplicate-text subset, "
        "not the full raw corpus.",
    ),
    (
        "code",
        """print('shape:', df.shape)
print('\\ndtypes:\\n', df.dtypes)
print('\\nmissing values per column:\\n', df.isna().sum().to_string())
empty_tweet = df['TWEET'].fillna('').astype(str).str.strip().eq('')
n_empty = int(empty_tweet.sum())
print(f'\\nempty or whitespace-only TWEET rows: {n_empty}')
df.info()""",
    ),
    ("md", "### Target variable: `EMOJI`"),
    (
        "code",
        """emoji_counts = df['EMOJI'].value_counts()
n_classes = df['EMOJI'].nunique()
print(f'Distinct EMOJI labels: {n_classes}')
print('\\nTop 15 counts:\\n', emoji_counts.head(15))
print('\\nTop 15 share of rows:\\n', (emoji_counts.head(15) / len(df)).round(4))

for k in (10, 50):
    rare_mask = emoji_counts < k
    n_rare_classes = int(rare_mask.sum())
    rows_in_rare = int(emoji_counts[rare_mask].sum())
    print(f'\\nClasses with fewer than {k} examples: {n_rare_classes} '
          f'(those classes cover {rows_in_rare} rows total)')""",
    ),
    (
        "code",
        """TOP_N = 25
top = emoji_counts.head(TOP_N)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].barh(top.index[::-1], top.values[::-1], color='steelblue')
axes[0].set_title(f'Top {TOP_N} EMOJI (count)')
axes[0].set_xlabel('Count')
axes[1].barh(top.index[::-1], top.values[::-1], color='steelblue')
axes[1].set_xscale('log')
axes[1].set_xlabel('Count (log scale)')
axes[1].set_title(f'Top {TOP_N} EMOJI (log-scaled count)')
plt.tight_layout()
plt.show()""",
    ),
    ("md", "### `CHAR_COUNT` distribution"),
    (
        "code",
        """fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.histplot(df['CHAR_COUNT'], bins=100, kde=False, ax=axes[0], color='steelblue')
axes[0].set_title('CHAR_COUNT (full range)')
axes[0].set_xlabel('Characters')
axes[0].set_ylabel('Tweets')

p99 = float(df['CHAR_COUNT'].quantile(0.99))
subset = df.loc[df['CHAR_COUNT'] <= p99, 'CHAR_COUNT']
sns.histplot(subset, bins=60, kde=False, ax=axes[1], color='steelblue')
axes[1].set_title(f'CHAR_COUNT ≤ 99th percentile ({p99:.0f} chars)')
axes[1].set_xlabel('Characters')
axes[1].set_ylabel('Tweets')

plt.tight_layout()
plt.show()

print(df['CHAR_COUNT'].describe())""",
    ),
    ("md", "### `CHAR_COUNT` by `EMOJI` (top labels)"),
    (
        "code",
        """TOP_K = 15
order = df['EMOJI'].value_counts().head(TOP_K).index.tolist()
sub = df[df['EMOJI'].isin(order)]

plt.figure(figsize=(10, 5))
sns.boxplot(data=sub, x='EMOJI', y='CHAR_COUNT', order=order)
plt.xticks(rotation=45, ha='right')
plt.title(f'CHAR_COUNT by EMOJI (top {TOP_K})')
plt.tight_layout()
plt.show()

char_by_emoji = (
    df.groupby('EMOJI', observed=False)['CHAR_COUNT']
    .agg(mean='mean', median='median', count='count')
    .loc[order]
    .round(2)
)
char_by_emoji""",
    ),
    ("md", "### Features vs `EMOJI` (top labels)"),
    (
        "code",
        """TOP_K = 15
order = df['EMOJI'].value_counts().head(TOP_K).index.tolist()
df_top = df[df['EMOJI'].isin(order)]

bool_features = [
    'HAS_QUESTION_MARK',
    'HAS_EXCLAMATION_MARK',
    'HAS_HASH',
    'IS_RETWEET',
]

for feat in bool_features:
    ct = pd.crosstab(df_top['EMOJI'], df_top[feat], normalize='index') * 100
    plt.figure(figsize=(6, 4))
    sns.heatmap(ct, annot=True, fmt='.1f', cmap='Blues', cbar_kws={'label': '% of row (emoji)'})
    plt.title(f'{feat}: row % within emoji (top {TOP_K})')
    plt.ylabel('EMOJI')
    plt.tight_layout()
    plt.show()

num_summary = (
    df_top.groupby('EMOJI', observed=False)[['MENTION_COUNT', 'CAPITAL_LETTER_COUNT']]
    .mean()
    .loc[order]
    .round(3)
)
print('Mean MENTION_COUNT and CAPITAL_LETTER_COUNT by EMOJI:')
num_summary""",
    ),
    (
        "code",
        """fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col in zip(axes, ['MENTION_COUNT', 'CAPITAL_LETTER_COUNT']):
    sns.boxplot(data=df_top, x='EMOJI', y=col, order=order, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title(col)
plt.tight_layout()
plt.show()""",
    ),
    ("md", "### Data quality: duplicates & label noise"),
    (
        "code",
        """dup_mask = df['TWEET'].duplicated(keep=False)
print(
    f'Rows with TWEET text appearing more than once: {dup_mask.sum()} / {len(df)} '
    '(this CSV is built to keep only such rows)'
)
ambig = df.groupby('TWEET')['EMOJI'].nunique()
conflict_tweets = ambig[ambig > 1]
print(f'Distinct TWEET strings with multiple EMOJI labels: {len(conflict_tweets)}')
if len(conflict_tweets) > 0:
    n_rows = int(df['TWEET'].isin(conflict_tweets.index).sum())
    print(f'Rows involved: {n_rows}')
    from IPython.display import display
    display(df[df['TWEET'].isin(conflict_tweets.index)].sort_values('TWEET').head(30))
else:
    print('No conflicting labels for the same tweet text in this file.')""",
    ),
]

for kind, text in cells_src:
    if kind == "md":
        cell = md(text)
    else:
        cell = code(text)
    cell["id"] = str(uuid.uuid4())[:8]
    nb["cells"].append(cell)

out = Path(__file__).resolve().parent / "eda.ipynb"
out.write_text(json.dumps(nb, indent=2), encoding="utf-8")
print("wrote", out, "cells:", len(nb["cells"]))
