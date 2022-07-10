import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
import pickle
from scipy.special import softmax
import numpy as np
import urllib.request
import csv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


sns.set_theme(style="whitegrid")
plt.rc('axes', titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rcParams.update({'font.size': 16})
plt.rcParams['axes.titlesize'] = 16
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams.update({'lines.markeredgewidth': 1})
plt.rcParams.update({'errorbar.capsize': 2})

file_names = {
    "df_es_mapping": "../../Data/mapping/df_es_mapping.pickle",
    "df_us_mapping": "../../Data/mapping/df_us_mapping.pickle",
    
    "df_es_test": "../../Data/test/df_es_test.pickle",
    "df_us_test": "../../Data/test/df_us_test.pickle",
    
    "df_es_train": "../../Data/train/df_es_train.pickle",
    "df_us_train": "../../Data/train/df_us_train.pickle",
    
    "df_es_trial": "../../Data/trial/df_es_trial.pickle",
    "df_us_trial": "../../Data/trial/df_es_trial.pickle",
}

df_us_mapping = pickle.load(open(file_names["df_us_mapping"], "rb")).sort_values("label")
df_es_mapping = pickle.load(open(file_names["df_es_mapping"], "rb")).sort_values("label")


def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def eval_text(text,model,tokenizer):
    # retorna el indice del emoji con mas probabilidad y los scores
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return np.argmax(scores), scores


labels_es = []
mapping_link = f"https://raw.githubusercontent.com/camilocarvajalreyes/beto-emoji/main/es_mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels_es = [row[1] for row in csvreader if len(row) > 1]

labels_en = []
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/emoji/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels_en = [row[1] for row in csvreader if len(row) > 1]

def rank_emojis_text(text,model,tokenizer,labels):
    # imprime los emojis ordenados por probabilidad
    """if idioma in ['en','EN','us','US']:
        labels = labels_en
    elif idioma in ['es','ES']:
        labels = labels_es"""
    _, scores = eval_text(text,model,tokenizer)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    for i in range(scores.shape[0]):
        l = labels[ranking[i]]
        s = scores[ranking[i]]
        print(f"{i+1}) {l} {np.round(float(s), 4)}")
