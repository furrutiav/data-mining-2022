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
import umap.umap_ as umap
import plotly.graph_objects as go


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


# plot naive bayes
def proba_matrix(clf,vocab_length):  # X_train_bow.shape[1]
    proba_matrix = np.array([clf.predict_proba(np.eye(1,vocab_length,k))[0] for k in range(vocab_length)])
    return proba_matrix

# to_R2
def umap_reducer(proba_matrix):
    reducer = umap.UMAP(n_neighbors=15)
    to_R2 = reducer.fit_transform(proba_matrix)
    return to_R2

def df_umap_clf(vectorizer,map_emojis,matrix):
    to_R2= umap_reducer(matrix)
    df = pd.DataFrame(to_R2)
    df["token"] = vectorizer.get_feature_names_out()
    df["label"] = map_emojis[np.argmax(matrix, axis=1).astype(int)]
    df["proba"] = np.max(matrix, axis=1)
    df = df.merge(df_us_mapping, on="label", how="left")
    return df

def fig_umap(clf,vectorizer,vocab_length):
    matrix = proba_matrix(clf,vocab_length)
    data = []
    for label in df_us_mapping["label"]:
        df_umap = df_umap_clf(vectorizer,df_us_mapping["label"].values,matrix)
        sub_df = df_umap[df_umap["label"] == label]
        data.append(
            go.Scattergl(
                x = sub_df[0],
                y = sub_df[1],
                mode='markers',
                text=sub_df["token"]+"<br>"+sub_df["emoji"]+"<br>"+sub_df["proba"].apply(lambda x: str(np.round(x, 3))),
                name=sub_df["emoji"].iloc[0],
                marker=dict(
                    size=25*sub_df["proba"],
                    line_width=0.2,
                )
            )
        )
        
    fig = go.Figure(data=data)
    fig.update_layout(
        title="Proyección (UMAP) de vectores de probabilidad de tokens",
        autosize=False,
        width=700,
        height=500,
    )
    return fig
