import plotly.express as px
import pandas as pd
import pickle
import numpy as np
from os import listdir
import matplotlib.pyplot as plt

def get_array_paths(root_file, lang="ES"):
    array_paths = [
        f 
        for f in listdir(root_file) 
        if (lang.lower() in f) and ("pickle" in f) and ("dm" in f or "report" in f or "projection" in f)
    ]
    return array_paths

def get_data(path_train, lang="ES"):
    df_train = pickle.load(open(path_train, "rb"))
    sample_index = np.random.RandomState(0).choice(range(df_train.shape[0]), 20000)
    labels = df_train.iloc[sample_index]["label"].values
    return df_train, labels, sample_index

def get_df_results(root_file, array_paths, lang="ES"):
    results = []
    for f in array_paths:
        if "report" in f:
            se, cm = f.replace(f"_{lang.lower()}_report.pickle", "").split("_")
            dict_m = pickle.load(open(root_file+"/"+f, "rb"))
            for k, v in dict_m["metricas"].items():
                o = {"Vector": se, "Clustering": cm, "Metrica": k, "Valor": v}
                results.append(o)
    return pd.DataFrame(results)

def plot_metric(df, metric, lang="ES"):
    fig = px.bar(df[df["Metrica"] == metric], x="Clustering", y="Valor", color="Vector",
                    width=600, height=400)
    fig.update_layout(
        title=f"{metric}: {lang}",
        barmode='group')
    fig.show(renderer="notebook")
    
def get_dm(root_file, array_paths, lang="ES"):
    dm = []
    for f in array_paths:
        if "dm" in f:
            se, cm = f.replace(f"_{lang.lower()}_dm.pickle", "").split("_")
            dict_m = pickle.load(open(root_file+"/"+f, "rb"))
            o = [(se, cm), dict_m]
            dm.append(o)
    return dm

def plot_dm(dm, save=False, lang="ES"):
    fig, ax = plt.subplots(4, 4, figsize=(8, 8))
    q1=0.05
    q3=0.95
    x0=-70
    scale=1.02
    bert = {"ES": "beto", "US": "bertweet"}[lang]
    for k, v in enumerate(dm):
        v0, v1 = v
        j, i = ["tfidf", "bow", "w2v", bert].index(v0[0]), ["kmeans", "gm", "hc", "dbscan"].index(v0[1])
        vmin = np.quantile(v1["matriz_distancias"].flatten(), q1)
        vmax = np.quantile(v1["matriz_distancias"].flatten(), q3)
        ax[i,j].imshow(v1["matriz_distancias"], cmap="jet", vmin=vmin, vmax=vmax)
        ax[i,j].set_yticks([])
        ax[i,j].set_xticks([])
        if i==0:
            ax[i,j].set_title(v0[0], size=18)
        if j==0:
            ax[i,j].set_ylabel(v0[1], size=18)
    plt.suptitle(f"Matriz de distancias: {lang}", size=20);
    if save:
        fig.savefig(f'MD_{lang}.jpg', bbox_inches='tight')
        
def get_prjt(root_file, array_paths, lang="ES"):
    prjt = {"pca": {}, "tsne": {}, "umap": {}}
    for f in array_paths:
        if "projection" in f:
            se, rd = f.split("_")[:2]
            reduced = pickle.load(open(root_file+"/"+f, "rb"))["reduced"]
            prjt[rd][se] = reduced
    return prjt

def get_clusters(root_file, array_paths, lang="ES"):
    clusters = {"kmeans": {}, "hc": {}, "gm": {}, "dbscan": {}}
    for f in array_paths:
        if "report" in f:
            se, cm = f.replace(f"_{lang.lower()}_report.pickle", "").split("_")
            dict_m = pickle.load(open(root_file+"/"+f, "rb"))
            clusters[cm][se] = dict_m["clusters"]
    return clusters

def plot_prj(name_prjt, prjt, clusters, save=False, lang="ES"):
    alpha=0.5
    ms=7
    fig, ax = plt.subplots(4, 4, figsize=(10, 10))
    plot_sample_index = np.random.RandomState(0).choice(range(20000), 1000)
    bert = {"ES": "beto", "US": "bertweet"}[lang]
    for i, cm in enumerate(["kmeans", "gm", "hc", "dbscan"]):
        for j, se in enumerate(["tfidf", "bow", "w2v", bert]):
            clusters_ij = clusters[cm][se][plot_sample_index]
            x = prjt[name_prjt][se][plot_sample_index,0]
            y = prjt[name_prjt][se][plot_sample_index,1]
            sorted_clusters_ij = pd.Series(clusters_ij).value_counts().sort_values(ascending=False).index
            for cluster in sorted_clusters_ij:
                ixs = (clusters_ij == cluster)
                ax[i, j].scatter(
                    x[ixs],
                    y[ixs],
                    alpha=alpha,
                    s=ms
                )
            ax[i,j].set_yticks([])
            ax[i,j].set_xticks([])
            if i==0:
                ax[i,j].set_title(se, size=18)
            if j==0:
                ax[i,j].set_ylabel(cm, size=18)
    plt.suptitle(f"{name_prjt.upper()}: {lang}", size=20);
    if save:
        fig.savefig(f'{name_prjt.upper()}_{lang}.jpg', bbox_inches='tight')