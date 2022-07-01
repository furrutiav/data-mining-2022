from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import time
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import rand_score, homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import normalized_mutual_info_score
import plotly.express as px
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd 
import numpy as np

def clustering_report(X, labels, clustering_method, params_clustering):
    # Clustering
    ## Fit
    print("Fit...")
    start = time.time()
    clustering = clustering_method(**params_clustering)
    clustering.fit(X)
    dt_fit = time.time()-start
    print("Fit (done)")
    
    ## Predict
    print("Predict...")
    clustering_clusters = None
    if clustering_method.__name__ in ["KMeans", "AgglomerativeClustering", "DBSCAN", "OPTICS"]:
        clustering_clusters = clustering.labels_
    elif clustering_method.__name__ in ["GaussianMixture"]:
        clustering_clusters = clustering.predict(X)        
    print("Predict (done)")
    
    # Metricas
    ## Silhouette
    print("Silhouette...")
    silhouette_value = np.nan
    dt_silhoutte = np.nan
    try:
        start = time.time()
        silhouette_value = silhouette_score(X, clustering_clusters)  
        dt_silhoutte = time.time()-start
    except: print("X")
    print("Silhouette (done)")
    
    print("Metricas...")
    ## Rand score
    rand_value = rand_score(labels, clustering_clusters)
    
    ## Mutual information
    nmi_value = normalized_mutual_info_score(labels, clustering_clusters)
    
    ## Homogeneity
    hom_value = homogeneity_score(labels, clustering_clusters)
    
    ## Completeness
    com_value = completeness_score(labels, clustering_clusters)

    ## V-measure
    v_me_value = v_measure_score(labels, clustering_clusters)
    print("Metricas (done)")
    
    dict_metrics = {
        "Silhouette": silhouette_value,
        "Rand score": rand_value,
        "Mutual information": nmi_value,
        "Homogeneity": hom_value,
        "Completeness": com_value,
        "V-measure": v_me_value
    }
    
    dict_dt = {
        "Silhouette": dt_silhoutte,
        "Fit": dt_fit
    }
    return {
        "clusters": clustering_clusters,
        "metricas": dict_metrics,
        "dt": dict_dt
    }

def distance_matrix(X, tags, size=0.1):
    # Indices ordenados
    print("Indices...")
    start = time.time()
    index = []
    size_tags = []
    tags_ord = pd.Series(tags).value_counts().sort_values(ascending=False).index
    for tag in tags_ord:
        index_tag = list(np.where(tags==tag)[0])
        index_tag = np.random.RandomState(0).choice(
            index_tag, 
            int(len(index_tag)*size))
        size_tags.append(len(index_tag))
        index += list(index_tag)
    dt_indices = time.time() - start
    print("Indices (done)")
    
    print("Matriz de distancias...")
    start = time.time()
    matriz_distancias = euclidean_distances(X[index], X[index])
    dt_matriz = time.time() - start
    print("Matriz de distancias (done)")
    
    dic_others = {
        "index": index,
        "tags_ord": tags_ord,
        "size_tags": size_tags
    }
    
    dic_dt = {
        "indices": dt_indices,
        "matriz": dt_matriz
    }
    return {
        "matriz_distancias": matriz_distancias, 
        "others": dic_others,
        "dt": dic_dt
    }

def plot_distance_matrix(dict_dm, q1=0.05, q3=0.69, x0=-70, scale=1.02, num_tags=5):
    fig, ax = plt.subplots(1,1, figsize=(5, 5))
    vmin = np.quantile(dict_dm["matriz_distancias"].flatten(), q1)
    vmax = np.quantile(dict_dm["matriz_distancias"].flatten(), q3)
    ax.imshow(dict_dm["matriz_distancias"], cmap="jet", vmin=vmin, vmax=vmax);
    ax.set_title("Matriz de distancias")
    ax.set_yticks([])
    y_text = [sum(dict_dm["others"]["size_tags"][:i])+dict_dm["others"]["size_tags"][i]/2 for i in range(len(dict_dm["others"]["size_tags"]))]
    for cluster in range(num_tags):
        ax.text((y_text[cluster]+x0)*scale, y_text[cluster]*scale, str(dict_dm["others"]["tags_ord"][cluster]), size=15, color="white")
    ax.set_xticks([])
    plt.show()
    
def projection(X, reducer, params_reducer):
    print("Reducer...")
    start = time.time()
    reduced = reducer(**params_reducer).fit_transform(X)
    print("Reducer (done)")
    df_reducer = time.time()-start
    
    return {
        "reduced": reduced,
        "dt": df_reducer
    }

def get_projection_df(dict_projection, clustering_clusters, labels):
    df = pd.DataFrame(dict_projection["reduced"])
    df["cluster"] = clustering_clusters
    df["cluster"] = df["cluster"].apply(str)
    df["label"] = labels
    return df

def plot_confussion_matrix(labels, clustering_clusters):
    df_comp = pd.DataFrame()
    df_comp["label"] = labels
    df_comp["cluster"] = clustering_clusters
    confussion_matrix = df_comp.groupby("cluster")["label"].value_counts().unstack(fill_value=0)
    fig,ax=plt.subplots(1,1,figsize=(6,5))
    sns.heatmap(confussion_matrix, ax=ax, vmin=0, cmap="RdYlBu_r")
    ax.set_title("Matriz de confusion")
    plt.show()