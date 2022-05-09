# Pregunta 2

## ¿Podemos utilizar representaciones vectoriales apropiadas de los tweets y encontrar modelos descriptivos de agrupamiento de datos como clustering capaces de relacionar más aquellos tweets asociados a un mismo emoji?

## Intro
El paradigma tradicional de representación vectorial de datos es a través de un diseño manual de atributos. 
En Procesamiento del Lenguaje Natural existen varias aproximaciones para representar vectorialmente el texto. 
Métodos como Bag-of-word y tf-idf son formas rudimentarias de codificar la información del texto. 
Mientras que el uso de word-embeddings como word2vec y el modelo del lenguaje como BERT son formas más sofisticadas de hacer lo mismo. 
Pero esta vez, aprovechándose de propiedades más complejas del texto. 
En nuestro caso, nos interesa explorar distintas aproximaciones que puedan vectorizar apropiadamente un tweet. 
Por otro lado, cada uno de los tweets de la base de datos posee un único emoji asociado. 
El emoji es una de las tantas características comunes que relacionan a los tweets. 
Hay características desde simples hasta complejas de detectar. 
Una característica simple que nos permite agrupar los tweet es la cantidad de palabras que poseen. 
Con esto, podemos tener dos grupos: aquellos con (1) varias palabras y (2) pocas palabras. 
Características complejas suelen ser más difíciles de detectar, como lo es el sentimiento de un tweet. 
Así, podemos tener tres grupos: aquellos con sentimiento (1) positivo, (2) neutro y (3) negativo. 
Ahora bien, estamos interesados en explorar distintos algoritmos de clustering con el objetivo de encontrar aquellos mas apropiados.
Es decir, que sean capaces de traducir grupos diferenciados según el único emoji presente en el tweet tanto para inglés como para español. 
Entre los algoritmos a explorar estan: k-means, aglomerativo, gaussian mixtures, dbscan y optics.

## Propuesta metodológica

Representación del tweet:

|       Notación    |   Método      |  EN     |   ES   | Libreria |
|-------------------|---------------|---------|--------|----------|
|       BOW         | [bag-of-words](https://en.wikipedia.org/wiki/Bag-of-words_model)                                                            |  X    |   X   | [sklearn](https://scikit-learn.org/stable/modules/feature_extraction.html#the-bag-of-words-representation)|
|       TFIDF       | [term-frecuency invert-document-frecuency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)                                    |  X    |   X   | [sklearn](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)           |
| W2V               | [Word2vec](https://en.wikipedia.org/wiki/Word2vec)                                                                          | X     | X     | [gensim](https://radimrehurek.com/gensim/models/word2vec.html)                                            |
|       BETO        | [BETO: Spanish BERT](https://github.com/dccuchile/beto)                                                                     |       |   X   | [huggingface](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased)                                |
|       BERTweet    | [BERTweet: A pre-trained language model for English Tweets](https://aclanthology.org/2020.emnlp-demos.2/)                   |  X    |       | [huggingface](https://huggingface.co/vinai/bertweet-base)                                                  |

Algoritmos de clustering:

|       Notación    |   Método      |  Libreria |
|-------------------|---------------|----------|
| kMeans           | [k-means](https://en.wikipedia.org/wiki/K-means_clustering)           | [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans) |
| Agg      | [Agglomerative Hierarchical](https://en.wikipedia.org/wiki/Hierarchical_clustering)      | [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering) |
| DBSCAN            | [Density-based spatial clustering of applications with noise](https://en.wikipedia.org/wiki/DBSCAN)            | [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN) | [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN) |
| OPTICS            | [Ordering points to identify the clustering structure](https://en.wikipedia.org/wiki/OPTICS_algorithm)      | [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html#sklearn.cluster.OPTICS) |
| GM  | [Gaussian Mixture](https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model)  | [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture) |

Funciones de distancia:

|       Notación    |   Método      |  Libreria |
|-------------------|---------------|-----------|




Métricas de evaluación para clustering:

|       Notación    |   Método      | kMeans | Agg | DBSCAN | OPTICS | GM |  Libreria |
|-------------------|---------------|--------|-----|--------|--------|----|-----------|
|      Corr         | Correlación (Pearson)  |   X    |  X  |   X    |   X    | X  |           |
|      WSS          |                        |   X    |     |        |        |    |           |
|      BSS          |  
|                   | Silhouette 
|                   | Purity
|                   | Entropy 



