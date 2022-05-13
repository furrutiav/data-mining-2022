# Pregunta 2

## ¿Podemos utilizar representaciones vectoriales apropiadas de los tweets y encontrar modelos descriptivos de agrupamiento de datos como clustering capaces de relacionar más aquellos tweets asociados a un mismo emoji?

## Preámbulo
El paradigma tradicional de representación vectorial de datos es a través de un diseño manual de atributos. 
En Procesamiento del Lenguaje Natural (PLN) existen varias aproximaciones para representar vectorialmente el texto. 
Métodos como Bag-of-word y tf-idf son formas rudimentarias de codificar la información del texto. 
Mientras que el uso de word-embeddings como word2vec y modelos del lenguaje como BERT son formas más sofisticadas de hacer lo mismo. 
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

En lo que sigue se detallara la propuesta metodológica ilustrada en la **Figura 1.**

**Figura 1.** *Diagrama de trabajo para analisis de clustering*

<img src="https://github.com/furrutiav/data-mining-2022/blob/main/Hitos/H2/p2_clustering.png" alt="drawing" width="600"/>

Primero, nuestro conjunto de datos esta separado por idioma (Ingles y Español) y por particion de ajuste (Entrenamiento/Evaluacion/Prueba). Cada dato consiste en un tweet junto a su etiqueta (indice asociado al emoji). Ahora bien, para utilizar algoritmos de clustering es necesario representar vectorialmente cada tweet. Para esto, consideramos cinco metodos. Tres de estos sirve tanto para Ingles como para Español. Estos son: bag-of-word, tf-idf y word2vec. Luego, para Español consideramos BETO (BERT pero entrenado con un corpus en español). Mientras que para Ingles, consideramos BERTweet (BERT pero entrenado con tweets en ingles). Ver **Tabla 1.** para mas detalles de estos metodos. Brevemente, los primeros dos metodos siguen la forma tradicional de representacion de texto. Mientras que los demas, siguen el paradigma de aprendizaje de representaciones. Estas representaciones provienen de capa especificas en redes neuronales. Por otro lado, aquellos inspirados en BERT, siguen el paradigma de aprendizaje profundo.

**Tabla 1.** *Representación del tweet*

|       Notación    |   Método      |  🇺🇸     |   🇪🇸    | Libreria |
|-------------------|---------------|---------|--------|----------|
|       BOW         | [bag-of-words](https://en.wikipedia.org/wiki/Bag-of-words_model)                                                            |  ✔️   |   ✔️   | [sklearn](https://scikit-learn.org/stable/modules/feature_extraction.html#the-bag-of-words-representation)|
|       TFIDF       | [term-frecuency invert-document-frecuency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)                                    |  ✔️    |   ✔️   | [sklearn](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)           |
| W2V               | [Word2vec](https://en.wikipedia.org/wiki/Word2vec)                                                                          | ✔️     | ✔️     | [gensim](https://radimrehurek.com/gensim/models/word2vec.html)                                            |
|       BETO        | [BETO: Spanish BERT](https://github.com/dccuchile/beto)                                                                     |       |   ✔️   | [🤗](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased)                                |
|       BERTweet    | [BERTweet: A pre-trained language model for English Tweets](https://aclanthology.org/2020.emnlp-demos.2/)                   |  ✔️    |       | [🤗](https://huggingface.co/vinai/bertweet-base)                                                  |

Luego de representar cada tweet como un vector con alguno de los metodos propuestos, consideramos cinco algoritmos distintos para agrupar los datos en clusters. Estos son: k-means, Jerarquico aglomerativo, dbscan, optics y Mixturas Gaussianas. Ver **Tabla 2.** para mas detalle. Se indicaran tantos clusters como emojis, siempre que el metodo lo permita. La idea es encontrar el metodo mas apropiado para nuestra tarea principal (predecir el emoji). Es decir, aquel que mejor agrupe aquellos tweets asociados a un mismo emoji y, al mismo tiempo, distinguir aquellos tweets asociado a un emoji diferente. Para medir esto finalizamos con una evaluacion y visualizacion del clustering.

**Tabla 2.** *Algoritmos de clustering*

|       Notación    |   Método      | param: #(clusters) |  Libreria |
|-------------------|---------------|------|---------|
| kMeans           | [k-means](https://en.wikipedia.org/wiki/K-means_clustering)           | ✔️ |[sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans) |
| Agg      | [Agglomerative Hierarchical](https://en.wikipedia.org/wiki/Hierarchical_clustering)      | ✔️ |[sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering) |
| DBSCAN            | [Density-based spatial clustering of applications with noise](https://en.wikipedia.org/wiki/DBSCAN)            | |[sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN) | 
| OPTICS            | [Ordering points to identify the clustering structure](https://en.wikipedia.org/wiki/OPTICS_algorithm)      |  | [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html#sklearn.cluster.OPTICS) |
| GM  | [Gaussian Mixture](https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model)  | ✔️ |[sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture) |

Para evaluar cuantitativamente el clustering, consideramos siete metricas. Como se tienen las etiquetas correctas de los puntos, entonces podemos considerar metricas similares a Accuracy, Preccion, Recall y F1-score, para evaluar los clusters. Estas metricas son: Rand index, Mutual Information, Homogeneity, Completeness y V-measure. Por otro lado, consideramos dos metricas adicionales que no requieren etiquetas correctas: Matriz de correlacion y Silhouette. Ver **Tabla 3.** para mas detalle.

**Tabla 3.** *Métricas de evaluación para clustering. El simbolo 🔒 indica que es necesaria dicha informacion para calcular la metrica*

|       Notación    |   Método               | Rango  | Ground truth (Etiquetas)    | kMeans | Agg | DBSCAN | OPTICS | GM |  Libreria |
|-------------------|------------------------|--------|-----------------------------|--------|-----|--------|--------|----|-----------|
|      Corr         | [Correlación (Pearson)](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)                        | [-1, 1]^n   |       ✔️       |   ✔️    |  ✔️  |   ✔️    |   ✔️    | ✔️  |     [🐼](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html)      |
|      SC           | [Silhouette](https://en.wikipedia.org/wiki/Silhouette_(clustering))                                           | [-1, 1]   |       ✔️       |  ✔️     | ✔️   |    ✔️   |    ✔️   |✔️   |   [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score)        |
|      Rand         | [Rand index](https://en.wikipedia.org/wiki/Rand_index)                                                        |  [0, 1]  |       🔒       |  ✔️     | ✔️   |    ✔️   |    ✔️   |✔️   |  [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.rand_score.html#sklearn.metrics.rand_score)        |
| NMI               | [Mutual Information](https://en.wikipedia.org/wiki/Mutual_information#Normalized_variants)                          | [0, 1]   |       🔒       |  ✔️     | ✔️   |    ✔️   |    ✔️   |✔️   |  [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html#sklearn.metrics.normalized_mutual_info_score)        |
| Hom               | [Homogeneity](https://scikit-learn.org/stable/modules/clustering.html#homogeneity-completeness-and-v-measure) | [0, 1]   |       🔒       |  ✔️     | ✔️   |    ✔️   |    ✔️   |✔️   |  [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_score.html#sklearn.metrics.homogeneity_score)        |
| Comp              | [Completeness](https://scikit-learn.org/stable/modules/clustering.html#homogeneity-completeness-and-v-measure)| [0, 1]   |       🔒       |  ✔️     | ✔️   |    ✔️   |    ✔️   |✔️   |  [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.completeness_score.html#sklearn.metrics.completeness_score)        |
| V                 | [V-measure](https://scikit-learn.org/stable/modules/clustering.html#homogeneity-completeness-and-v-measure)   | [0, 1]   |       🔒       |  ✔️     | ✔️   |    ✔️   |    ✔️   |✔️   |  [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.v_measure_score.html#sklearn.metrics.v_measure_score)        |


Ahora bien, para evaluar cualitativamente el clustering, se visualizaran los clusters obtenidos. Sin embargo, como los vectores viven en espacios de dimensiones mayores que dos, es necesario reducir la dimensionalidad de los vectores. Para esto, consideramos tres metodos distintos. Estos son: PCA, tSNE y UMAP. Ver **Tabla 4.** para mas detalle. La idea, es reducirnos a dimensiones interpretables como 2 y 3-dimensiones.

**Tabla 4.** *Reductores de dimensionalidad*

|       Notación    |   Método               | Libreria |
|-------------------|------------------------|----------|
|      PCA          | [Principal component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis)  |   [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)       |
|      tSNE         | [t-Distributed Stochastic Neighbor Embedding](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)  | [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) |
|      UMAP         | [Uniform Manifold Approximation and Projection](https://arxiv.org/abs/1802.03426) | [📄](https://umap-learn.readthedocs.io/en/latest/) |

Finalmente, este analisis nos dara un entendimento de la dificultad de predecir un emoji. Si los datos no se expresan en grupos diferenciados por el emoji, es decir, que se solapan los grupos de datos para emojis distintos. Entonces, modelos de clasificacion tenderan a obtener un desempeño deficiente. Por otro lado, con esta informacion podremos encontrar otro grupos (disintos a los indicados por el emoji) que pudiesen existir en el conjunto de datos. En cambio, si los datos logran expresarse para cierta representacion del texto junto a un metodo de clustering particular, responderemos afirmativamente la pregunta.
