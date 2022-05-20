# Pregunta 1

## Para cada idioma, ¿somos capaces de ajustar un modelo predictivo que reciba un tweet y prediga su emoji asociado?

## Intro
Las herramientas de procesamiento de texto natural han mostrado capacidades muy parecidas a las humanas. Testear su potencial en el contexto de este dataset es interesante puesto a que la variable a predecir es inherentemente subjetiva. En general, se espera que el emoji esté asociado al carácter emocional del tweet en cuestión, por ende tiene sentido testear modelos que han sido entrenados o ajustados para detectar sentimientos. No obstante, en este desafío hay emojis que presentan similar valor emocional. Además puede ser que el emoji corresponda a variables de mayor complejidad, como el sarcasmo del mensaje. Es por esto que el éxito en la predicción sería tarea difícil incluso para un humano.

Para responder a esta pregunta podemos usar modelos como Naïve Bayes, en el cual tomamos en consideración la ocurrencia de cada palabra en tweets de cada emoji, información que luego se usa para generar una probabilidad de emoji dado el tweet. También podría ser interesante usar modelos que tomen en consideración la interacción entre palabras. Un ejemplo de esto son los modelos de lenguaje. Podemos usar modelos de lenguajes pre-entrenados basados en redes neuronales, como es el caso de BERT/BETO, y ajustarlos para la predicción de emojis.


## Propuesta metodológica

Para responder a esta pregunta queremos usar distintos métodos de clasificación. Puede que algunos tengan más éxito que otros y es de nuestro interés analizar por qué, de ser el caso.

Como clasificadores hay muchos, usaremos los los de la lista siguiente:
- Naïve Bayes [(1)](#ref1)
- Clasificadores basados en Transformers

Antes que preferir una lista extensa de métodos, queremos analizar adecuadamente cada uno de ellos. Además, como este desafío se enmarcó en la competencia SEMEVAL, contamos con una extensa lista de competidores que incluye sus métricas globales de clasificación. Podemos, en la mayoría de los casos, averiguar qué método usaron. De este modo tendremos un análisis global del uso de diferentes métodos para clasificación multimodal de texto.

**Por qué y como usar Naïve Bayes?**

Creemos que este método, pese a su simpleza, puede dar resultados interesantes en esta tarea. Como ejemplo consideremos el siguiente tweet:

_Nearly halfway to **Christmas** 🎄 
Let me know, what's your favorite thing about **winter**?
Share this post with someone who celebrates **Christmas** all year round! 🎄_

Este tweet está relacionado con navidad, lo cual es evidente gracias a la presencia de ciertas palabras como: _christmas_ y _winter_. Como consecuencia, está muy propenso a que la clase en cuestión sea aquella del emoji _christmas_tree_, lo cual es efectivmente el caso. Si bien esto es menos claro para otros emoji, podemos generalizar esta idea y asumir que la clase del tweet estará dada por las palabras que lo componen. A su vez, cada palabra tendrá una probabilidad de pertenecer a las clases en cuestión.

El procedimiento para el clasificador será el siguiente:

``` 
Dado un parámetro alfa y un vocabulario, ajustamos un clasificador Naïve Bayes en base al conjunto de entrenamiento. Luego testeados la calidad de su evaluación con diferentes métricas usando el conjuntos de prueba (test).
```

Una razón para el uso de este método es que los resultados de Naïve Bayes son altamente interpretables puesto a que a cada palabra se le asigna la probabilidad de pertenecer a las distintas clases. Esto nos da un eje extra de exploración que usaremos del siguiente modo, dado un clasificador entrenado : 

```
Para cada emoji, seleccionar k palabras con probabilidad más alta de ser catalogadas con el emoji.
```

Otra manera de interpretar los resultados del clasificador es el siguiente: para cada palabra poseemos la probabilidad de pertenecer a alguna clase (emoji). Como tenemos 20 emojis (19 respectivamente), entonces esto nos dota de un vector 20-dimensional (resp. 19-d) a valores en [0,1]. Esto nos permite usar alguna técnica de reducción de dimensionalidad para visualizar el espacio que se genera con tales representaciones. Para esta tarea elegimos umap, pues algunos de los miembros del grupo ya tienen vasta experiencia usándola y afinando sus hiperparámetros. En resumen realizamos lo siguiente:

```
Para cada palabra del vocabulario, tomar su vector n-dimensional dado por la probabilidad de pertenecer a las n clases. Realizar una reducción de dimensión usando UMAP y proyectalos en el plano junto con el color de la clase más probable y con tamaño del punto dependiendo de cuan fuerte es la probabilidad de pertenecer a su clase más probable. Realizar un análisis cualitativo.
```

Para la implementación del clasificador usaremos la librería scikitlearn. Un parámetro a ajustar es el alpha. Este corresponde a la suavización de la verosimilitud, que está dada por la ecuación siguiente:

$$ \theta_{y^i} = \frac{N_{y^i}+\alpha}{N_y+\alpha n} $$

Donde $N_{y^i}$ es el número de veces que la palabra $i$ aparece en la clase $y$, y $N_y$ es el conteo total de palabras para la clase $y$. El valor $\theta_{y^i}$ corresponde a la probabilidad de que una palabra $i$ aparezca en la clase (emoji) $y$.

Por otro lado, la definición del vocabulario es importante a la hora de usar NB. Usamos también la librería scikitlearn para esto. Esta posee un parámetro min_df, que corresponde a la mínima cantidad de ocurrencia que debe tener una palabra para que está sea tomada en cuenta en el clasificador. De este modo, un min_df = 1 tomara todas las palabras. Usar un min_df más elevado nos permitirá analizar solo aquellas palabras que suceden seguido y por ende que portarán más información acerca de la pertenencia o no a una cierta clase.

Realizaremos un grid search para ajustar ambos parámetros en nuestro clasificador. Exploraremos los valores siguientes:

$$\alpha \in \[0.0,0.2,0.4,0.6,0.8,1.0\]$$

$$min\textunderscore df \in \[ 1,2,3,4,5,6,7,8,9,10 \] $$

Y escogeremos el ganador en base a la métrica macro f1 para ser consistentes con el resultado de la competición SEMEVAL.

```
Para distintos valores de alpha y min_df, entrenar un clasificador NB y escoger aquel con mayor macro f1
```

**Por qué y como usar Transformers?**

Transformers [2] es una arquitectura de redes neuronales que ha mostrado una enorme capacidad de modelar texto. Su uso se ha masificado y simplificado gracias a la existencia de bibliotecas como _transformers_, que ponen a libre disposición modelos pre-entrenados. Este punto es importante puesto a que entrenar un modelo de lenguaje robusto desde cero puede tomar tiempo y capacidad de cómputo que van más allá de nuestras capacidades.

Creemos que explorar su uso y compararlo con un clasificador como Naïve Bayes será una buena manera de complementar los conocimientos adquiridos en el curso con recursos más avanzados de interés personal. Sin embargo nos centraremos en modelos pre-entrenados sin ajustar. En particular analizaremos los siguientes:
- [BERTweet base (vinai)](https://huggingface.co/docs/transformers/model_doc/bertweet) [(3)](#ref3)
- [BERTweet ajustado para predicción de emojis (CardiffNLP)](https://huggingface.co/cardiffnlp/bertweet-base-emoji)
- [Twitter-roberta base (CardiffNLP)](https://huggingface.co/cardiffnlp/twitter-roberta-base) [(4)](#ref4)
- [Twitter-roberta ajustado para predicción de emojis (CardiffNLP)](https://huggingface.co/cardiffnlp/twitter-roberta-base-emoji)
- [BETO base (DCC-UChile)](https://huggingface.co/dccuchile/bert-base-spanish-wwm-uncased) [(5)](#ref5)

Esta eleccion nos permitirá comparar el efecto de hacer un ajuste a la tarea de predicción de emojis para dos modelos de transformers distintos en inglés. Por otro lado, testearemos el uso de un modelo de lenguaje sin ajustar para los tweets en español. Se hará un análisis de las predicciones consistente con el que usamos en Naïve Bayes, haciendo uso de las métricas de clasificación que provee la biblioteca _scikitlearn_.

Por otro lado, se buscarán modos de interpretación al análizar algunas capas de atención de los modelos. Para esto usaremos la herramienta open source [_bertviz_](https://github.com/jessevig/bertviz) [(6)](#ref6) que nos provee de una herramienta de visualización interactiva.

Finalmente, cabe señalar que hay una conexión directa con la pregunta 2, que pretende analizar clusters usando distintas representaciones de los tweets. En efecto, los modelos de transformers anteriormente mencionados nos otorgan un vector por tweet que puede será usado con este efecto. De este modo estaremos explorando también el clasificador en cuestión a través del espacio semántico que define.

**Referencias**

<a name="ref1"></a>[1] Metsis, V., Androutsopoulos, I., & Paliouras, G. (2006, July). Spam filtering with naive bayes-which naive bayes?. In CEAS (Vol. 17, pp. 28-69).

<a name="ref2"></a>[2] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is All you Need. Advances in Neural Information Processing Systems, 30, 5998–6008. https://arxiv.org/abs/1706.03762

<a name="ref3"></a>[3] Nguyen, D. Q., Vu, T., & Nguyen, A. T. (2020, October). BERTweet: A pre-trained language model for English Tweets. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations (pp. 9-14). https://aclanthology.org/2020.emnlp-demos.2.pdf

<a name="ref4"></a>[4] Barbieri, F., Camacho-Collados, J., Anke, L. E., & Neves, L. (2020, November). TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification. In Findings of the Association for Computational Linguistics: EMNLP 2020 (pp. 1644-1650). https://arxiv.org/pdf/2010.12421.pdf

<a name="ref5"></a>[5] Canete, J., Chaperon, G., Fuentes, R., Ho, J. H., Kang, H., & Pérez, J. (2020). Spanish pre-trained bert model and evaluation data. Pml4dc at iclr, 2020, 2020. https://users.dcc.uchile.cl/~jperez/papers/pml4dc2020.pdf

<a name="ref6"></a>[6] Vig, J. (2019, July). A Multiscale Visualization of Attention in the Transformer Model. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics: System Demonstrations (pp. 37-42). https://aclanthology.org/P19-3007.pdf
