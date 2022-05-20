# Pregunta 2

## Para cada idioma, ¿somos capaces de ajustar un modelo predictivo que reciba un tweet y prediga su emoji asociado?

## Intro
Las herramientas de procesamiento de texto natural han mostrado capacidades muy parecidas a las humanas. Testear su potencial en el contexto de este dataset es interesante puesto a que la variable a predecir es inherentemente subjetiva. En general, se espera que el emoji esté asociado al carácter emocional del tweet en cuestión, por ende tiene sentido testear modelos que han sido entrenados o ajustados para detectar sentimientos. No obstante, en este desafío hay emojis que presentan similar valor emocional. Además puede ser que el emoji corresponda a variables de mayor complejidad, como el sarcasmo del mensaje. Es por esto que el éxito en la predicción sería tarea difícil incluso para un humano.

Para responder a esta pregunta podemos usar modelos como Naïve Bayes, en el cual tomamos en consideración la ocurrencia de cada palabra en tweets de cada emoji, información que luego se usa para generar una probabilidad de emoji dado el tweet. También podría ser interesante usar modelos que tomen en consideración la interacción entre palabras. Un ejemplo de esto son los modelos de lenguaje. Podemos usar modelos de lenguajes pre-entrenados basados en redes neuronales, como es el caso de BERT/BETO, y ajustarlos para la predicción de emojis.


## Propuesta metodológica

Para responder a esta pregunta queremos usar distintos métodos de clasificación. Puede que algunos tengan más éxito que otros y es de nuestro interés analizar por qué, de ser el caso.

Como clasificadores hay muchos, usaremos los los de la lista siguiente:
- Naïve Bayes
- Clasificadores basados en Transformers

Antes que preferir una lista extensa de métodos, queremos analizar adecuadamente cada uno de ellos. Además, como este desafío se enmarcó en la competencia SEMEVAL, contamos con una extensa lista de competidores que incluye sus métricas globales de clasificación. Podemos, en la mayoría de los casos, averiguar qué método usaron. De este modo tendremos un análisis global del uso de diferentes métodos para clasificación multimodal de texto.

**Por qué usar Naïve Bayes?**

Creemos que este método, pese a su simpleza, puede dar resultados interesantes en esta tarea. Como ejemplo consideremos el siguiente tweet:

_Nearly halfway to **Christmas** 🎄 
Let me know, what's your favorite thing about **winter**?
Share this post with someone who celebrates **Christmas** all year round! 🎄_

Este tweet está relacionado con navidad, lo cual es evidente gracias a la presencia de ciertas palabras como: _christmas_ y _winter_. Como consecuencia, está muy propenso a que la clase en cuestión sea aquella del emoji _christmas_tree_, lo cual es efectivmente el caso. Si bien esto es menos claro para otros emoji, podemos generalizar esta idea y asumir que la clase del tweet estará dada por las palabras que lo componen. A su vez, cada palabra tendrá una probabilidad de pertenecer a las clases en cuestión.

El procedimiento para el clasificador será el siguiente:

``` Dado un parámetro alfa y un vocabulario, ajustamos un clasificador Naïve Bayes en base al conjunto de entrenamiento. Luego testeados la calidad de su evaluación con diferentes métricas usando el conjuntos de prueba (test).
```

Una razón para el uso de este método es que los resultados de Naïve Bayes son altamente interpretables puesto a que a cada palabra se le asigna la probabilidad de pertenecer a las distintas clases. Esto nos da un eje extra de exploración que usaremos del siguiente modo, dado un clasificador entrenado : 

```Para cada emoji, seleccionar k palabras con probabilidad más alta de ser catalogadas con el emoji.```

Otra manera de interpretar los resultados del clasificador es el siguiente: para cada palabra poseemos la probabilidad de pertenecer a alguna clase (emoji). Como tenemos 20 emojis (19 respectivamente), entonces esto nos dota de un vector 20-dimensional (resp. 19-d) a valores en [0,1]. Esto nos permite usar alguna técnica de reducción de dimensionalidad para visualizar el espacio que se genera con tales representaciones. Para esta tarea elegimos umap, pues algunos de los miembros del grupo ya tienen vasta experiencia usándola y afinando sus hiperparámetros. En resumen realizamos lo siguiente:

```Para cada palabra del vocabulario, tomar su vector n-dimensional dado por la probabilidad de pertenecer a las n clases. Realizar una reducción de dimensión usando UMAP y proyectalos en el plano junto con el color de la clase más probable y con tamaño del punto dependiendo de cuan fuerte es la probabilidad de pertenecer a su clase más probable. Realizar un análisis cualitativo.```

Para la implementación del clasificador usaremos la librería scikitlearn. Un parámetro a ajustar es el alpha. Este corresponde a la suavización de la verosimilitud, que está dada por la ecuación siguiente:

$$ \theta_{y^i} = \frac{N_{y^i}+\alpha}{N_y+\alpha n} $$

Donde $N_{y^i}$ es el número de veces que la palabra $i$ aparece en la clase $y$, y $N_y$ es el conteo total de palabras para la clase $y$. El valor $\theta_{y^i}$ corresponde a la probabilidad de que una palabra $i$ aparezca en la clase (emoji) $y$.

Por otro lado, la definición del vocabulario es importante a la hora de usar NB. Usamos también la librería scikitlearn para esto. Esta posee un parámetro min_df, que corresponde a la mínima cantidad de ocurrencia que debe tener una palabra para que está sea tomada en cuenta en el clasificador. De este modo, un min_df = 1 tomara todas las palabras. Usar un min_df más elevado nos permitirá analizar solo aquellas palabras que suceden seguido y por ende que portarán más información acerca de la pertenencia o no a una cierta clase.

Realizaremos un grid search para ajustar ambos parámetros en nuestro clasificador. Exploraremos los valores siguientes:
- $\alpha \in \{0.0,0.2,0.4,0.6,0.8,1.0\}$
- $min_df \in \{1,2,3,4,5,6,7,8,9,10\}$
Y escogeremos el ganador en base a la métrica macro f1 para ser consistentes con el resultado de la competición SEMEVAL.

```
Para distintos valores de alpha y min_df, entrenar un clasificador NB y escoger aquel con mayor macro f1
```
