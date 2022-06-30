# Proyecto Minería de Datos
## Grupo 1 - Predicción de emojis en tweets

### El Dataset

El dataset Multilingual Emoji Prediction (Barbieri et al. 2010, test y trial sets descargables con [este link](https://github.com/fvancesco/Semeval2018-Task2-Emoji-Detection/blob/master/dataset/Semeval2018-Task2-EmojiPrediction.zip?raw=true), train set descargable con [este otro link](https://drive.google.com/file/d/11Q6Y4cYKuWd8mys90l_50JYeWQo0nd81/view?usp=sharing)) contiene alrededor de 500k tweets, todos conteniendo un emoji, de un conjunto de 20 comúnmente usados. El desafío de base es predecir el emoji en cuestión desde el texto del tweet. Esta tarea puede ser interpretada como una de análisis de sentimiento multimodal puesto a que el emoji comúnmente denota información no verbal del mensaje o contexto, muchas veces emocional.

Barbieri, F., Camacho-Collados, J., Ronzano, F., Espinosa Anke, L., Ballesteros, M., Basile, V., ... & Saggion, H. (2018). Semeval 2018 task 2: Multilingual emoji prediction. In 12th International Workshop on Semantic Evaluation (SemEval 2018) (pp. 24-33). Association for Computational Linguistics. [http://dx.doi.org/10.18653/v1/S18-1003](http://dx.doi.org/10.18653/v1/S18-1003)

### Instalación

**Versión de python: 3.8.13**

Para clasificador basado en transformers se necesita la librería _pytorch_. Los comandos de instalación dependen de cada computador y se pueden encontrar en [este link](https://pytorch.org/get-started/locally/).

Para el resto de las bibliotecas ejecutar

```pip install -r requirements.txt```


### Organización
**[Hito 1](https://github.com/furrutiav/data-mining-2022/tree/main/Hitos/H1)**

**[Hito 2](https://github.com/furrutiav/data-mining-2022/tree/main/Hitos/H2)**

**[Hito 3](https://github.com/furrutiav/data-mining-2022/tree/main/Hitos/H3)**


### Notebooks
**[Exploración](https://github.com/furrutiav/data-mining-2022/blob/main/Hitos/H1/00%20Exploracion.ipynb)**

**[Clasificador con Naive Bayes](https://github.com/furrutiav/data-mining-2022/blob/main/Hitos/H2/clasificador1_en.ipynb)**

**Clasificador con Transformers**:
- [bertweet-base-En](https://github.com/furrutiav/data-mining-2022/blob/main/Hitos/H3/clasificador2_transformer_en_bertweet-base-emoji.ipynb)
- [twitter-roberta-En](https://github.com/furrutiav/data-mining-2022/blob/main/Hitos/H3/clasificador2_transformer_en_twitter-roberta-base.ipynb)
- **[Visualizaciones](https://www.youtube.com/watch?v=dQw4w9WgXcQ)**

**[Resumen clasificadores](https://github.com/furrutiav/data-mining-2022/blob/main/Hitos/H3/resumen_clasificadores.ipynb)**

### Entregables
**[Hito 1](https://github.com/furrutiav/data-mining-2022/blob/main/Hitos/H1/Informe_Hito_01.html)**

**[Hito 2](https://github.com/furrutiav/data-mining-2022/blob/main/Hitos/H2/Informe_Hito_2.html)**
