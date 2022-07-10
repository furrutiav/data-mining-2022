import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
import pickle
from scipy.special import softmax
import numpy as np
import urllib.request
import csv


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

###############
# Cargamos las librerias necesarias

import pandas as pd
import numpy as np
import pickle

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
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

from nltk.tokenize import TweetTokenizer
tt = TweetTokenizer() 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support


###############
# Llamamos a los clasificadores

from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report

reg = LinearRegression()
##############################

##############################
# Cargamos dataset


file_names = {
    "df_es_mapping": "../../Data/mapping/df_es_mapping.pickle",
    "df_us_mapping": "../../Data/mapping/df_us_mapping.pickle",
    
    "df_es_test": "../../Data/test/df_es_test.pickle",
    "df_us_test": "../../Data/test/df_us_test.pickle",
    
    "df_es_train": "../../Data/train/df_es_train.pickle",
    "df_us_train": "../../Data/train/df_us_train.pickle",
    
    "df_es_trial": "../../Data/trial/df_es_trial.pickle",
    "df_us_trial": "../../Data/trial/df_us_trial.pickle",
}

#Ingles
df_us_train = pickle.load(open(file_names["df_us_train"], "rb"))
df_us_trial = pickle.load(open(file_names["df_us_trial"], "rb"))
df_us_test = pickle.load(open(file_names["df_us_test"], "rb"))
#Español
df_es_train = pickle.load(open(file_names["df_es_train"], "rb"))
df_es_trial = pickle.load(open(file_names["df_es_trial"], "rb"))
df_es_test = pickle.load(open(file_names["df_es_test"], "rb"))

##############################
# Transformamos el atributo multiclase categorico a variables binarias
#Ingles
df_us_train2 = pd.get_dummies(df_us_train , columns = ["label"])
df_us_trial2 = pd.get_dummies(df_us_trial , columns = ["label"])
df_us_test2 = pd.get_dummies(df_us_test , columns = ["label"])
#Español
df_es_train2 = pd.get_dummies(df_es_train , columns = ["label"])
df_es_trial2 = pd.get_dummies(df_es_trial , columns = ["label"])
df_es_test2 = pd.get_dummies(df_es_test , columns = ["label"])

##############################
# Tokenizacion
#Ingles
df_us_train2['tokenized_text'] = df_us_train2['text'].str.lower().apply(lambda x: " ".join(tt.tokenize(x)))
df_us_test2['tokenized_text'] = df_us_test2['text'].str.lower().apply(lambda x: " ".join(tt.tokenize(x)))  
#Español
df_es_train2['tokenized_text'] = df_es_train2['text'].str.lower().apply(lambda x: " ".join(tt.tokenize(x)))
df_es_test2['tokenized_text'] = df_es_test2['text'].str.lower().apply(lambda x: " ".join(tt.tokenize(x)))  

##############################

df_us_mapping = pickle.load(open(file_names["df_us_mapping"], "rb"))#.sort_values("label")
df_es_mapping = pickle.load(open(file_names["df_es_mapping"], "rb"))#.sort_values("label")


# Vectorización

# minima cantidad de veces (10) que tiene que aparecer una palabra para ser considerado optimo en ingles
#Ingles
vectorizer = CountVectorizer(min_df=10) 
X_train_bow = vectorizer.fit_transform(df_us_train2["tokenized_text"])
X_test_bow = vectorizer.transform(df_us_test2["tokenized_text"])

# minima cantidad de veces (5) que tiene que aparecer una palabra para ser considerado optimo en español
#Español
vectorizer_es = CountVectorizer(min_df=5)   
X_train_bow_es = vectorizer_es.fit_transform(df_es_train2["tokenized_text"])
X_test_bow_es = vectorizer_es.transform(df_es_test2["tokenized_text"])

#




################################
# Funciones para la pregunta 3

################################################################################################

# y pred no es binario, hay que darle un cuttoff donde aceptemos que el algoritmo acerto, eligimos el 50 % 
# de esta forma podemos usar metricas tipicas
# c = corte

# Entonces si el algoritmo predice con un número entre 0 y 1 aprox, 
def cutoff(lista, c):
    i = 0
    listanueva = []
    for i in lista:
        if i > c:
            listanueva.append(1)
        else:
            listanueva.append(0)
            
    return listanueva

################################################################################################
# Función que nos entrega el precision recall f1score de cada label por separado us


def metrica_clase1(y_real,y_pred,emoji):
    Met_regresion_dict = classification_report(y_real, y_pred , output_dict = True )
    return print("|",df_us_mapping["emoji"][emoji],"|","Precisión =", Met_regresion_dict.get("1").get("precision"),"|", "Recall =" ,Met_regresion_dict.get("1").get("recall"),"|","f1-score =" ,Met_regresion_dict.get("1").get("f1-score"),"|",df_us_mapping["emoji"][emoji],"|")


################################################################################################
# Función que nos entrega el precision recall f1score de cada label por separado es


def metrica_clase1_es(y_real,y_pred,emoji):
    Met_regresion_dict = classification_report(y_real, y_pred , output_dict = True )
    return print("|",df_es_mapping["emoji"][emoji],"|","Precisión =", Met_regresion_dict.get("1").get("precision"),"|", "Recall =" ,Met_regresion_dict.get("1").get("recall"),"|","f1-score =" ,Met_regresion_dict.get("1").get("f1-score"),"|",df_es_mapping["emoji"][emoji],"|")


################################################################################################
# Función metrica clase 1, pero para todas las clases/labels us

Puntodecorte = 0.5 # punto del cuttoff
def metrica_clase1_todos(train_bow,df_label):
    Puntodecorte = 0.5
    i = 0
    f = df_label.shape[1]
    # el for pasa de i = 0 hasta el 17 maoma
    for i in range(f-3):
        ###
        reg = LinearRegression()
        reg.fit(train_bow, df_label[str("label_")+str(i)])
        y_predcut = cutoff(reg.predict(train_bow), Puntodecorte)
        print (metrica_clase1(df_label[str("label_")+str(i)],y_predcut, i))

################################################################################################
# Función metrica clase 1, pero para todas las clases/labels es
Puntodecorte = 0.5 # punto del cuttoff
def metrica_clase1_todos_es(train_bow,df_label):
    Puntodecorte = 0.5
    i = 0
    f = df_label.shape[1]
    # el for pasa de i = 0 hasta el 17 maoma
    for i in range(f-3):
        ###
        reg = LinearRegression()
        reg.fit(train_bow, df_label[str("label_")+str(i)])
        y_predcut = cutoff(reg.predict(train_bow), Puntodecorte)
        print (metrica_clase1_es(df_label[str("label_")+str(i)],y_predcut, i))


################################################################################################
# Función que seleciona las 5 palabras mas frecuentes según que emoji para us

def topPalabras(vocab_length,proba_matrix,emoji_id,k=5):
    # retorna las palabras para las cuales el emoji en cuestión tiene mas probabilidad
    prob = proba_matrix[:]  # mmm
    ind = np.argpartition(prob,-k)[-k:]
    val = prob[ind]
    palabras = [vectorizer.inverse_transform([np.eye(1,vocab_length,k)[0]])[0][0] for k in ind]
    return palabras, val


################################################################################################
# Función que seleciona las 5 palabras mas frecuentes según que emoji para es
def topPalabras_es(vocab_length,proba_matrix,emoji_id,k=5):
    # retorna las palabras para las cuales el emoji en cuestión tiene mas probabilidad
    prob = proba_matrix[:]  # mmm
    ind = np.argpartition(prob,-k)[-k:]
    val = prob[ind]
    palabras = [vectorizer_es.inverse_transform([np.eye(1,vocab_length,k)[0]])[0][0] for k in ind]
    return palabras, val

###############################
#  Función que entrega ejemplo de coeficientes con el emoji de navidad en ingles

def ejemplo_navidad_us():    
    i = 17
    reg.fit(X_train_bow, df_us_train2["label_17"])
    vocab_length = X_train_bow.shape[1]
    map_emojis = df_us_mapping["label"].values
    proba_matrix = np.array([reg.predict(np.eye(1,vocab_length,k))[0] for k in range(vocab_length)])
    print(df_us_mapping["emoji"][int(map_emojis[i])])
    print(topPalabras(vocab_length , proba_matrix , i))
    

########################################
# Funcion que crea la lista de top de palabras us

def Listatop_us():
    map_emojis = df_us_mapping["label"].values
    vocab_length = X_train_bow.shape[1]
    for i in range(20):
       reg = LinearRegression()
       reg.fit(X_train_bow, df_us_train2[str("label_")+str(i)])
       proba_matrix = np.array([reg.predict(np.eye(1,vocab_length,k))[0] for k in range(vocab_length)])
       print(df_us_mapping["emoji"][int(map_emojis[i])])
       pal, val = topPalabras(vocab_length ,proba_matrix,i)
       print(dict([(pal[j],val[j]) for j in range(len(pal))]))
    
    
    
########################################
# Funcion que crea la lista de top de palabras es

def Listatop_es():
    map_emojis_es = df_es_mapping["label"].values
    vocab_length_es = X_train_bow_es.shape[1]
    for i in range(19):           
         reg = LinearRegression()
         reg.fit(X_train_bow_es, df_es_train2[str("label_")+str(i)])
         proba_matrix_es = np.array([reg.predict(np.eye(1,vocab_length_es,k))[0] for k in    range(vocab_length_es)])
         print(df_es_mapping["emoji"][int(map_emojis_es[i])])
         pal, val = topPalabras_es(vocab_length_es,proba_matrix_es,i)
         print(dict([(pal[j],val[j]) for j in range(len(pal))]))         
    return


