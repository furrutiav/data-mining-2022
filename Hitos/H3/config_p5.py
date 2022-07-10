import pandas as pd
import numpy as np
import pickle
from string import punctuation
 
import matplotlib.pyplot as plt
import seaborn as sns
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
import plotly.express as px

from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import PorterStemmer
import re 

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TweetTokenizer # tokenizer especial para tweets
tt = TweetTokenizer()


def create_df_español():

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

    df_es_train = pickle.load(open(file_names["df_es_train"], "rb"))
    df_es_trial = pickle.load(open(file_names["df_es_trial"], "rb"))
    df_es_test = pickle.load(open(file_names["df_es_test"], "rb"))

    df_us_train = pickle.load(open(file_names["df_us_train"], "rb"))
    df_us_trial = pickle.load(open(file_names["df_us_trial"], "rb"))
    df_us_test = pickle.load(open(file_names["df_us_test"], "rb"))

    df_es_mapping= pickle.load(open(file_names["df_es_mapping"], "rb"))
    df_us_mapping= pickle.load(open(file_names["df_us_mapping"], "rb"))


    df_es_train['tokenized_text'] = df_es_train['text'].str.lower().apply(tt.tokenize) 

    nltk.download('stopwords')

    stopwords_es = stopwords.words('spanish')

    stopwords_es_withpunct = set(stopwords_es).union(set(punctuation))


    df_es_train['tokenized_text'] = df_es_train['tokenized_text'].apply(lambda x: [word for word in x if word not in (stopwords_es_withpunct)])
    df_es_train["length tokenized"] = df_es_train['tokenized_text'].apply(len)


    # puede ser porter o wnl
    porter = PorterStemmer()
    wnl = WordNetLemmatizer()
    df_es_train['StemmedTokenized_text'] = df_es_train['tokenized_text'].apply(lambda x: [porter.stem(word) for word in x])


    # Obtener el contenido de los hashtags de cada tweet del df ya filtrado
    df_es_only_hashtags = pd.DataFrame()
    df_es_only_hashtags['text']=df_es_train['text'].apply(lambda x: re.findall(r"#(\w+)", x))
    df_es_only_hashtags[df_es_only_hashtags['text'].apply(lambda d: len(d)) > 0]

    df_es_train_filtered = df_es_train[df_es_only_hashtags['text'].apply(lambda d: len(d)) > 0]
    df_es_train_filtered['hashtags'] = df_es_only_hashtags['text']  
    df_es_train_filtered['StemmedTokenized_text_wo_hashtag'] = df_es_train_filtered['StemmedTokenized_text'].apply(lambda li: [word for word in li if not word.startswith('#')])


    df_es_train_filtered['StemmedTokenized_text_as_str'] = df_es_train_filtered['StemmedTokenized_text'].astype(object).apply(lambda x: ' '.join([str(i) for i in x]))
    df_es_train_filtered['StemmedTokenized_text_wo_hashtag_as_str'] = df_es_train_filtered['StemmedTokenized_text_wo_hashtag'].astype(object).apply(lambda x: ' '.join([str(i) for i in x]))
    df_es_train_filtered['hashtags_as_str'] = df_es_train_filtered['hashtags'].astype(object).apply(lambda x: ' '.join([str(i) for i in x]))
    
    return df_es_train_filtered





def create_df_english(): 

    from nltk.corpus import stopwords
    from string import punctuation
    from nltk.stem import PorterStemmer

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

    df_es_train = pickle.load(open(file_names["df_es_train"], "rb"))
    df_es_trial = pickle.load(open(file_names["df_es_trial"], "rb"))
    df_es_test = pickle.load(open(file_names["df_es_test"], "rb"))

    df_us_train = pickle.load(open(file_names["df_us_train"], "rb"))
    df_us_trial = pickle.load(open(file_names["df_us_trial"], "rb"))
    df_us_test = pickle.load(open(file_names["df_us_test"], "rb"))

    df_es_mapping= pickle.load(open(file_names["df_es_mapping"], "rb"))
    df_us_mapping= pickle.load(open(file_names["df_us_mapping"], "rb"))

    df_us_train['tokenized_text'] = df_us_train['text'].str.lower().apply(tt.tokenize) 

    nltk.download('stopwords')

    stopwords_us = stopwords.words('english')
    stopwords_us_withpunct = set(stopwords_us).union(set(punctuation))

    df_us_train['tokenized_text'] = df_us_train['tokenized_text'].apply(lambda x: [word for word in x if word not in (stopwords_us_withpunct)])
    df_us_train["length tokenized"] = df_us_train['tokenized_text'].apply(len)


    # puede ser porter o wnl
    porter = PorterStemmer()
    wnl = WordNetLemmatizer()

    df_us_train['StemmedTokenized_text'] = df_us_train['tokenized_text'].apply(lambda x: [porter.stem(word) for word in x])


    df_us_only_hashtags = pd.DataFrame()
    df_us_only_hashtags['text']=df_us_train['text'].apply(lambda x: re.findall(r"#(\w+)", x))
    df_us_only_hashtags[df_us_only_hashtags['text'].apply(lambda d: len(d)) > 0]

    df_us_train_filtered = df_us_train[df_us_only_hashtags['text'].apply(lambda d: len(d)) > 0]
    df_us_train_filtered['hashtags'] = df_us_only_hashtags['text']

    df_us_train_filtered['StemmedTokenized_text_wo_hashtag'] = df_us_train_filtered['StemmedTokenized_text'].apply(lambda li: [word for word in li if not word.startswith('#')])

    df_us_train_filtered['StemmedTokenized_text_as_str'] = df_us_train_filtered['StemmedTokenized_text'].astype(object).apply(lambda x: ' '.join([str(i) for i in x]))
    df_us_train_filtered['StemmedTokenized_text_wo_hashtag_as_str'] = df_us_train_filtered['StemmedTokenized_text_wo_hashtag'].astype(object).apply(lambda x: ' '.join([str(i) for i in x]))
    df_us_train_filtered['hashtags_as_str'] = df_us_train_filtered['hashtags'].astype(object).apply(lambda x: ' '.join([str(i) for i in x]))


    return df_us_train_filtered