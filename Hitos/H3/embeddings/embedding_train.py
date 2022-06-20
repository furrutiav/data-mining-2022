from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
import pickle
import os

folder_emb = "bertweet_base_emoji"

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

task='emoji'
MODEL = f"cardiffnlp/bertweet-base-{task}"
folder = MODEL.replace('cardiffnlp','../modelos')

# tokenizer = AutoTokenizer.from_pretrained(folder)
# model = AutoModelForSequenceClassification.from_pretrained(folder)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


def sentence_clf_output(text):
    # retorna el SequenceClassifierOutput dado un tweet
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input, return_dict=True, output_hidden_states=True)
    ###
    """ArithmeticErrorscores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return np.argmax(scores), scores"""
    return output


def first_tok_embedding(cfl_output):
    # retorna un numpy array correspondiente al token <s> contextualizado según el tweet
    return cfl_output['hidden_states'][-1][0][0].detach().numpy().reshape(1,768)


def sum_embedding(cfl_output):
    # retorna un numpy array correspondiente a la suma de los vectores contextualizados
    return cfl_output['hidden_states'][-1][0].detach().numpy().sum(axis=0).reshape(1,768)


def logits_embedding(clf_output):
    # retorna el vector de scores de clasificacion (antes de la capa softmax)
    return clf_output['logits'][0].detach().numpy().reshape(1,20)


embedding_types = [logits_embedding, sum_embedding, first_tok_embedding]


def guardar(y_list,idx,conjunto):
    embedding_types = [logits_embedding, sum_embedding, first_tok_embedding]

    for emb_func in embedding_types:
        arr = np.concatenate([emb_func(clf_obj) for clf_obj in y_list], axis=0)
        np.save(os.path.join(folder_emb,'vec_{}_{}_{}'.format(conjunto,emb_func.__name__,idx)), arr)


if __name__=='main':

    path =  "../../../Data/train/df_us_train.pickle"
    df_us_train = pickle.load(open(path, "rb"))

    y_clf_obj = []
    length = len(df_us_train)
    # save_rate = 5000
    save_rate = 200

    for i, texto in enumerate(df_us_train['text']):
        idx = 0
        clf_obj = sentence_clf_output(texto)
        y_clf_obj.append(clf_obj)
        if i%(save_rate)==0 and i!=0:
            guardar(y_clf_obj,idx,'train')
            idx += 1
            y_clf_obj = []
            print('archivo guardado: porcentaje = {}%'.format(100*(i)/length))
        if i==1000:
            break
