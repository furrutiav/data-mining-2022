from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
import pickle
import os
import gc


# folder_emb = "bertweet_base_emoji"
folder_emb = "Hitos/H3/embeddings/bertweet_base_emoji"

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

task='emoji'
MODEL = f"cardiffnlp/bertweet-base-{task}"
folder = MODEL.replace('cardiffnlp','Hitos/H3/modelos')

try:
    tokenizer = AutoTokenizer.from_pretrained(folder)
    model = AutoModelForSequenceClassification.from_pretrained(folder)
except ValueError:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)


def sentence_clf_output(text):
    # retorna el SequenceClassifierOutput dado un tweet
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input, return_dict=True, output_hidden_states=True)
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


def embeddings_conjunto(df,save_rate,limit=None,name='train'):
    # itera sobre un df dado. genera y guarda los embeddings
    y_clf_obj = []
    idx = 0
    length = len(df)
    for i, texto in enumerate(df['text']):
        clf_obj = sentence_clf_output(texto)
        y_clf_obj.append(clf_obj)
        if i%(save_rate)==0 and i!=0:
            guardar(y_clf_obj,idx,name)
            idx += 1
            del y_clf_obj
            gc.collect()
            y_clf_obj = []
            print('archivo guardado: porcentaje = {}%'.format(100*(i)/length))
        if i==limit:
            break


if __name__=='__main__':

    # path = "Data/train/df_us_train.pickle"
    # df_us_train = pickle.load(open(path, "rb"))
    path = "Data/test/df_us_test.pickle"
    df_us_test = pickle.load(open(path, "rb"))

    # parametros de guardado de embeddings
    save_rate = 1000
    # save_rate = 200
    # limit = 300  # solo para testeo

    # embeddings_conjunto(df_us_train,save_rate=save_rate,limit=limit)
    embeddings_conjunto(df_us_test,save_rate=save_rate,name='test')
