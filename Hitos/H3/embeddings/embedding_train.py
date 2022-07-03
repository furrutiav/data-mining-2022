from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
import pickle
import os
import gc


# folder_emb = "bertweet_base_emoji"
folder_emb = "Hitos/H3/embeddings/beto_emoji"

n_labels = 19  # n_labels = 20

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

task='emoji'
MODEL = f"ccarvajal/beto-{task}"
folder = MODEL.replace('ccarvajal','Hitos/H3/modelos')
# folder = MODEL.replace('cardiffnlp','Hitos/H3/modelos')

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
    return cfl_output['hidden_states'][-1][0].detach().numpy().mean(axis=0).reshape(1,768)


def logits_embedding(clf_output):
    # retorna el vector de scores de clasificacion (antes de la capa softmax)
    return clf_output['logits'][0].detach().numpy().reshape(1,n_labels)


embedding_types = [logits_embedding, sum_embedding, first_tok_embedding]


def guardar(y_dic,idx,conjunto):
    # recibe un diccionario que mapea metodos de embedding con listas de array
    for emb_func_name in y_dic.keys():
        arr = np.concatenate(y_dic[emb_func_name], axis=0)
        np.save(os.path.join(folder_emb,'vec_{}_{}_{}'.format(conjunto,emb_func_name,idx)), arr)


def embeddings_conjunto(df,save_rate,limit=None,name='train',embedding_types=[logits_embedding, sum_embedding, first_tok_embedding]):
    # itera sobre un df dado. genera y guarda los embeddings
    y_clf_obj = {emb_func.__name__:[] for emb_func in embedding_types}
    idx = 0
    length = len(df)
    for i, texto in enumerate(df['text']):
        clf_obj = sentence_clf_output(texto)

        for emb_func in embedding_types:
            arr = emb_func(clf_obj)
            y_clf_obj[emb_func.__name__].append(arr)

        del clf_obj
        gc.collect()

        if ((i+1)%(save_rate)==0 and i!=0) or i==length-1:
            guardar(y_clf_obj,idx,name)
            idx += 1

            del y_clf_obj
            gc.collect()
            y_clf_obj = {emb_func.__name__:[] for emb_func in embedding_types}

            print('archivo guardado: porcentaje = {}%'.format(100*(i+1)/length))
        
        if i==limit:
            break

process = 'train'  # 'test
language = 'es'

if __name__=='__main__':
    # parametros de guardado de embeddings
    save_rate = 2500
    # save_rate = 200
    limit = None
    # limit = 5000  # solo para testeo

    if process=='test':
        path = "Data/test/df_{}_test.pickle".format(language)
        df_test = pickle.load(open(path, "rb"))
        embeddings_conjunto(df_test,save_rate=save_rate,name='test',limit=limit)
    elif process=='train':
        path = "Data/train/df_{}_train.pickle".format(language)
        df_train = pickle.load(open(path, "rb"))
        embeddings_conjunto(df_train,save_rate=save_rate,limit=limit)
