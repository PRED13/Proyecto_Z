import nltk
import string
import email
import os
import pandas as pd
import numpy as np
from html.parser import HTMLParser
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

# Descarga de recursos necesarios
nltk.download('stopwords')

# =============================================================================
# SECCIÓN 1: PROCESAMIENTO DE HTML
# =============================================================================

# Esta clase facilita el preprocesamiento de correos electronicos que poseen codigo HTML 
class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []
        
    def handle_data(self, d):
        self.fed.append(d)
        
    def get_data(self):
        return ''.join(self.fed) 

# Esta funcion se encarga de eliminar los tags HTML que se encuentran en el texto del e-mail
def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

# Ejemplo de eliminacion de los tags HTML de un texto
t = '<tr><td align="left"><a href="../../issues/51/16.html#article">Phrack world News</a><td>'
print("Resultado strip_tags:", strip_tags(t))

# =============================================================================
# SECCIÓN 2: CLASE PARSER Y TOKENIZACIÓN
# =============================================================================

class Parser: 
    def __init__(self):
        self.stemmer = nltk.PorterStemmer()
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.punctuation = list(string.punctuation)
    
    def parse(self, email_path):
        """Parse an email."""
        with open(email_path, errors='ignore') as e:
            msg = email.message_from_file(e)
        return None if not msg else self.get_email_content(msg)
    
    def get_email_content(self, msg):
        """Extract the email content"""
        subject = self.tokenize(msg['Subject']) if msg['Subject'] else []
        body = self.get_email_body(msg.get_payload(),
                                   msg.get_content_type())
        content_type = msg.get_content_type()
        # Returning the content of the email
        return {"subject": subject, 
                "body": body,
                "content_type": content_type}

    def get_email_body(self, payload, content_type):
        """Extract the body of the email"""
        body = []
        if isinstance(payload, str) and content_type == 'text/plain':
            return self.tokenize(payload)
        elif isinstance(payload, str) and content_type == 'text/html':
            return self.tokenize(strip_tags(payload))
        elif isinstance(payload, list): 
            for p in payload:
                body += self.get_email_body(p.get_payload(),
                                            p.get_content_type())
        return body

    def tokenize(self, text):
        """Transform a text string in tokens. perform two main actions,
        clean the punctuation symbols and do steamming of the text."""
        for c in self.punctuation:
            text = text.replace(c, "")
        text = text.replace("\t", " ")
        text = text.replace("\n", " ")
        tokens = list(filter(None, text.split(" ")))      
        # Steamming of the tokens
        return [self.stemmer.stem(w) for w in tokens if w not in self.stopwords]

# =============================================================================
# SECCIÓN 3: LECTURA DE ARCHIVOS E ÍNDICES
# =============================================================================

# Lectura directa del primer correo para visualizar (según tu código)
try:
    inmail = open("/home/pred/Documentos/datasets/datasets/trec07p/data/inmail.1").read()
    print("\nContenido de inmail.1 (resumen):", inmail[:100])
except FileNotFoundError:
    print("\nArchivo inmail.1 no encontrado en la ruta específica.")

p = Parser()
# El parse que haces aquí se guarda para pruebas
p.parse("/home/pred/Documentos/datasets/datasets/trec07p/data/inmail.1")

# Lectura del índice
try:
    index_content = open("/home/pred/Documentos/datasets/datasets/trec07p/full/index").readlines()
except FileNotFoundError:
    index_content = []

DATASET_PATH = "/home/pred/Documentos/datasets/datasets/trec07p/full/index"

def parse_index(path_to_index, n_elements):
    ret_indexes = []
    with open(path_to_index) as f:
        index = f.readlines()
    for i in range(n_elements):
        mail = index[i].split("../")
        label = mail[0].strip()
        path = mail[1][:-1]
        # Se construye la ruta manteniendo la lógica de tu código
        ret_indexes.append({"label": label, "email_path": os.path.join("/home/pred/Documentos/datasets/datasets/trec07p/full/", path)})
    return ret_indexes

indexes = parse_index("/home/pred/Documentos/datasets/datasets/trec07p/full/index", 10)

# LEER EL PRIMER CORREO (Ajuste de ruta según tu comentario en el código)
email_path = indexes[0]["email_path"].replace("/full/index/data", "/data")
try:
    with open(email_path, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
except FileNotFoundError:
    lines = []

# Parsear el primer correo
def parse_email(index):
    p = Parser()
    email_path_corr = index["email_path"].replace("/full/index/data", "/data")
    pmail = p.parse(email_path_corr)
    return pmail, index["label"]

mail, label = parse_email(indexes[0])
print("\nEl correo es:", label)

# =============================================================================
# SECCIÓN 4: VECTORIZACIÓN Y CODIFICACIÓN (EXPERIMENTOS)
# =============================================================================

# Preparacion del email en una cadena de texto 
prep_email = [" ".join(mail['subject']) + " " + " ".join(mail['body'])]

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(prep_email)

print("\nCaracteristica de entradas:", vectorizer.get_feature_names_out())
print("Values:\n", x.toarray())

# Ejemplo con OneHotEncoder
prep_email_oh = [[w] for w in mail['subject'] + mail['body']]
enc = OneHotEncoder(handle_unknown='ignore')
X_oh = enc.fit_transform(prep_email_oh)
print("\nOneHotEncoder features:", enc.get_feature_names_out())

# =============================================================================
# SECCIÓN 5: CREACIÓN DEL DATASET Y ENTRENAMIENTO
# =============================================================================

def create_prep_dataset(index_path, nlements):
    X = []
    y = []
    indexes_list = parse_index(index_path, nlements)
    for i in range(nlements):
        print("\rProcessing email: {0}".format(i+1), end='')
        mail_data, label_data = parse_email(indexes_list[i])
        X.append(" ".join(mail_data['subject']) + " " + " ".join(mail_data['body']))
        y.append(label_data)
    print()
    return X, y

# Leer unicamente un subconjunto de correos 
X_train_raw, y_train = create_prep_dataset("/home/pred/Documentos/datasets/datasets/trec07p/full/index", 10)

# Vectorización del conjunto
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)

# Visualización en DataFrame
df_final = pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names_out())
print("\nDataFrame final (primeras filas):\n", df_final.head())

# Entrenamiento del modelo
clf = LogisticRegression()
clf.fit(X_train, y_train)
print("\nEntrenamiento completado.")