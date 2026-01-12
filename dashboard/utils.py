import pandas as pd
import arff
import os
import numpy as np
import io
import urllib, base64
from pathlib import Path
from pandas.plotting import scatter_matrix 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, RobustScaler, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# CONFIGURACIÓN CRÍTICA PARA RENDER
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# --- CLASE AUXILIAR PARA ARCHIVO 09 ---
class DeleteNanRows(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.dropna()

# --- CARGA DE DATOS OPTIMIZADA ---
def load_kdd_data():
    BASE_DIR = Path(__file__).resolve().parent.parent
    path = BASE_DIR / "KDDTrain+.arff"
    
    if not path.exists():
        return None

    try:
        with open(path, 'r') as f:
            dataset = arff.load(f)
            
        attributes = [attr[0] for attr in dataset['attributes']]
        # Muestreo de 1000 filas para no saturar los 512MB de RAM
        data_slice = dataset['data'][:1000]
        
        df = pd.DataFrame(data_slice, columns=attributes)
        
        # Reducción de precisión para ahorrar memoria
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
            
        return df
    except Exception as e:
        print(f"Error cargando ARFF: {e}")
        return None

def get_base64_graph():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    uri = urllib.parse.quote(base64.b64encode(buf.read()))
    plt.close('all') # LIBERA TODA LA MEMORIA DE LA GRÁFICA
    return uri

# --- ARCHIVO 06: VISUALIZACIONES ---
def generate_visualizations_archivo_06():
    df = load_kdd_data()
    if df is None: return []
    imgs = []
    
    # 1. Histogramas
    df.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1, figsize=(10, 8))
    imgs.append(get_base64_graph())
    
    # 2. Matriz de Correlación
    df_c = df.copy()
    df_c['class'] = LabelEncoder().fit_transform(df_c['class'])
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(df_c.corr(numeric_only=True), vmin=-1, vmax=1, interpolation='none')
    fig.colorbar(cax)
    imgs.append(get_base64_graph())
    
    # 3. Scatter Matrix
    cols = ['same_srv_rate', 'dst_host_srv_count', 'class', 'dst_host_same_srv_rate']
    scatter_matrix(df_c[cols], figsize=(10, 8))
    imgs.append(get_base64_graph())

    # 4. Bar Chart Protocol
    plt.figure(figsize=(7, 5))
    df['protocol_type'].value_counts().plot(kind='bar', grid=True)
    imgs.append(get_base64_graph())
    
    return imgs

# --- ARCHIVO 07: BARRAS DE PROTOCOLOS ---
def generate_visualizations_archivo_07():
    df = load_kdd_data()
    if df is None: return []
    imgs = []
    counts = df['protocol_type'].value_counts()
    
    # Generamos 4 variantes del mismo gráfico de barras
    for i in range(4):
        plt.figure(figsize=(7, 5)) # Crear figura nueva en cada iteración
        if i == 1: 
            counts.sort_index(ascending=False).plot(kind='bar', grid=True, color='orange')
        elif i == 2: 
            counts.sort_values(ascending=True).plot(kind='bar', grid=True, color='green')
        elif i == 3: 
            counts.reindex(['tcp', 'udp', 'icmp']).plot(kind='bar', grid=True, color='red')
        else: 
            counts.plot(kind='bar', grid=True, color='blue')
        
        imgs.append(get_base64_graph()) # get_base64_graph ya contiene plt.close('all')
        
    return imgs

# --- ARCHIVO 08: PARTICIONADO Y ESCALADO ---
def generate_data_processing_08():
    df = load_kdd_data()
    if df is None: return {}
    
    train_set, test_set = train_test_split(df, test_size=0.4, random_state=42, stratify=df['protocol_type'])
    X_train = train_set.drop("class", axis=1)
    X_train_num = X_train.select_dtypes(exclude=['object'])
    
    imputer = SimpleImputer(strategy="median")
    X_train_clean_all = pd.DataFrame(imputer.fit_transform(X_train_num), columns=X_train_num.columns)
    
    robust_scaler = RobustScaler()
    X_train_scaled_all = pd.DataFrame(
        robust_scaler.fit_transform(X_train_clean_all), 
        columns=X_train_clean_all.columns
    )

    X_train_with_nulls = X_train.copy()
    # Muestreo seguro para insertar nulos
    sample_size = min(10, len(X_train_with_nulls))
    X_train_with_nulls.loc[X_train_with_nulls.sample(sample_size).index, "src_bytes"] = np.nan
    filas_nulas = X_train_with_nulls[X_train_with_nulls.isnull().any(axis=1)].head(10)

    return {
        "train_len": len(train_set), 
        "test_len": len(test_set),
        "nulos": filas_nulas.to_html(classes='table table-danger table-sm'),
        "X_train_clean": X_train_clean_all.head(5).to_html(classes='table table-success table-sm'),
        "X_train_scaled": X_train_scaled_all.head(10).to_html(classes='table table-warning table-sm')
    }

# --- ARCHIVO 05: NLP ---
def generate_email_processing_05():
    df = load_kdd_data()
    if df is None: return {"error": "Archivo no encontrado"}

    if 'subject' in df.columns and 'body' in df.columns:
        mail = {
            'subject': str(df['subject'].iloc[0]).split(),
            'body': str(df['body'].iloc[0]).split(),
            'content_type': str(df['content_type'].iloc[0])
        }
    else:
        mail = {
            'subject': ['gener', 'ciali', 'brand', 'qualiti'],
            'body': ['do', 'feel', 'pressur', 'perform', 'rise', 'occas', 'tri', 'viagra', 'anxieti', 'thing', 'past', 'back', 'old', 'self'],
            'content_type': 'multipart/alternative'
        }

    prep_email_list = [" ".join(mail['subject']) + " " + " ".join(mail['body'])]
    vectorizer = CountVectorizer()
    vectorizer.fit(prep_email_list)
    cv_features = vectorizer.get_feature_names_out()

    tokens = mail['subject'] + mail['body']
    prep_email_oh = [[w] for w in tokens]
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_oh = enc.fit_transform(prep_email_oh)
    oh_features = enc.get_feature_names_out()

    return {
        "mail_dict": mail,
        "prep_email_text": str(prep_email_list),
        "cv_features": str(cv_features),
        "oh_features": str(oh_features),
        "oh_values": np.array2string(X_oh, separator=' ', formatter={'float_kind':lambda x: "%.1f" % x})
    }

# --- ARCHIVO 09: PIPELINE AVANZADO ---
def generate_pipeline_processing_09():
    df = load_kdd_data()
    if df is None: return {}
    # Reducimos aún más el set para el pipeline de visualización
    train_set, _ = train_test_split(df, test_size=0.4, random_state=42)
    X_train = train_set.drop("class", axis=1).head(200).copy()
    
    # Simulación de nulos
    X_train.iloc[0:5, 0] = np.nan 
    table_1 = X_train[X_train.isnull().any(axis=1)].head(10).to_html(classes='table table-sm table-secondary')

    delete_nan = DeleteNanRows()
    X_train_prep_rows = delete_nan.fit_transform(X_train)
    table_2 = X_train_prep_rows.head(10).to_html(classes='table table-sm table-info')

    num_attribs = list(X_train.select_dtypes(exclude=['object']).columns)
    cat_attribs = ["protocol_type", "service", "flag"]
    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")), ('std_scaler', StandardScaler())])
    full_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs), ("cat", OneHotEncoder(handle_unknown='ignore'), cat_attribs)])
    
    X_train_prep = full_pipeline.fit_transform(X_train)
    column_names = full_pipeline.get_feature_names_out()
    X_train_prep_df = pd.DataFrame(X_train_prep, columns=column_names, index=X_train.index)
    table_3 = X_train_prep_df.head(10).to_html(classes='table table-sm table-primary')

    return {"table_1": table_1, "table_2": table_2, "table_3": table_3}
