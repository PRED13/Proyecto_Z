import arff
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# ==========================================
# 1. FUNCIONES DE CARGA Y PARTICIONADO
# ==========================================

def load_kdd_dataset(data_path):
    """Lectura del Dataset NSL-KDD"""
    with open(data_path, 'r') as train_set:
        dataset = arff.load(train_set)
        attributes = [attr[0] for attr in dataset['attributes']]
        return pd.DataFrame(dataset["data"], columns=attributes)

# Construccion de uuna funcion que realice el particionado completo. 
def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_Set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_Set[stratify] if stratify else None 
    val_set, test_Set = train_test_split(
        test_Set, test_size=0.5,  random_state=rstate, shuffle=shuffle, stratify=strat)
    return(train_set, val_set, test_Set)

# ==========================================
# 2. CARGA Y PREPARACIÓN INICIAL
# ==========================================

df = load_kdd_dataset("/home/pred/Documentos/datasets/datasets/NSL-KDD/KDDTrain+.arff")

# Ejecución del split
train_set, val_set, test_set = train_val_test_split(df, stratify='protocol_type')

print("longitud del Training Set:", len(train_set))
print("longitud del Validation Set:", len(val_set))
print("longitud del Test Set:", len(test_set))

X_train = train_set.drop("class", axis=1)
y_train = train_set["class"].copy() # Corregido: añadido ()

## PARA FACILITAR ESTA SECCION ES NECESARIO AÑADIR ALGUNOS VALORES NLOS
X_train.loc[(X_train["src_bytes"]>400) &(X_train["src_bytes"] < 800), "src_bytes"] = np.nan
X_train.loc[(X_train["src_bytes"]>500) &(X_train["dst_bytes"] < 2000), "dst_bytes"] = np.nan

# ==========================================
# 3. TRANSFORMADORES PERSONALIZADOS
# ==========================================

# Transfomador creado para eliminar las filas con valores nulos
class DeleteNanRows(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self 
    def transform(self, X, y=None):
        return X.dropna()

delete_nan = DeleteNanRows()
X_train_prep = delete_nan.fit_transform(X_train)

# Tranformador diseñado para escalar de manera sencilla unicamente unas columnas seleccionadas
class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, attributes):
        self.attributes = attributes
    def fit(self, X, y=None):
        return self #nada mas que hacer
    
    def transform(self, X, y=None):
        X_copy = X.copy()
        scale_attrs = X_copy[self.attributes]
        robust_scaler = RobustScaler()
        X_scaled = robust_scaler.fit_transform(scale_attrs)
        X_scaled = pd.DataFrame(X_scaled, columns=self.attributes, index=X_copy.index)
        for attr in self.attributes:
            X_copy[attr] = X_scaled[attr]
        return X_copy # Corregido: movido fuera del bucle para devolver el df completo

custom_scaler = CustomScaler(["src_bytes"]) 
X_train_prep = custom_scaler.fit_transform(X_train_prep)

# ==========================================
# 4. PIPELINES Y TRANSFORMACIÓN FINAL
# ==========================================

# Construcción de un pipeline para los atributos númericos
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('rbst_scaler', RobustScaler()),
])

# Nota: Tu código original contenía una clase CustomOneHotEncoding con algunos errores 
# de sintaxis. Para que el Pipeline funcione, se utiliza la lógica de ColumnTransformer.

# La clase imputer no admite valores categoricos, se eliminan los atributos categoricos.
X_train_num = X_train.select_dtypes(exclude=['object'])

X_train_prep_num = num_pipeline.fit_transform(X_train_num)
X_train_prep_num_df = pd.DataFrame(X_train_prep_num, columns=X_train_num.columns, index=X_train_num.index)

# Integración total con ColumnTransformer
num_attributes = list(X_train.select_dtypes(exclude=['object']))
cat_attribs = list(X_train.select_dtypes(include=['object']))

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attributes),
    ("cat", OneHotEncoder(), cat_attribs),
])

# Transformación final
X_train_prep_final = full_pipeline.fit_transform(X_train)

# Obtenemos los nombres reales de las columnas generadas por el pipeline
column_names = full_pipeline.get_feature_names_out()

# Creamos el DataFrame usando los nombres correctos
X_train_prep_df = pd.DataFrame(X_train_prep_final, 
                               columns=column_names, 
                               index=X_train.index)

# ==========================================
# 5. VISUALIZACIÓN DE RESULTADOS
# ==========================================

print("\n--- X_train Original (10 filas) ---")
print(X_train.head(10))

print("\n--- X_train_prep_df Procesado (Pipeline Final) ---")
print(X_train_prep_df.head())