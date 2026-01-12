import arff
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin

# ==========================================
# 1. FUNCIONES DE CARGA Y PARTICIONADO
# ==========================================

def load_kdd_dataset(data_path):
    """Lectura del Dataset NSL-KDD"""
    with open(data_path, 'r') as train_set:
        dataset = arff.load(train_set)
        attributes = [attr[0] for attr in dataset['attributes']]
        return pd.DataFrame(dataset["data"], columns=attributes)

def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    """Construccion de una funcion que realice el particionado completo."""
    strat = df[stratify] if stratify else None
    train_set, test_Set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    
    strat = test_Set[stratify] if stratify else None 
    val_set, test_Set = train_test_split(
        test_Set, test_size=0.5,  random_state=rstate, shuffle=shuffle, stratify=strat)
    
    return(train_set, val_set, test_Set)

# ==========================================
# 2. CARGA Y DIVISIÓN INICIAL
# ==========================================

df = load_kdd_dataset("/home/pred/Documentos/datasets/datasets/NSL-KDD/KDDTrain+.arff")

train_set, val_set, test_set = train_val_test_split(df, stratify='protocol_type')

print("longitud del Training Set:", len(train_set))
print("longitud del Validation Set:", len(val_set))
print("longitud del Test Set:", len(test_set))

# Separar las caracteristicas de entrada de las caracteristicas de salida
X_train = train_set.drop("class", axis=1)
y_train = train_set["class"].copy()

# ==========================================
# 3. MANIPULACIÓN DE VALORES NULOS
# ==========================================

## PARA FACILITAR ESTA SECCION ES NECESARIO AÑADIR ALGUNOS VALORES NULOS
X_train.loc[(X_train["src_bytes"]>400) &(X_train["src_bytes"] < 800), "src_bytes"] = np.nan
X_train.loc[(X_train["src_bytes"]>500) &(X_train["dst_bytes"] < 2000), "dst_bytes"] = np.nan

# Comprobar si existe algun atributo con valores nulos
print("\n¿Existen nulos?:", X_train.isna().any().any())

# Seleccionar las filas que contengan valores nulos
filas_valores_nulos = X_train[X_train.isnull().any(axis=1)]

# --- OPCIÓN 1: Eliminar filas con nulos ---
X_train_copy = X_train.copy()
X_train_copy.dropna(subset = ["src_bytes", "dst_bytes"], inplace=True)
print("el numero de filas eliminadas es:", len((X_train)) - len((X_train_copy)))

# --- OPCIÓN 2: Eliminar atributos (columnas) ---
X_train_copy = X_train.copy()
X_train_copy.drop([ "src_bytes", "dst_bytes"], axis=1, inplace=True)
print("el numero de atributos eliminados es:", len(list(X_train)) - len(list(X_train_copy)))

# --- OPCIÓN 3: Rellenar con la media ---
X_train_copy = X_train.copy()
media_srcbytes = X_train_copy["src_bytes"].mean()
media_dstbytes = X_train_copy["dst_bytes"].mean()

X_train_copy["src_bytes"].fillna(media_srcbytes, inplace=True)
X_train_copy["dst_bytes"].fillna(media_dstbytes, inplace=True)

# --- OPCIÓN 4: Uso de SimpleImputer ---
X_train_copy = X_train.copy()
imputer = SimpleImputer(strategy = "median")

# LA CLAVE IMPUTER NO ACEPTA VALORES CATEGORICOS; ELIMINAR LOS ATRIBUTO CATEGORICOS
X_train_copy_num = X_train_copy.select_dtypes(exclude = ['object'])
imputer.fit(X_train_copy_num)

# Rellenar los valores nulos
X_train_copy_num_nonan = imputer.transform(X_train_copy_num)
# Transformar el resultado a un DataFrame de Pandas
X_train_copy = pd.DataFrame(X_train_copy_num_nonan, columns=X_train_copy_num.columns)

# ==========================================
# 4. CODIFICACIÓN DE VARIABLES CATEGÓRICAS
# ==========================================

X_train = train_set.drop("class", axis=1)
y_train = train_set["class"].copy()

# --- Método 1: Factorize ---
protocol_type = X_train['protocol_type']
protocol_type_encode, categorias = protocol_type.factorize()

print("\nEjemplo Factorize:")
for i in range(10):
    print(protocol_type.iloc[i], "=", protocol_type_encode[i])
print(categorias)

# --- Método 2: OrdinalEncoder ---
protocol_type = X_train[['protocol_type']]
ordinal_encoder = OrdinalEncoder()
protocol_type_encode = ordinal_encoder.fit_transform(protocol_type)

print("\nEjemplo OrdinalEncoder:")
for i in range(10):
    print(protocol_type["protocol_type"].iloc[i], "=", protocol_type_encode[i])
print(ordinal_encoder.categories_)

# --- Método 3: OneHotEncoder ---
oh_encoder = OneHotEncoder()
protocol_type_oh = oh_encoder.fit_transform(protocol_type)

print("\nEjemplo OneHotEncoder (a array):")
print(protocol_type_oh.toarray())

for i in range(10):
    print(protocol_type["protocol_type"].iloc[i], "=", protocol_type_oh.toarray()[i])

# --- Método 4: Get Dummies ---
oh_encoder_ignore = OneHotEncoder(handle_unknown='ignore')
dummies = pd.get_dummies(X_train['protocol_type'])

# ==========================================
# 5. ESCALADO DE DATOS
# ==========================================

X_train = train_set.drop("class", axis=1)
y_train = train_set["class"].copy()

scale_attrs = X_train[['src_bytes', 'dst_bytes']]
robust_scaler = RobustScaler()
X_train_scaled = robust_scaler.fit_transform(scale_attrs)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=['src_bytes', 'dst_bytes'])

# ==========================================
# 6. TRANSFORMADOR PERSONALIZADO (CLASE)
# ==========================================

class CustomOneHotEncoding(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._oh = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self._columns = None

    def fit(self, X, y=None):
        X_cat = X.select_dtypes(include=['object'])
        self._oh.fit(X_cat)
        self._columns = self._oh.get_feature_names_out(X_cat.columns)
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_cat = X_copy.select_dtypes(include=['object'])
        X_num = X_copy.select_dtypes(exclude=['object'])

        X_cat_oh = pd.DataFrame(self._oh.transform(X_cat),
                                columns=self._columns,
                                index=X_copy.index)

        return pd.concat([X_num, X_cat_oh], axis=1)

# Ejemplo de uso de la clase
custom_oh = CustomOneHotEncoding()
X_train_final = custom_oh.fit_transform(X_train)
print("\nDataset final tras CustomOneHotEncoding:")
print(X_train_final.head())