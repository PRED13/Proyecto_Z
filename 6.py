import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from pandas.plotting import scatter_matrix

# --- 1. INSTALACIÓN DE DEPENDENCIAS ---
try:
    import arff
except ImportError:
    import subprocess
    print("Instalando liac-arff...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "liac-arff"])
    import arff

# --- 2. DEFINICIÓN DE FUNCIONES ---

def load_kdd_dataset(data_path):
    """Lectura del Dataset NSL-KDD en formato .arff"""
    with open(data_path, 'r') as train_set:
        dataset = arff.load(train_set)
        attributes = [attr[0] for attr in dataset['attributes']]
        return pd.DataFrame(dataset["data"], columns=attributes)

# --- 3. CARGA DE DATOS ---

PATH_DATASET = "/home/pred/Documentos/datasets/datasets/NSL-KDD/KDDTrain+.arff"

if not os.path.exists(PATH_DATASET):
    print(f"Error: No se encontró el archivo en {PATH_DATASET}")
    sys.exit()

print("Cargando dataset...")
df_orig = load_kdd_dataset(PATH_DATASET)
df = df_orig.copy()

# --- 4. EXPLORACIÓN INICIAL ---

print("\n--- Primeras 10 filas ---")
print(df.head(10))

print("\n--- Información del Dataset ---")
df.info()

# --- 5. VISUALIZACIÓN DE DATOS (PARTE 1) ---

print("\nGenerando gráficas iniciales... (Cierra la ventana para continuar)")
# Gráfica 1: Histograma de Protocol Type
plt.figure(figsize=(8, 6))
df["protocol_type"].value_counts().plot(kind='bar')
plt.title("Distribución de Protocol Type")
plt.xlabel("Tipo de Protocolo")
plt.ylabel("Frecuencia")
plt.show() # Bloquea hasta que cierres la ventana

# Gráfica 2: Histogramas generales
df.hist(bins=50, figsize=(20, 15))
plt.suptitle("Distribución de atributos numéricos")
plt.show() 

# --- 6. PREPROCESAMIENTO ---

print("\nTransformando etiquetas de 'class' a numéricas...")
le = LabelEncoder()
df["class"] = le.fit_transform(df["class"])

# --- 7. ANÁLISIS DE CORRELACIÓN ---

# Seleccionamos solo columnas numéricas para evitar errores
numeric_df = df.select_dtypes(include=[np.number])

print("\n--- Correlación con respecto a la clase ---")
corr_matrix = numeric_df.corr()
print(corr_matrix["class"].sort_values(ascending=False))

# Gráfica 3: Representación gráfica de la matriz de correlación
fig, ax = plt.subplots(figsize=[12, 12])
cax = ax.matshow(corr_matrix, cmap='coolwarm')
fig.colorbar(cax)

plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title("Matriz de Correlación", pad=20)
plt.show()

# --- 8. MATRIZ DE DISPERSIÓN (SCATTER MATRIX) ---

print("\nGenerando matriz de dispersión...")
# Gráfica 4: Scatter Matrix
selected_attributes = ["same_srv_rate", "dst_host_srv_count", "class", "dst_host_same_srv_rate"]
scatter_matrix(df[selected_attributes], figsize=[12, 8], alpha=0.3)
plt.suptitle("Matriz de Dispersión de Atributos Seleccionados")
plt.show()

print("\nProceso finalizado correctamente.")