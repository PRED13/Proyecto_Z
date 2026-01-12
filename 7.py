import arff
import pandas as pd

def load_kdd_dataset(data_path):
    """Lectura del Dataset NSL-KDD"""
    with open(data_path, 'r') as train_set:
        dataset = arff.load(train_set)
        attributes = [attr[0] for attr in dataset['attributes']]
        return pd.DataFrame(dataset["data"], columns=attributes)
    
df = load_kdd_dataset("/home/pred/Documentos/datasets/datasets/NSL-KDD/KDDTrain+.arff")

from sklearn.model_selection import train_test_split
train_set, test_Set = train_test_split(df, test_size=0.4, random_state=42)

 # Separar el DATASET del pruebas 50% validation set; 50% test set
val_set, test_Set = train_test_split(test_Set, test_size=0.5, random_state=42)

print("Longitud del Training Set:", len(train_set))
print("Longitud del Validacion Set:", len(val_set))
print("Longitud del Train Set:", len(test_Set))

# Si Shuffle = False, el DataSet no mezclara antes del particionado. 
train_set, test_Set = train_test_split(df, test_size=0.4, random_state=42, shuffle=False)

train_set, test_Set = train_test_split(df, test_size=0.4, random_state=42, stratify=df["protocol_type"])

# Construccion de uuna funcion que realice el particionado completo. 
def train_val_test_split(df, rstate= 42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_Set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_Set[stratify] if stratify else None 
    val_set, test_Set = train_test_split(
        test_Set, test_size=0.5,  random_state=rstate, shuffle=shuffle, stratify=strat)
    return(train_set, val_set, test_Set)

print("Longitud del DataSet:", len(df))

train_set, val_set, test_Set = train_val_test_split(df, stratify='protocol_type')

print("Longitud del Training Set:", len(train_set))
print("Longitud del Validacion Set:", len(val_set))
print("Longitud del Train Set:", len(test_Set))

# Comparacion de que statify manietiene la proporcion de la caracteristica en los conjuntos. 
#grafica 1
import matplotlib.pyplot as plt
df['protocol_type'].hist()
plt.show()

#grafica 2
train_set['protocol_type'].hist()
plt.show()

#grafica 3
val_set['protocol_type'].hist()
plt.show()

#grafica 4
test_Set['protocol_type'].hist()
plt.show()