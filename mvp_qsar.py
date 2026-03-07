import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

print("\n[1] Cargando el dataset de serotonina...")
# Leemos el archivo que scrapeaste en el paso anterior
df = pd.read_csv('E:\\proyecto-ia\\antidepresivos\\dataset_serotonina.csv')
df = df.dropna(subset=['SMILES', 'Eficacia (pActivity)'])

print("[2] RDKit procesando: Convirtiendo SMILES a Huellas Dactilares (ECFP4)...")
def smiles_a_matriz(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return np.zeros((2048,))
    # Sacamos la huella dactilar de 2048 bits (El mismo estándar del paper de Icesi)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

# Aplicamos la función a toda la columna del dataset
X = np.array([smiles_a_matriz(s) for s in df['SMILES']])
y = df['Eficacia (pActivity)'].values

print(f"    -> Matriz lista: {X.shape[0]} moléculas convertidas a código binario.")

print("\n[3] Entrenando el cerebro predictivo (XGBoost)...")
# Separamos los datos: 80% para que la IA aprenda, 20% para ponerla a prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializamos y entrenamos el modelo
modelo = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
modelo.fit(X_train, y_train)

print("\n[+] ¡MVP COMPLETADO! Evaluando resultados...")
predicciones = modelo.predict(X_test)

print("-" * 50)
# Mostramos 3 moléculas de prueba para comparar qué tan cerca estuvo la IA de la realidad
print("Prueba en moléculas que el modelo nunca había visto:")
for i, (real, pred) in enumerate(zip(y_test[:3], predicciones[:3])):
    print(f"  Molécula {i+1} -> Eficacia real de laboratorio: {real:.2f} | El modelo predijo: {pred:.2f}")
print("-" * 50)
print("\n[!] El flujo de trabajo In Silico funciona de extremo a extremo.")