from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem

# 1. El código SMILES de la Fluoxetina (Prozac)
smiles_fluoxetina = "CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F"

# 2. RDKit "renderiza" la molécula a partir del texto
molecula = Chem.MolFromSmiles(smiles_fluoxetina)

# 3. Calculamos descriptores matemáticos clave
peso_molecular = Descriptors.MolWt(molecula)
log_p = Descriptors.MolLogP(molecula) # Qué tanto penetra barreras celulares
tpsa = Descriptors.TPSA(molecula)     # Superficie polar

# 4. Sacamos la Huella Dactilar (ECFP4 a 2048 bits)
# Esto convierte la estructura química en ceros y unos para el Machine Learning
huella_dactilar = AllChem.GetMorganFingerprintAsBitVect(molecula, 2, nBits=2048)
vector_bits = list(huella_dactilar)

# 5. Mostramos la magia en la terminal
print("\n=== ANÁLISIS IN SILICO: FLUOXETINA (PROZAC) ===")
print(f"SMILES original: {smiles_fluoxetina}")
print("-" * 45)
print(f"Peso Molecular:       {peso_molecular:.2f} Da")
print(f"Lipofilicidad (LogP): {log_p:.2f}")
print(f"Polaridad (TPSA):     {tpsa:.2f}")
print("-" * 45)
print(f"Huella Dactilar Binaria (Primeros 60 bits de 2048):")
print(vector_bits[:60])
print("... (continúa hasta 2048 bits)")
print("\n[+] ¡Matriz de datos lista para alimentar el modelo predictivo XGBoost!")