import pandas as pd
import pyarrow.parquet as pq

print("=== 1. INFORMACIÓN DE TAMAÑO Y ESTRUCTURA ===")

# Para train.parquet (analizamos filas y row groups con PyArrow)
parquet_file = pq.ParquetFile("/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra/train.parquet")
print(f"train.parquet -> Filas exactas: {parquet_file.metadata.num_rows:,}")
print(f"train.parquet -> Número de row groups: {parquet_file.metadata.num_row_groups}")

# Para train.csv y submissions.csv (contamos filas con pandas, de forma eficiente)
# Nota: Si son archivos muy grandes, contar las filas exactas puede tardar unos segundos.
train_csv_rows = sum(1 for _ in open("train.csv", encoding="utf-8")) - 1 # Restando el header
sub_rows = sum(1 for _ in open("submissions.csv", encoding="utf-8")) - 1

print(f"train.csv     -> Filas exactas (sin header): {train_csv_rows:,}")
print(f"submissions.csv -> Filas exactas (sin header): {sub_rows:,}")

print("\n=== 2. NOMBRES Y TIPOS DE COLUMNAS (train.parquet) ===")
# Cargamos los metadatos del parquet para ver las columnas y tipos sin cargar todo a RAM
df_parquet_sample = pd.read_parquet("train.parquet", nrows=sub_rows)
print(df_parquet_sample.dtypes)

print("\n=== 3. ANÁLISIS DE MOLÉCULAS, ESPECTROS Y EJEMPLOS (train.parquet) ===")
# Asumiendo columnas estándar en este tipo de datasets químicos (como mass spec / metabolómica)
# Ajusta los nombres de columnas ('molecule_id', 'smiles', 'ms2_mzs', etc.) si difieren ligeramente en tu CSV/Parquet.

# Cargamos una muestra o leemos el parquet completo si la RAM lo permite
df = pd.read_parquet("train.parquet")

# Detectar nombres de columnas clave de forma flexible
col_smiles = next((c for c in df.columns if 'smiles' in c.lower()), None)
col_molecule = next((c for c in df.columns if 'molecule' in c.lower() or 'id' in c.lower()), None)
col_mzs = next((c for c in df.columns if 'mz' in c.lower() or 'mass' in c.lower()), None)
col_intensities = next((c for c in df.columns if 'intensity' in c.lower() or 'intensities' in c.lower()), None)

print(f"Columnas detectadas -> ID/Molécula: {col_molecule}, SMILES: {col_smiles}, MZs: {col_mzs}, Intensidades: {col_intensities}")

if col_molecule and col_smiles:
    # 5-10 moléculas reales y cuántos espectros tiene cada una (agrupando por molécula)
    conteo_espectros = df.groupby([col_molecule, col_smiles]).size().reset_index(name='num_espectros')
    muestra_moleculas = conteo_espectros.head(10)
    
    print("\n--- 5-10 Moléculas Reales y sus Conteo de Espectros ---")
    for idx, row in muestra_moleculas.iterrows():
        print(f"Molécula ID: {row[col_molecule]} | SMILES: {row[col_smiles]} | Espectros: {row['num_espectros']}")

# Un ejemplo real de ms2_mzs y ms2_normalized_intensities
if col_mzs and col_intensities:
    print("\n--- Ejemplo Real de Espectro (MS2) ---")
    ejemplo_mz = df[col_mzs].iloc[0]
    ejemplo_int = df[col_intensities].iloc[0]
    
    print(f"ms2_mzs (Ejemplo): {ejemplo_mz[:10]} ... (muestra parcial de array)")
    print(f"ms2_normalized_intensities (Ejemplo): {ejemplo_int[:10]} ... (muestra parcial de array)")
else:
    print("\n[!] No se pudieron mapear automáticamente las columnas de espectros MS2. Revisa df.columns.")
