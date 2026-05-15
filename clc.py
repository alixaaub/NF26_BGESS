import os
import sys
from datetime import datetime, timedelta
import warnings

# Configuration de l'environnement Spark / PyArrow
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

warnings.filterwarnings("ignore", message="If `index_col` is not specified*")

import pyspark.pandas as ps
from pyspark.sql import SparkSession
# Import propre des fonctions Spark SQL avec alias pour éviter les conflits
import pyspark.sql.functions as F
from sklearn.linear_model import LinearRegression

# Initialisation de la Session Spark
spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.session.timeZone", "UTC")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_DIR = os.path.abspath("BDD_BGES/BDD_BGES")
SITES = ["BERLIN", "LONDON", "LOSANGELES", "NEWYORK", "PARIS", "SHANGHAI"]
IMPACT_PATH = os.path.join(BASE_DIR, "materiel_informatique_impact.csv")
LANGUE_CIBLE = "fr"

TRANSLATIONS_MATRIX = {
    "Ökonom": "Economiste",
    "Führungskraft": "Cadre",
    "Personalleiter": "DRH",
    "Computeringenieur": "Ingénieur Informaticien",
    "Dateningenieur": "Ingénieur Data",
    "Economist": "Economiste",
    "Business Executive": "Cadre",
    "HRD": "DRH",
    "Computer Engineer": "Ingénieur Informaticien",
    "Data Engineer": "Ingénieur Data",
    "Geschäftstreffen": "Rencontre entreprises",
    "Konferenz": "Conférence",
    "Schulung": "Formation",
    "Meeting": "Réunion",
    "Entwicklung": "Développement",
    "Conference": "Conférence",
    "Vocational Training": "Formation",
    "Team Meeting": "Réunion",
    "Business Meeting": "Rencontre entreprises",
    "Development": "Développement"
}

# Dictionnaire pour l'API Pandas-on-Spark
df_traductions = ps.DataFrame(
    list(TRANSLATIONS_MATRIX.items()), 
    columns=["MOT_ORIGINE", "MOT_TRADUIT"]
)

# ==============================================================================
# FONCTIONS UTILITAIRES & ETAPES ETL
# ==============================================================================

def init_constellation() -> dict:
    print("--- Initialisation du Schéma Constellation ---")
    
    # 1A. Référentiel IMPACT
    psdf_impact = ps.read_csv(IMPACT_PATH)
    psdf_impact.columns = [str(c).strip().upper().replace("È", "E") for c in psdf_impact.columns]
    
    # 1B. Référentiel PERSONNEL
    personnel_path = f"file://{BASE_DIR}/BDD_BGES_*/PERSONNEL_*.txt"
    try:
        psdf_pers = ps.read_csv(personnel_path, sep=";")
        psdf_pers = psdf_pers.drop_duplicates()

        # Jointure pour récupérer la traduction
        psdf_pers = psdf_pers.merge(df_traductions, left_on='FONCTION_PERSONNEL', right_on='MOT_ORIGINE', how='left')
        
        # --- CORRECTION DU TYPE ERROR ICI ---
        # On passe en Spark natif pour utiliser la fonction coalesce (100% stable)
        spark_df = psdf_pers.to_spark()
        # coalesce prend la valeur de MOT_TRADUIT, si elle est nulle, elle prend FONCTION_PERSONNEL
        spark_df = spark_df.withColumn("FONCTION_PERSONNEL", F.coalesce(F.col("MOT_TRADUIT"), F.col("FONCTION_PERSONNEL")))
        # On repasse en pyspark pandas
        psdf_pers = spark_df.pandas_api()
        
        # --- CORRECTION DE L'EXTRACTION ICI ---
        # Au lieu de .extract() qui peut planter selon la version, on utilise split et str.get()
        # "PERS_BERLIN_001" -> on split sur "_" et on prend l'élément à l'index 1 ("BERLIN")
        psdf_pers["SITE"] = psdf_pers["ID_PERSONNEL"].str.split("_").str.get(1)

        dim_personnel = psdf_pers[['ID_PERSONNEL', 'FONCTION_PERSONNEL', 'DT_NAISS', 'SITE']].drop_duplicates()
        
    except Exception as e:
        print(f"Attention : aucun fichier personnel trouvé ({e}). Tables initialisées vides.")
        dim_personnel = ps.DataFrame(columns=['ID_PERSONNEL', 'FONCTION_PERSONNEL', 'DT_NAISS', 'SITE'])
        
    schema_initial = {
        "DF_IMPACT": psdf_impact, 
        "DIM_PERSONNEL": dim_personnel,
        "DIM_MISSION": ps.DataFrame(columns=["ID_MISSION", "TYPE_MISSION", "VILLE_DEPART", "PAYS_DEPART", "VILLE_DESTINATION", "PAYS_DESTINATION", "TRANSPORT", "ALLER_RETOUR"]),
        "DIM_MATERIEL": ps.DataFrame(columns=["ID_MATERIEL", "TYPE", "MODELE"]),
        "FAIT_MISSION": ps.DataFrame(columns=["SITE", "ID_PERSONNEL", "ID_MISSION", "ID_DATE_MISSION"]),
        "FAIT_MATERIEL": ps.DataFrame(columns=["SITE", "ID_PERSONNEL", "ID_MATERIEL", "ID_DATE_ACHAT"])
    }
    print("[OK] Schéma de constellation initialisé.")
    return schema_initial


def etl_daily_missions(date_str: str, schema_existant: dict) -> dict:
    print(f"\n--- Lancement ETL Transactionnel pour le : {date_str} ---")
    
    psdf_impact = schema_existant["DF_IMPACT"]
    
    mission_path = f"file://{BASE_DIR}/BDD_BGES_*/BDD_BGES_*_MISSION/MISSION_{date_str}.txt"
    it_path = f"file://{BASE_DIR}/BDD_BGES_*/BDD_BGES_*_INFORMATIQUE/MATERIEL_INFORMATIQUE_{date_str}.txt"
    
    mission_ok = False
    it_ok = False

    try:
        psdf_mission = ps.read_csv(mission_path, sep=";")
        if not psdf_mission.empty:
            mission_ok = True
    except:
        psdf_mission = ps.DataFrame()

    try:
        psdf_it = ps.read_csv(it_path, sep=";")
        if not psdf_it.empty:
            it_ok = True
    except:
        psdf_it = ps.DataFrame()

    # --- TRANSFORM ---
    if mission_ok:
        psdf_mission = psdf_mission.drop_duplicates()
        psdf_mission['TYPE_MISSION'] = psdf_mission['TYPE_MISSION'].replace(TRANSLATIONS_MATRIX)
        psdf_mission['TRANSPORT'] = psdf_mission['TRANSPORT'].replace(TRANSLATIONS_MATRIX)
        psdf_mission = handle_missing_values(psdf_mission)
        
    if it_ok: 
        psdf_it = psdf_it.drop_duplicates()
        psdf_it = psdf_it.merge(psdf_impact, on=['TYPE', 'MODELE'], how="left")
        psdf_it = handle_missing_values(psdf_it)

    # --- LOAD ---

    # Mise à jour DIM_MISSION & FAIT_MISSION
    if mission_ok and not psdf_mission.empty:
        dim_mission_jour = psdf_mission[
            ["ID_MISSION", "TYPE_MISSION", "VILLE_DEPART", "PAYS_DEPART", 
             "VILLE_DESTINATION", "PAYS_DESTINATION", "TRANSPORT", "ALLER_RETOUR"]
        ].drop_duplicates()
        schema_existant["DIM_MISSION"] = ps.concat([schema_existant["DIM_MISSION"], dim_mission_jour]).drop_duplicates(["ID_MISSION"])
        
        fait_mission_jour = psdf_mission[["ID_PERSONNEL", "ID_MISSION", "DATE_MISSION"]].drop_duplicates()
        fait_mission_jour = fait_mission_jour.rename(columns={"DATE_MISSION": "ID_DATE_MISSION"})
        
        # Remplacement ici aussi par str.get(1)
        fait_mission_jour["SITE"] = fait_mission_jour["ID_PERSONNEL"].str.split("_").str.get(1)

        schema_existant["FAIT_MISSION"] = ps.concat([schema_existant["FAIT_MISSION"], fait_mission_jour]).drop_duplicates(["ID_PERSONNEL", "ID_MISSION"])

    # Mise à jour DIM_MATERIEL & FAIT_MATERIEL
    if it_ok and not psdf_it.empty:
        dim_mat_jour = psdf_it[["ID_MATERIELINFO", "TYPE", "MODELE"]].drop_duplicates().rename(columns={"ID_MATERIELINFO": "ID_MATERIEL"})
        schema_existant["DIM_MATERIEL"] = ps.concat([schema_existant["DIM_MATERIEL"], dim_mat_jour]).drop_duplicates(["ID_MATERIEL"])
        
        fait_mat_jour = psdf_it[["ID_PERSONNEL", "ID_MATERIELINFO", "DATE_ACHAT"]].drop_duplicates().rename(columns={
            "ID_MATERIELINFO": "ID_MATERIEL", 
            "DATE_ACHAT": "ID_DATE_ACHAT"
        })
        
        # Remplacement ici aussi par str.get(1)
        fait_mat_jour["SITE"] = fait_mat_jour["ID_PERSONNEL"].str.split("_").str.get(1)
        
        schema_existant["FAIT_MATERIEL"] = ps.concat([schema_existant["FAIT_MATERIEL"], fait_mat_jour]).drop_duplicates(["ID_PERSONNEL", "ID_MATERIEL"])

    print(f"[SUCCES] Données du jour fusionnées.")
    return schema_existant


def handle_missing_values(df: ps.DataFrame, strategy: str = "mean", target_col: str = None, feature_cols: list = None) -> ps.DataFrame:
    """Complète les infos manquantes par moyenne ou régression linéaire"""
    if df.empty:
        return df

    if strategy == "mean" and target_col:
        # Ici fillna est OK car la moyenne est une valeur scalaire (un chiffre), pas une Series !
        df[target_col] = df[target_col].fillna(df[target_col].mean())

    elif strategy == "regression" and target_col and feature_cols:
        train_data = df.dropna(subset=feature_cols + [target_col])
        missing_data = df[df[target_col].isnull() & df[feature_cols].notnull().all(axis=1)]

        if not missing_data.empty and not train_data.empty:
            X_train = train_data[feature_cols].to_pandas()
            y_train = train_data[target_col].to_pandas()
            X_missing = missing_data[feature_cols].to_pandas()

            model = LinearRegression()
            model.fit(X_train, y_train)
            
            predictions = model.predict(X_missing)
            df_pandas = df.to_pandas()
            df_pandas.loc[missing_data.index.to_pandas(), target_col] = predictions
            df = ps.from_pandas(df_pandas)

    return df


def standardize_timezone(df: ps.DataFrame, column: str) -> ps.DataFrame:
    """Convertit une colonne de dates en datetime standardisé."""
    if column in df.columns:
        spark_df = df.to_spark()
        spark_df = spark_df.withColumn(column, F.to_timestamp(F.col(column)))
        return spark_df.pandas_api()
    return df


def ask_questions(rep: str) -> None:
    pass

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    schema = init_constellation()

    current_date = datetime(2026, 4, 29)
    end_date = datetime(2026, 11, 5)
    delta = timedelta(days=1)

    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        print(f"\nLancement du processus ETL pour le jour : {date_str}")

        schema = etl_daily_missions(date_str=date_str, schema_existant=schema)

        print("\n--- FAIT_MISSION (head) ---")
        print(schema["FAIT_MISSION"].head(5))
        print("\n--- FAIT_MATERIEL (head) ---")
        print(schema["FAIT_MATERIEL"].head(5))

        rep = str(input("\nSouhaitez-vous poser une question? (0 pour continuer, q pour quitter) : "))
        if rep.lower() == "q":
            break
        elif rep != "0":
            ask_questions(rep)
            
        print("\nOn passe au jour suivant!")
        current_date += delta

    print("Processus terminé avec succès.")


if __name__ == "__main__":
    main()