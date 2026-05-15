import os
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
import sys
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from datetime import datetime, timedelta
import asyncio
import nest_asyncio
nest_asyncio.apply()
from googletrans import Translator
import warnings
warnings.filterwarnings("ignore", message="If `index_col` is not specified*")

import pyspark.pandas as ps
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import StringType
from pyspark.sql.window import Window
from sklearn.linear_model import LinearRegression

spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.session.timeZone", "UTC")
from pyspark.sql.functions import *

# ==============================================================================
# CONFIGURATION (codé en dur)
# ==============================================================================

BASE_DIR = os.path.abspath("BDD_BGES/BDD_BGES")

SITES = ["BERLIN", "LONDON", "LOSANGELES", "NEWYORK", "PARIS", "SHANGHAI"]

IMPACT_PATH = os.path.join(BASE_DIR, "materiel_informatique_impact.csv")


# Dictionnaires de traductions (à compléter)
TRANSLATIONS_MATRIX = {"Ökonom":"Economiste",
    "Führungskraft":"Cadre",
    "Personalleiter":"DRH",
    "Computeringenieur":"Ingénieur Informaticien",
    "Dateningenieur":"Ingénieur Data",
    "Economist":"Economiste",
    "Business Executive":"Cadre",
    "HRD":"DRH",
    "Computer Engineer":"Ingénieur Informaticien",
    "Data Engineer" :"Ingénieur Data",
    "Geschäftstreffen": "Rencontre entreprises",
    "Konferenz": "Conférence",
    "Schulung": "Formation",
    "Meeting": "Réunion",
    "Entwicklung": "Développement",
    "Conference":"Conférence",
    "Vocational Training":"Formation",
    "Team Meeting":"Réunion",
    "Business Meeting":"Rencontre entreprises",
    "Development":"Développement"
}

#Map pour la traduction
dict_expr = create_map([lit(x) for k, v in TRANSLATIONS_MATRIX.items() for x in (k, v)])
df_traductions = ps.DataFrame(
    list(TRANSLATIONS_MATRIX.items()), 
    columns=["MOT_ORIGINE", "MOT_TRADUIT"]
)
LANGUE_CIBLE = "fr"

def get_site_from_path(col_path):
    """Extrait le nom du site à partir du chemin du fichier"""
    return regexp_extract(col_path, r'PERSONNEL_([A-Z]+)\.txt|BDD_BGES_([A-Z]+)', 1)


def init_constellation():
    print("--- Initialisation du Schéma Constellation ---")
    
    # 1A. Référentiel IMPACT
    psdf_impact = ps.read_csv(IMPACT_PATH)
    psdf_impact.columns = [str(c).strip().upper().replace("È", "E") for c in psdf_impact.columns]
    
    # 1B. Référentiel PERSONNEL
    personnel_path = f"file://{BASE_DIR}/BDD_BGES_*/PERSONNEL_*.txt"
    try:
        psdf_pers = ps.read_csv(personnel_path, sep=";")
        psdf_pers = psdf_pers.drop_duplicates()

        psdf_pers = psdf_pers.merge(df_traductions, left_on='FONCTION_PERSONNEL', right_on='MOT_ORIGINE', how='left')
        # On remplace par la traduction si elle existe, sinon on garde le mot d'origine
        psdf_pers['FONCTION_PERSONNEL'] = psdf_pers['MOT_TRADUIT'].fillna(psdf_pers['FONCTION_PERSONNEL'])
        
        # --- SOLUTION INFAILLIBLE POUR LE SITE ---
        # Au lieu de .split(), on utilise une regex (extract) qui est nativement supportée
        # (Capture tout ce qui se trouve entre "PERS_" et "_")
        psdf_pers["SITE"] = psdf_pers["ID_PERSONNEL"].str.extract(r'PERS_(.*?)_')

        # Création de la dimension
        dim_personnel = psdf_pers[['ID_PERSONNEL', 'FONCTION_PERSONNEL', 'DT_NAISS', 'SITE']].drop_duplicates()
        
    except Exception as e:
        print(f"Attention : aucun fichier personnel trouvé ({e}). Tables initialisées vides.")
        dim_personnel = ps.DataFrame(columns=['ID_PERSONNEL', 'FONCTION_PERSONNEL', 'DT_NAISS', 'SITE'])
        
    # Initialisation de la structure de la constellation
    schema_initial = {
        "DF_IMPACT": psdf_impact, # Stocké ici, sera utilisé avec un simple .merge() dans l'ETL quotidien, sans broadcast
        "DIM_PERSONNEL": dim_personnel,
        "DIM_MISSION": ps.DataFrame(columns=["ID_MISSION", "TYPE_MISSION", "VILLE_DEPART", "PAYS_DEPART", "VILLE_DESTINATION", "PAYS_DESTINATION", "TRANSPORT", "ALLER_RETOUR"]),
        "DIM_MATERIEL": ps.DataFrame(columns=["ID_MATERIEL", "TYPE", "MODELE"]),
        "FAIT_MISSION": ps.DataFrame(columns=["SITE", "ID_PERSONNEL", "ID_MISSION", "ID_DATE_MISSION"]),
        "FAIT_MATERIEL": ps.DataFrame(columns=["SITE", "ID_PERSONNEL", "ID_MATERIEL", "ID_DATE_ACHAT"])
    }
    print("[OK] Schéma de constellation initialisé.")
    return schema_initial


def etl_daily_missions(date_str, schema_existant):
    print(f"\n--- Lancement ETL Transactionnel pour le : {date_str} ---")
    
    # Récupération du référentiel impact depuis le schéma reçu en paramètre
    psdf_impact = schema_existant["DF_IMPACT"]
    
    # --- EXTRACT ---
    mission_path = f"file://{BASE_DIR}/BDD_BGES_*/BDD_BGES_*_MISSION/MISSION_{date_str}.txt"
    it_path = f"file://{BASE_DIR}/BDD_BGES_*/BDD_BGES_*_INFORMATIQUE/MATERIEL_INFORMATIQUE_{date_str}.txt"
    mission_ok = False
    it_ok = False

    try:
        psdf_mission = ps.read_csv(mission_path, sep=";")
        mission_ok = True
    except:
        psdf_mission = ps.DataFrame()

    try:
        psdf_it = ps.read_csv(it_path, sep=";")
        it_ok = True
    except:
        psdf_it = ps.DataFrame()

    # --- TRANSFORM ---
    try:
        if mission_ok :
            psdf_mission = psdf_mission.drop_duplicates()
            psdf_mission['TYPE_MISSION'] = psdf_mission['TYPE_MISSION'].replace(TRANSLATIONS_MATRIX)
            psdf_mission['TRANSPORT'] = psdf_mission['TRANSPORT'].replace(TRANSLATIONS_MATRIX)
            psdf_mission.apply(handle_missing_values)
        if it_ok : 
            psdf_it = psdf_it.drop_duplicates()
            psdf_it = psdf_it.merge(psdf_impact, on=['TYPE', 'MODELE'], how="left")
            psdf_it.apply(handle_missing_values)
    except:
        print("Problemo commando")
        pass

    # --- Loader ---

    # Mettre à jour DIM_MISSION et FAIT_MISSION
    if not psdf_mission.empty:
        dim_mission_jour = psdf_mission[
            ["ID_MISSION", "TYPE_MISSION", "VILLE_DEPART", "PAYS_DEPART", 
             "VILLE_DESTINATION", "PAYS_DESTINATION", "TRANSPORT", "ALLER_RETOUR"]
        ].drop_duplicates()
        schema_existant["DIM_MISSION"] = ps.concat([schema_existant["DIM_MISSION"], dim_mission_jour]).drop_duplicates(["ID_MISSION"])
        
        fait_mission_jour = psdf_mission[["ID_PERSONNEL", "ID_MISSION", "DATE_MISSION"]].drop_duplicates()
        fait_mission_jour = fait_mission_jour.rename(columns={"DATE_MISSION": "ID_DATE_MISSION"})
        fait_mission_jour["SITE"] =  fait_mission_jour["ID_PERSONNEL"].str.split("_")[1]

        schema_existant["FAIT_MISSION"] = ps.concat([schema_existant["FAIT_MISSION"], fait_mission_jour]).drop_duplicates(["ID_PERSONNEL", "ID_MISSION"])

    # 3. Mettre à jour DIM_MATERIEL et FAIT_MATERIEL
    if not psdf_it.empty:
        dim_mat_jour = psdf_it[["ID_MATERIELINFO", "TYPE", "MODELE"]].drop_duplicates().rename(columns={"ID_MATERIELINFO": "ID_MATERIEL"})
        schema_existant["DIM_MATERIEL"] = ps.concat([schema_existant["DIM_MATERIEL"], dim_mat_jour]).drop_duplicates(["ID_MATERIEL"])
        
        fait_mat_jour = psdf_it[["ID_PERSONNEL", "ID_MATERIELINFO", "DATE_ACHAT"]].drop_duplicates().rename(columns={
            "ID_MATERIELINFO": "ID_MATERIEL", 
            "DATE_ACHAT": "ID_DATE_ACHAT"
        })
        fait_mat_jour["SITE"] =  fait_mat_jour["ID_PERSONNEL"].str.split("_")[1]
        schema_existant["FAIT_MATERIEL"] = ps.concat([schema_existant["FAIT_MATERIEL"], fait_mat_jour]).drop_duplicates(["ID_PERSONNEL", "ID_MATERIEL"])

    print(f"[SUCCES] Données du jour fusionnées. Le schéma global mis à jour a été renvoyé.")
    return schema_existant



def main():

    schema= init_constellation()


    current_date = datetime(2026, 4, 29)
    end_date = datetime(2026, 11, 5)
    delta = timedelta(days=1)
    date_str = current_date.strftime("%Y%m%d")

    while current_date <= end_date:
      print(f"\nLancement du processus ETL pour le jour : {date_str}")

      schema = etl_daily_missions(date_str=date_str, schema_existant = schema)

      schema["FAIT_MISSION"].show(5)
      schema["FAIT_MATERIEL"].show(5)

      # Poser une question?
      rep = str(input("Souhaitez-vous poser une question? Si oui, laquelle? (Sinon, 0 pour continuer, q pour quitter)"))
      if rep == "q":
          break
      elif rep !=0:
          ask_questions(rep)
      print("\nOn passe au jour suivant!")

      current_date += delta

    print("Processus terminé avec succès.")


if __name__ == "__main__":

    main()



def _filter_columns(df: ps.DataFrame, cols: list) -> ps.DataFrame:
    """Supprime toutes les colonnes qui ne sont pas dans 'cols'."""
    # En PySpark, on manipule directement la liste des colonnes pour le formatage
    df.columns = [str(c).strip() for c in df.columns]
    cols = [str(c).strip() for c in cols]
    
    existing = [c for c in cols if c in df.columns]
    missing  = [c for c in cols if c not in df.columns]
    if len(missing)>0:
        print(f"    [WARN] Colonnes attendues mais absentes : {missing}")
    return df[existing].copy()


def _append_unique(df_existing: ps.DataFrame, df_new: ps.DataFrame, pk: str) -> ps.DataFrame:
    """Ajoute df_new dans df_existing en dédupliquant sur la clé primaire 'pk'."""
    if df_new.empty:
        return df_existing
    combined = ps.concat([df_existing, df_new], ignore_index=True)
    return combined.drop_duplicates(subset=[pk], keep="last")


def traduire_texte(texte, target=LANGUE_CIBLE):
    """Fonction isolée qui traite un seul texte."""

    if not texte or texte != texte: # texte != texte vérifie les NaN
        return texte
    
    #translator = Translator()
        
    try:
        if texte in TRANSLATIONS_MATRIX.keys():
            return TRANSLATIONS_MATRIX[texte]
        
        #detection = asyncio.run(translator.detect(texte))
        #code_detecte = detection.lang 
        
        #if code_detecte == target:
            #return texte
        else:

            #resultat = asyncio.run(translator.translate(text= texte, dest=target))
            #return resultat.text 
            return texte
    except Exception as e:
        print(e)
        return texte 

def normalize_data(df: ps.DataFrame, cols: list[str]) -> ps.DataFrame:
    """Normalise la colonne entière: Traduit toutes les valeurs en LANGUE_CIBLE"""
    for col in cols:
        df[col] = df[col].apply(traduire_texte)
    return df


def standardize_timezone(df: ps.DataFrame, column: str) -> ps.DataFrame:
    """Convertit une colonne de dates en datetime standardisé."""
    if column in df.columns:
        # 1. On passe du monde "Pandas-on-Spark" au monde "Spark Natif"
        spark_df = df.to_spark()
        
        # 2. On utilise la fonction de conversion native de Spark (infaillible)
        spark_df = spark_df.withColumn(column, to_timestamp(col(column)))
        
        # 3. On revient dans le monde "Pandas-on-Spark"
        return spark_df.pandas_api()
        
    return df


def handle_missing_values(df, strategy="mean", target_col=None, feature_cols=None):
    """Complète les infos manquantes par moyenne ou régression linéaire"""
    if df.empty:
        return df

    if strategy == "mean" and target_col:
        df[target_col] = df[target_col].fillna(df[target_col].mean())

    elif strategy == "regression" and target_col and feature_cols:
        # Isolation des données incomplètes
        train_data = df.dropna(subset=feature_cols + [target_col])
        missing_data = df[df[target_col].isnull() & df[feature_cols].notnull().all(axis=1)]

        if not missing_data.empty and not train_data.empty:
            X_train = train_data[feature_cols]
            y_train = train_data[target_col]
            X_missing = missing_data[feature_cols]

            model = LinearRegression()
            model.fit(X_train, y_train)
            df.loc[missing_data.index, target_col] = model.predict(X_missing)

    return df

def ask_questions(rep = int):
    pass