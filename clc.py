import os
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import asyncio
import nest_asyncio
nest_asyncio.apply()
from googletrans import Translator
import warnings
warnings.filterwarnings("ignore", message="If `index_col` is not specified*")

import pyspark.pandas as ps
from pyspark.sql import SparkSession
from pyspark.sql import Row
from numpy._core.multiarray import empty_like
from sklearn.linear_model import LinearRegression

spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.session.timeZone", "UTC")
from pyspark.sql.functions import *

# ==============================================================================
# CONFIGURATION (codé en dur)
# ==============================================================================

BASE_DIR = os.path.join(os.path.dirname("BDD_BGES"), "BDD_BGES/BDD_BGES")

SITES = ["BERLIN", "LONDON", "LOSANGELES", "NEWYORK", "PARIS", "SHANGHAI"]

IMPACT_PATH = os.path.join(BASE_DIR, "materiel_informatique_impact.csv")

# Colonnes à conserver par type de fichier (Adaptées au nouveau schéma)
COLS_PERSONNEL = [
    'ID_PERSONNEL', 'FONCTION_PERSONNEL', 'DT_NAISS', 'TS_CREATION_PERSONNEL'
]

COLS_MATERIEL = [
    'ID_MATERIELINFO', 'ID_PERSONNEL', 'DATE_ACHAT', 'TYPE', 'MODELE'
]

COLS_MISSION = [
    'ID_MISSION', 'ID_PERSONNEL', 'DATE_MISSION', 'TYPE_MISSION', 
    'VILLE_DEPART', 'PAYS_DEPART', 'VILLE_DESTINATION', 'PAYS_DESTINATION', 
    'TRANSPORT', 'ALLER_RETOUR'
]

# Dictionnaires de traductions 
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

LANGUE_CIBLE = "fr"

# ==============================================================================
# INITIALISATION DU SCHEMA (Adapté à la photo)
# ==============================================================================
schema = {
    # Tables de faits
    'FAIT_MISSION': ps.DataFrame(columns=[
        'ID_PERSONNEL', 'ID_MISSION', 'ID_SITE', 'ID_DATE_MISSION'
    ]),
    'FAIT_MATERIEL': ps.DataFrame(columns=[
        'ID_PERSONNEL', 'ID_MATERIEL', 'ID_SITE', 'ID_DATE_ACHAT'
    ]),
    # Dimensions
    'DIM_PERSONNEL': ps.DataFrame(columns=[
        'ID_PERSONNEL', 'FONCTION_PERSONNEL', 'ID_SITE', 'DT_NAISS'
    ]),
    'DIM_MATERIEL':  ps.DataFrame(columns=[
        'ID_MATERIEL', 'TYPE', 'MODELE', 'IMPACT'
    ]),
    'DIM_MISSION':   ps.DataFrame(columns=[
        'ID_MISSION', 'TYPE_MISSION', 'VILLE_DEPART', 'PAYS_DEPART', 
        'VILLE_DESTINATION', 'PAYS_DESTINATION', 'TRANSPORT', 'ALLER_RETOUR'
    ]),
}

# ==============================================================================
# HELPERS
# ==============================================================================

def _filter_columns(df: ps.DataFrame, cols: list) -> ps.DataFrame:
    """Supprime toutes les colonnes qui ne sont pas dans 'cols'."""
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
        
    if df_existing.empty:
        return df_new.drop_duplicates(subset=[pk], keep="last")
        
    combined = ps.concat([df_existing, df_new], ignore_index=True)
    return combined.drop_duplicates(subset=[pk], keep="last")


def traduire_texte(texte, target=LANGUE_CIBLE):
    """Fonction isolée qui traite un seul texte."""
    if not texte or texte != texte:
        return texte
        
    try:
        if texte in TRANSLATIONS_MATRIX.keys():
            return TRANSLATIONS_MATRIX[texte]
        else:
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
        spark_df = df.to_spark()
        spark_df = spark_df.withColumn(column, to_timestamp(col(column)))
        return spark_df.pandas_api()
    return df


def handle_missing_values(df, strategy="mean", target_col=None, feature_cols=None):
    """Complète les infos manquantes par moyenne ou régression linéaire"""
    if df.empty:
        return df

    if strategy == "mean" and target_col:
        df[target_col] = df[target_col].fillna(df[target_col].mean())

    elif strategy == "regression" and target_col and feature_cols:
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

def get_site(string :str) ->str:
    return string.split("_")[1]

def initialiser_warehouse():
    """
    Initialise la structure globale de la base de données (schéma en étoile),
    charge les données de personnel statiques de tous les sites,
    et met en cache le référentiel d'impact matériel.
    """
    print("\n" + "═"*60)
    print("  INITIALISATION DE LA DATA WAREHOUSE (DONNÉES STATIQUES)")
    print("═"*60)
    
    # 1. Structure initiale du schéma cible (conforme à la modélisation)
    db_schema = {
        # Tables de faits (alimentées chaque jour)
        'FAIT_MISSION': ps.DataFrame(columns=['ID_PERSONNEL', 'ID_MISSION', 'ID_SITE', 'ID_DATE_MISSION']),
        'FAIT_MATERIEL': ps.DataFrame(columns=['ID_PERSONNEL', 'ID_MATERIEL', 'ID_SITE', 'ID_DATE_ACHAT']),
        
        # Dimensions statiques / dynamiques
        'DIM_PERSONNEL': ps.DataFrame(columns=['ID_PERSONNEL', 'FONCTION_PERSONNEL', 'ID_SITE', 'AGE']),
        'DIM_MATERIEL':  ps.DataFrame(columns=['ID_MATERIEL', 'TYPE', 'MODELE', 'IMPACT']),
        'DIM_MISSION':   ps.DataFrame(columns=['ID_MISSION', 'TYPE_MISSION', 'VILLE_DEPART', 'PAYS_DEPART', 
                                               'VILLE_DESTINATION', 'PAYS_DESTINATION', 'TRANSPORT', 'ALLER_RETOUR']),
    }
    
    # 2. Chargement unique et nettoyage du référentiel d'impact matériel
    if os.path.exists(IMPACT_PATH):
        impact_ref = ps.read_csv(IMPACT_PATH, dtype=str)
        impact_ref.columns = [str(c).strip().upper().replace("È", "E") for c in impact_ref.columns]
        print("[OK] Référentiel d'impact matériel chargé en mémoire.")
    else:
        print("[WARN] Fichier d'impact introuvable. Initialisation d'un référentiel vide.")
        impact_ref = ps.DataFrame(columns=['TYPE', 'MODELE', 'IMPACT'])

    # 3. Chargement et consolidation unique du PERSONNEL pour TOUS les sites
    personnel_statique = []
    for site in SITES:
        personnel_path = os.path.join(BASE_DIR, f"BDD_BGES_{site}", f"PERSONNEL_{site}.txt")
        
        if os.path.exists(personnel_path):
            df_pers = ps.read_csv(personnel_path, sep=';')
            df_pers = df_pers.drop_duplicates()
            df_pers = _filter_columns(df_pers, COLS_PERSONNEL)
            df_pers = normalize_data(df_pers, ['FONCTION_PERSONNEL'])
            
            # Extraction du site optimisée nativement sous Spark Pandas (sans lambda)
            df_pers['ID_SITE'] = df_pers['ID_PERSONNEL'].apply(get_site)
            
            personnel_statique.append(df_pers)
            print(f"    [OK] Personnel extrait pour le site : {site}")
        else:
            print(f"    [SKIP] Aucun fichier personnel trouvé pour le site : {site}")
            
    if personnel_statique:
        df_all_personnel = ps.concat(personnel_statique, ignore_index=True)
        # On ne garde que les colonnes requises par la dimension cible
        dim_cols_personnel = [c for c in db_schema["DIM_PERSONNEL"].columns if c in df_all_personnel.columns]
        db_schema["DIM_PERSONNEL"] = df_all_personnel[dim_cols_personnel].drop_duplicates(subset=["ID_PERSONNEL"])
        print(f"[OK] Dimension DIM_PERSONNEL initialisée ({len(db_schema['DIM_PERSONNEL'])} individus insérés).")
    
    print("═"*60 + "\n")
    return db_schema, impact_ref


def etl(current_date, schema, impact_ref):
    date_str = current_date.strftime("%Y%m%d")
    print(f"--- Lancement ETL pour le jour : {date_str} ---")

    # Tableaux pour accumuler les données incrémentales du jour J
    missions_jour  = []
    info_jour      = []

    for site in SITES:
        # Fichiers quotidiens uniquement
        mission_path   = os.path.join(BASE_DIR, f"BDD_BGES_{site}", f"BDD_BGES_{site}_MISSION", f"MISSION_{date_str}.txt")
        it_path        = os.path.join(BASE_DIR, f"BDD_BGES_{site}", f"BDD_BGES_{site}_INFORMATIQUE", f"MATERIEL_INFORMATIQUE_{date_str}.txt")

        print(f"\n  [SITE : {site}]")

        # ── TRAITEMENT MISSION QUOTIDIENNE ────────────────────────────────────
        if os.path.exists(mission_path):
            df_mission = ps.read_csv(mission_path, sep=';')
            df_mission = df_mission.drop_duplicates()
            df_mission = _filter_columns(df_mission, COLS_MISSION)
            df_mission = normalize_data(df_mission, ['TYPE_MISSION', 'TRANSPORT'])
            df_mission = standardize_timezone(df_mission, 'DATE_MISSION')
            
            # Adaptation formatage
            df_mission = df_mission.rename(columns={'DATE_MISSION': 'ID_DATE_MISSION'})
            df_mission['ID_SITE'] = df_mission['ID_PERSONNEL'].apply(get_site)

            missions_jour.append(df_mission)
            print(f"    [OK] MISSION")

        # ── TRAITEMENT MATÉRIEL QUOTIDIEN ─────────────────────────────────────
        if os.path.exists(it_path):
            df_it = ps.read_csv(it_path, sep=';')
            df_it = df_it.drop_duplicates()
            df_it = _filter_columns(df_it, COLS_MATERIEL)
            df_it = standardize_timezone(df_it, 'DATE_ACHAT')

            # Adaptation formatage
            df_it = df_it.rename(columns={'DATE_ACHAT': 'ID_DATE_ACHAT', 'ID_MATERIELINFO': 'ID_MATERIEL'})
            df_it['ID_SITE'] = df_it['ID_PERSONNEL'].apply(get_site)

            # /!\ JOINTURE DIRECTE AVEC LE RÉFÉRENTIEL PASSÉ EN PARAMÈTRE
            df_it = df_it.merge(impact_ref, on=['TYPE', 'MODELE'], how="left")
            
            info_jour.append(df_it)
            print(f"    [OK] MATERIEL INFORMATIQUE")

    # ── PHASE DE CHARGEMENT INCRÉMENTAL (LOAD) ────────────────────────────────
    if missions_jour or info_jour:
        df_all_missions  = ps.concat(missions_jour,  ignore_index=True) if missions_jour  else ps.DataFrame()
        df_all_materiel  = ps.concat(info_jour,      ignore_index=True) if info_jour      else ps.DataFrame()

        # -- Alimentation incrémentale : MISSIONS --
        if not df_all_missions.empty:
            # Dimension (S'enrichit s'il y a de nouvelles missions)
            dim_cols_mission = [c for c in schema["DIM_MISSION"].columns if c in df_all_missions.columns]
            dim_mission = df_all_missions[dim_cols_mission].drop_duplicates()
            schema["DIM_MISSION"] = _append_unique(schema["DIM_MISSION"], dim_mission, pk="ID_MISSION")

            # Table de Faits (Ajout des liaisons du jour)
            fait_cols_mission = [c for c in schema["FAIT_MISSION"].columns if c in df_all_missions.columns]
            fait_mission = df_all_missions[fait_cols_mission].drop_duplicates()
            schema["FAIT_MISSION"] = ps.concat([schema["FAIT_MISSION"], fait_mission], ignore_index=True).drop_duplicates()


        
        # -- Alimentation incrémentale : MATERIEL --
        if not df_all_materiel.empty:
            # Dimension (Nouveaux matériels physiques enregistrés)
            dim_cols_materiel = [c for c in schema["DIM_MATERIEL"].columns if c in df_all_materiel.columns]
            dim_materiel = df_all_materiel[dim_cols_materiel].drop_duplicates()
            schema["DIM_MATERIEL"] = _append_unique(schema["DIM_MATERIEL"], dim_materiel, pk="ID_MATERIEL")

            # Table de Faits (Historisation des achats du jour)
            fait_cols_materiel = [c for c in schema["FAIT_MATERIEL"].columns if c in df_all_materiel.columns]
            fait_materiel = df_all_materiel[fait_cols_materiel].drop_duplicates()
            schema["FAIT_MATERIEL"] = ps.concat([schema["FAIT_MATERIEL"], fait_materiel], ignore_index=True).drop_duplicates()

        if schema["FAIT_MATERIEL"].empty:
            schema["FAIT_MATERIEL"] = fait_materiel
        else:
            schema["FAIT_MATERIEL"] = ps.concat([schema["FAIT_MATERIEL"], fait_materiel], ignore_index=True).drop_duplicates()
    
    
    print(f"\n{'─'*50}\nSchéma flocon incrémenté pour le jour donné.\n{'─'*50}\n")
    return schema
# ==============================================================================
# MAIN
# ==============================================================================

def main():
    current_date = datetime(2026, 4, 29)
    end_date = datetime(2026, 11, 5)
    delta = timedelta(days=1)

    schema, impact_ref = initialiser_warehouse()

    while current_date <= end_date:
      print(f"\nLancement du processus ETL pour le jour : {current_date.strftime('%Y-%m-%d')}")
      schema = etl(current_date, schema, impact_ref)

      rep = str(input("Souhaitez-vous poser une question? Si oui, laquelle? (Sinon, 0 pour continuer, q pour quitter)"))
      if rep == "q":
          
          break
      elif rep != "0":
          ask_questions(rep)
      print("\nOn passe au jour suivant!")

      current_date += delta

    print("Processus terminé avec succès.")

if __name__ == "__main__":
    main()