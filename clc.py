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

# Colonnes à conserver par type de fichier
COLS_PERSONNEL = [
    'ID_PERSONNEL', 'NOM_PERSONNEL', 'PRENOM_PERSONNEL',
    'NUM_VOIE', 'CMPL_VOIE', 'CD_POSTAL', 'VILLE', 'PAYS',
    'FONCTION_PERSONNEL', 'TS_CREATION_PERSONNEL'
]

COLS_MATERIEL = [
    'ID_MATERIELINFO', 'ID_PERSONNEL', 'NOM_PERSONNEL', 'PRENOM_PERSONNEL',
    'DATE_ACHAT', 'TYPE', 'MODELE'
]

COLS_MISSION = [
    'ID_MISSION', 'ID_PERSONNEL', 'NOM_PERSONNEL', 'PRENOM_PERSONNEL',
    'DATE_MISSION', 'TYPE_MISSION', 'VILLE_DEPART', 'PAYS_DEPART',
    'VILLE_DESTINATION', 'PAYS_DESTINATION', 'TRANSPORT', 'ALLER_RETOUR'
]

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

LANGUE_CIBLE = "fr"




# Initialisation des tables en tant que DataFrame PySpark Pandas

schema = {
    # Table de faits centrale
    'ALICIA_KEYS': ps.DataFrame(columns=[
        'ID_PERSONNEL', 'ID_MATERIELINFO', 'ID_MISSION'
    ]),
    # Dimensions
    'DF_PERSONNEL': ps.DataFrame(columns=COLS_PERSONNEL),
    'DF_MATERIEL':  ps.DataFrame(columns=COLS_MATERIEL + ['IMPACT']),
    'DF_MISSION':   ps.DataFrame(columns=COLS_MISSION),
}

# ==============================================================================
# HELPERS
# ==============================================================================

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


def etl(current_date):
    date_str = current_date.strftime("%Y%m%d")
    print(f"--- Lancement ETL pour le jour : {date_str} ---")

    # Chargement du référentiel IMPACT une seule fois
    impact_ref = ps.read_csv(IMPACT_PATH, dtype=str)
    impact_ref.columns = [str(c).strip().upper().replace("È", "E") for c in impact_ref.columns]
    print(f"[OK] Référentiel IMPACT chargé")

    # Dictionnaires pour stocker les données du transformer
    missions_jour  = []
    info_jour      = []
    personnel_jour = []

    for site in SITES:

        # ── EXTRACTOR ─────────────────────────────────────────────────────────

        mission_path   = os.path.join(BASE_DIR, f"BDD_BGES_{site}", f"BDD_BGES_{site}_MISSION",
                                      f"MISSION_{date_str}.txt")
        it_path        = os.path.join(BASE_DIR, f"BDD_BGES_{site}", f"BDD_BGES_{site}_INFORMATIQUE",
                                      f"MATERIEL_INFORMATIQUE_{date_str}.txt")
        personnel_path = os.path.join(BASE_DIR, f"BDD_BGES_{site}",
                                      f"PERSONNEL_{site}.txt")

        print(f"\n  [SITE : {site}]")

        # ── MISSION ───────────────────────────────────────────────────────────
        if os.path.exists(mission_path):
            df_mission = ps.read_csv(mission_path, sep=';')

            # --- TRANSFORMER ---
            #traduire aller_retour
            df_mission = df_mission.drop_duplicates()
            df_mission = _filter_columns(df_mission, COLS_MISSION)
            df_mission = normalize_data(df_mission, ['TYPE_MISSION', 'TRANSPORT'])
            df_mission = standardize_timezone(df_mission, 'DATE_MISSION')

            missions_jour.append(df_mission)
            print(f"    [OK] MISSION")
        else:
            print(f"    [SKIP] Pas de fichier MISSION pour ce jour")

        # ── MATÉRIEL INFORMATIQUE ─────────────────────────────────────────────
        if os.path.exists(it_path):
            df_it = ps.read_csv(it_path, sep=';')

            # --- TRANSFORMER ---

            #Traduire Type, modele
            df_it = df_it.drop_duplicates()
            df_it = _filter_columns(df_it, COLS_MATERIEL)
            df_it = standardize_timezone(df_it, 'DATE_ACHAT')

            # Jointure distribuée avec le référentiel IMPACT
            df_it = df_it.merge(impact_ref, on=['TYPE', 'MODELE'], how="left")
            
            # nb_sans_impact = df_it["IMPACT"].isna().sum() 
            # Note : En spark, un `.sum()` déclenche un Job (Action). Peut ralentir si fait à chaque boucle.
            
            info_jour.append(df_it)
            print(f"    [OK] MATERIEL INFORMATIQUE")
        else:
            print(f"    [SKIP] Pas de fichier MATERIEL pour ce jour")

        # ── PERSONNEL ─────────────────────────────────────────────────────────
        if os.path.exists(personnel_path):
            df_pers = ps.read_csv(personnel_path, sep=';')

            # --- TRANSFORMER ---
            #traduire fonction_personnel
            df_pers = df_pers.drop_duplicates()
            df_pers = _filter_columns(df_pers, COLS_PERSONNEL)
            df_pers = normalize_data(df_pers, ['FONCTION_PERSONNEL'])
            df_pers = standardize_timezone(df_pers, 'TS_CREATION_PERSONNEL')

            personnel_jour.append(df_pers)
            print(f"    [OK] PERSONNEL")
        else:
            print(f"    [SKIP] Pas de fichier PERSONNEL pour ce site")

    # ── LOAD — Chargement dans le schéma flocon ────────────────────────────────

    if missions_jour or info_jour or personnel_jour:

        # Concaténation de tous les sites du jour (Opération optimisée dans Spark)
        df_all_missions  = ps.concat(missions_jour,  ignore_index=True) if missions_jour  else ps.DataFrame(columns=COLS_MISSION)
        df_all_materiel  = ps.concat(info_jour,      ignore_index=True) if info_jour      else ps.DataFrame(columns=COLS_MATERIEL + ['IMPACT'])
        df_all_personnel = ps.concat(personnel_jour, ignore_index=True) if personnel_jour else ps.DataFrame(columns=COLS_PERSONNEL)

        df_all_missions.to_spark().show(5)
        df_all_materiel.to_spark().show(5)
        df_all_personnel.to_spark().show(5)

        if not df_all_missions.empty:
            schema["DF_MISSION"] = _append_unique(schema["DF_MISSION"], df_all_missions, pk="ID_MISSION")

        if not df_all_materiel.empty:
            schema["DF_MATERIEL"] = _append_unique(schema["DF_MATERIEL"], df_all_materiel, pk="ID_MATERIELINFO")

        if not df_all_personnel.empty:
            schema["DF_PERSONNEL"] = _append_unique(schema["DF_PERSONNEL"], df_all_personnel, pk="ID_PERSONNEL")

        # ==============================================================================
        # Table de faits ALICIA_KEYS — croisement des clés du jour
        # /!\ Les boucles for python imbriquées ont été remplacées par des "merge" (jointures) PySpark
        # ==============================================================================
        
        if not df_all_personnel.empty:
            # Extraction des clés uniques du personnel
            df_pers_keys = df_all_personnel[["ID_PERSONNEL"]].drop_duplicates()
            
            # Extraction des clés matériels existantes pour le jour
            if not df_all_materiel.empty:
                df_mat_keys = df_all_materiel[["ID_PERSONNEL", "ID_MATERIELINFO"]].drop_duplicates()
            else:
                df_mat_keys = ps.DataFrame(columns=["ID_PERSONNEL", "ID_MATERIELINFO"])
                
            # Extraction des clés missions existantes pour le jour
            if not df_all_missions.empty:
                df_mis_keys = df_all_missions[["ID_PERSONNEL", "ID_MISSION"]].drop_duplicates()
            else:
                df_mis_keys = ps.DataFrame(columns=["ID_PERSONNEL", "ID_MISSION"])
                
            # Jointure gauche (Le DataFrame Spark s'occupe du produit cartésien interne)
            df_facts = df_pers_keys.merge(df_mat_keys, on="ID_PERSONNEL", how="left")
            df_facts = df_facts.merge(df_mis_keys, on="ID_PERSONNEL", how="left")

            if not df_facts.empty:
                combined_facts = ps.concat([schema["ALICIA_KEYS"], df_facts], ignore_index=True)
                schema["ALICIA_KEYS"] = combined_facts.drop_duplicates(
                    subset=["ID_PERSONNEL", "ID_MATERIELINFO", "ID_MISSION"], 
                    keep="last"
                )
                # print(f"[LOAD] ALICIA_KEYS   ← {len(df_facts)} ligne(s) ajoutée(s) ce jour")
                df_facts.to_spark().show(5)
    # Résumé
    print(f"\n{'─'*50}")
    print("Schéma flocon à jour (Note : les `.len()` forcent une évaluation sous PySpark)")
    print(f"{'─'*50}\n")
    return

# ==============================================================================
# MAIN
# ==============================================================================


def main():

    current_date = datetime(2026, 4, 29)
    end_date = datetime(2026, 11, 5)
    delta = timedelta(days=1)

    while current_date <= end_date:
      print(f"\nLancement du processus ETL pour le jour : {current_date.strftime('%Y-%m-%d')}")
      # On appelle l'ETL pour le jour N
      etl(current_date)

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