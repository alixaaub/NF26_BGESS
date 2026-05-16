import os
import shutil
import subprocess
import sys

os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

# Spark 3.5 : Java 17 requis (Java 21+ → UnsupportedOperationException: getSubject)
if not os.environ.get("JAVA_HOME"):
    try:
        os.environ["JAVA_HOME"] = subprocess.check_output(
            ["/usr/libexec/java_home", "-v", "17"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

def _ensure_distutils_for_spark():
    """PySpark 3.5 + Python 3.12 : distutils requis pour les workers Spark."""
    import site
    import tempfile

    for site_dir in site.getsitepackages():
        if not os.path.isdir(site_dir):
            continue
        for name in os.listdir(site_dir):
            if name.endswith(".pth"):
                subprocess.run(
                    ["chflags", "nohidden", os.path.join(site_dir, name)],
                    stderr=subprocess.DEVNULL,
                    check=False,
                )

    import _distutils_hack

    _distutils_hack.add_shim()

    # Workers : « python -m pyspark.worker » sans site-packages → wrapper temporaire
    wrapper = os.path.join(tempfile.gettempdir(), "nf26_pyspark_python.sh")
    python = sys.executable.replace("\\", "\\\\").replace('"', '\\"')
    bootstrap = (
        "import runpy, sys\n"
        "import _distutils_hack\n"
        "_distutils_hack.add_shim()\n"
        "i = sys.argv.index('-m')\n"
        "sys.argv = sys.argv[i:]\n"
        "runpy.run_module(sys.argv[1], run_name='__main__', alter_sys=True)\n"
    )
    with open(wrapper, "w", encoding="utf-8") as f:
        f.write(f'#!/bin/bash\nexec "{python}" -c "\n{bootstrap}" "$@"\n')
    os.chmod(wrapper, 0o755)
    os.environ["PYSPARK_PYTHON"] = wrapper
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


_ensure_distutils_for_spark()

# Avant création de la session : désactive les barres [Stage …] dans le terminal
os.environ.setdefault("SPARK_UI_SHOWCONSOLEPROGRESS", "false")

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

spark = (
    SparkSession.builder
    .config("spark.ui.showConsoleProgress", "false")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.default.parallelism", "8")
    .getOrCreate()
)
spark.conf.set("spark.sql.session.timeZone", "UTC")
from pyspark.sql.functions import *

# ==============================================================================
# CONFIGURATION (codé en dur)
# ==============================================================================

BASE_DIR = os.path.join(os.path.dirname("BDD_BGES"), "BDD_BGES/BDD_BGES")
EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export", "warehouse")

SITES = ["BERLIN", "LONDON", "LOSANGELES", "NEWYORK", "PARIS", "SHANGHAI"]

# Perf : DIM_PERSONNEL ne change pas après init (évite 20k lignes × N exports/jour)
EXPORT_SKIP_PERSONNEL_AFTER_INIT = True

IMPACT_PATH = os.path.join(BASE_DIR, "materiel_informatique_impact.csv")
DISTANCE_REF_PATH = os.path.join(BASE_DIR, "referentiel_distance_ville.csv")

DISTANCE_KEYS = [
    "VILLE_DEPART", "PAYS_DEPART", "VILLE_DESTINATION", "PAYS_DESTINATION",
]
DISTANCE_MEME_VILLE_KM = 8.0  # km estimés pour un déplacement dans la même ville

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CO2_TRANSPORTS_PATH = os.path.join(PROJECT_ROOT, "CO2_transports.tsv")
CO2_VOITURE_PATH = os.path.join(PROJECT_ROOT, "CO2_voiture.tsv")

# Colonnes calculées missions (Alix / BGES-12-05)
COLS_MISSION_CALC = ["DISTANCE_KM", "CO2EQ", "IMPACT"]

# TYPE absent + « modèle par défaut » dans les données (cf. PC fixe sans ecran du référentiel)
IMPACT_SANS_TYPE_MODELE_DEFAUT = "350"  # str : même type que le référentiel (dtype=str)

MAP_TRANSPORT_CO2 = {
    "avion": "AVION",
    "plane": "AVION",
    "airplane": "AVION",
    "train": "TRAIN",
    "taxi": "TAXI",
    "transports en commun": "TC",
    "public transport": "TC",
}

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
# INITIALISATION DU SCHEMA (Format constellation)
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
        'VILLE_DESTINATION', 'PAYS_DESTINATION', 'TRANSPORT', 'ALLER_RETOUR',
        'DISTANCE_KM', 'CO2EQ', 'IMPACT',
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


def standardize_timezone(df, column: str):
    """Convertit une colonne de dates en datetime UTC (pandas, sans aller-retour Spark)."""
    if column not in df.columns or (hasattr(df, "empty") and df.empty):
        return df
    etait_spark = hasattr(df, "to_pandas")
    pdf = _en_pandas(df)
    # UTC puis naïf : pyspark.pandas ne supporte pas datetime64[ns, UTC]
    pdf[column] = pd.to_datetime(pdf[column], utc=True, errors="coerce").dt.tz_localize(None)
    return _en_pyspark_pandas(pdf, etait_spark)


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

def get_site(string: str) -> str:
    return string.split("_")[1]


def _ajouter_id_site(df):
    """ID_SITE = 2e segment de ID_PERSONNEL (KeyPers_Berlin_… → Berlin)."""
    etait_spark = hasattr(df, "to_pandas")
    pdf = _en_pandas(df)
    pdf["ID_SITE"] = pdf["ID_PERSONNEL"].astype(str).str.split("_").str[1]
    return _en_pyspark_pandas(pdf, etait_spark)


def _en_pandas(df):
    """pyspark.pandas → pandas (sinon copie pandas)."""
    if isinstance(df, pd.DataFrame):
        return df.copy()
    return df.to_pandas()


def _en_pyspark_pandas(df, etait_spark: bool):
    """pandas → pyspark.pandas si l'entrée venait de Spark."""
    if etait_spark:
        return ps.from_pandas(df)
    return df


def _charger_referentiel_impact():
    """
    Découpe materiel_informatique_impact.csv :
    - impact_par_type : une ligne « modèle par défaut » par TYPE (PC portable → 260, etc.)
    - impact_par_modele : tous les modèles précis (hors lignes modèle par défaut)
    """
    ref = ps.read_csv(IMPACT_PATH, dtype=str)
    ref.columns = [str(c).strip().upper().replace("È", "E") for c in ref.columns]
    ref["MODELE"] = ref["MODELE"].str.strip()
    ref["TYPE"] = ref["TYPE"].str.strip()

    modele_defaut = (
        ref["MODELE"]
        .str.lower()
        .str.replace("è", "e", regex=False)
        .str.replace("é", "e", regex=False)
        == "modele par defaut"
    )
    impact_par_type = ref.loc[modele_defaut, ["TYPE", "IMPACT"]].drop_duplicates(
        subset=["TYPE"], keep="first"
    )
    impact_par_modele = ref.loc[~modele_defaut].drop_duplicates(subset=["MODELE"], keep="first")[
        ["MODELE", "IMPACT"]
    ]
    return impact_par_modele, impact_par_type


def _est_modele_par_defaut(modele) -> bool:
    if modele is None or (isinstance(modele, float) and pd.isna(modele)):
        return False
    norm = str(modele).strip().lower().replace("è", "e").replace("é", "e")
    return norm == "modele par defaut"


def _joindre_impact_materiel(df_it, impact_par_modele, impact_par_type):
    """
    - TYPE renseigné, modèle vide ou « modèle par défaut » → impact par TYPE (ligne défaut du CSV).
    - Modèle précis renseigné → impact par MODELE.
    - TYPE vide + « modèle par défaut » → 350 (règle jeu de données artificiel).
    """
    etait_spark = hasattr(df_it, "to_pandas")
    df_it = _en_pandas(df_it)
    ref_modele = _en_pandas(impact_par_modele)
    ref_type = _en_pandas(impact_par_type)

    df_it["MODELE"] = df_it["MODELE"].fillna("").astype(str).str.strip()
    df_it["TYPE"] = df_it["TYPE"].fillna("").astype(str).str.strip()

    par_type = (df_it["MODELE"] == "") | df_it["MODELE"].apply(_est_modele_par_defaut)

    df_it = df_it.merge(ref_modele, on="MODELE", how="left")
    df_it = df_it.rename(columns={"IMPACT": "IMPACT_MODELE"})
    df_it = df_it.merge(
        ref_type.rename(columns={"IMPACT": "IMPACT_TYPE"}),
        on="TYPE",
        how="left",
    )
    df_it["IMPACT"] = np.where(par_type, df_it["IMPACT_TYPE"], df_it["IMPACT_MODELE"])
    sans_type_defaut = (df_it["TYPE"] == "") & df_it["MODELE"].apply(_est_modele_par_defaut)
    df_it.loc[sans_type_defaut, "IMPACT"] = IMPACT_SANS_TYPE_MODELE_DEFAUT
    df_it["IMPACT"] = df_it["IMPACT"].astype(str).replace("nan", np.nan)
    df_it = df_it.drop(columns=["IMPACT_MODELE", "IMPACT_TYPE"])
    return _en_pyspark_pandas(df_it, etait_spark)


def _compter_impact_vides(df: ps.DataFrame) -> int:
    if df.empty or "IMPACT" not in df.columns:
        return 0
    return int(df["IMPACT"].isna().sum())


def _charger_referentiel_distance() -> pd.DataFrame:
    cols = DISTANCE_KEYS + ["DISTANCE_KM"]
    if not os.path.exists(DISTANCE_REF_PATH):
        return pd.DataFrame(columns=cols)
    ref = pd.read_csv(DISTANCE_REF_PATH, sep=";")
    ref.columns = [str(c).strip() for c in ref.columns]
    for col in DISTANCE_KEYS:
        ref[col] = ref[col].astype(str).str.strip()
    ref["DISTANCE_KM"] = pd.to_numeric(ref["DISTANCE_KM"], errors="coerce")
    return ref


_coords_cache = {}
_geolocator = None


def _get_geolocator():
    global _geolocator
    if _geolocator is None:
        from geopy.geocoders import Nominatim
        _geolocator = Nominatim(user_agent="nf26_bges_clc_bis")
    return _geolocator


def _calculer_distance_km(ville_dep, pays_dep, ville_dest, pays_dest):
    """Distance à vol d'oiseau (km) ; 8 km si même ville."""
    from geopy.distance import geodesic
    import time

    vd, pd_ = str(ville_dep).strip(), str(pays_dep).strip()
    va, pa = str(ville_dest).strip(), str(pays_dest).strip()
    if vd == va and pd_ == pa:
        return DISTANCE_MEME_VILLE_KM

    key1, key2 = (vd, pd_), (va, pa)
    if key1 not in _coords_cache:
        lieu = _get_geolocator().geocode(f"{vd}, {pd_}")
        time.sleep(1)
        if lieu is None:
            return None
        _coords_cache[key1] = (lieu.latitude, lieu.longitude)
    if key2 not in _coords_cache:
        lieu = _get_geolocator().geocode(f"{va}, {pa}")
        time.sleep(1)
        if lieu is None:
            return None
        _coords_cache[key2] = (lieu.latitude, lieu.longitude)

    return round(geodesic(_coords_cache[key1], _coords_cache[key2]).km, 2)


def _ajouter_au_referentiel_distance(ref: pd.DataFrame, nouvelles_lignes: list) -> pd.DataFrame:
    if not nouvelles_lignes:
        return ref
    ref = pd.concat([ref, pd.DataFrame(nouvelles_lignes)], ignore_index=True)
    ref = ref.drop_duplicates(subset=DISTANCE_KEYS, keep="last")
    ref.to_csv(DISTANCE_REF_PATH, sep=";", index=False)
    return ref


def _joindre_distance_mission(df_mission: ps.DataFrame, ref_distance: pd.DataFrame):
    """Jointure référentiel distances ; geopy si trajet absent."""
    if df_mission.empty:
        return df_mission, ref_distance

    etait_spark = hasattr(df_mission, "to_pandas")
    df = _en_pandas(df_mission)
    ref = ref_distance.copy()

    for col in DISTANCE_KEYS:
        df[col] = df[col].astype(str).str.strip()
        if not ref.empty:
            ref[col] = ref[col].astype(str).str.strip()

    df = df.merge(ref, on=DISTANCE_KEYS, how="left")

    manquant = df["DISTANCE_KM"].isna()
    if manquant.any():
        paires = df.loc[manquant, DISTANCE_KEYS].drop_duplicates()
        nouvelles = []
        for _, row in paires.iterrows():
            km = _calculer_distance_km(
                row["VILLE_DEPART"], row["PAYS_DEPART"],
                row["VILLE_DESTINATION"], row["PAYS_DESTINATION"],
            )
            if km is not None:
                nouvelles.append({
                    "VILLE_DEPART": row["VILLE_DEPART"],
                    "PAYS_DEPART": row["PAYS_DEPART"],
                    "VILLE_DESTINATION": row["VILLE_DESTINATION"],
                    "PAYS_DESTINATION": row["PAYS_DESTINATION"],
                    "DISTANCE_KM": km,
                })

        if nouvelles:
            ref_distance = _ajouter_au_referentiel_distance(ref_distance, nouvelles)
            print(f"    [OK] {len(nouvelles)} trajet(s) ajouté(s) au référentiel distance")
            df = df.drop(columns=["DISTANCE_KM"])
            ref = ref_distance.copy()
            for col in DISTANCE_KEYS:
                ref[col] = ref[col].astype(str).str.strip()
            df = df.merge(ref, on=DISTANCE_KEYS, how="left")

    reste = int(df["DISTANCE_KM"].isna().sum())
    if reste:
        print(f"    [WARN] {reste} mission(s) sans DISTANCE_KM")

    return _en_pyspark_pandas(df, etait_spark), ref_distance


_FACTEURS_CO2 = None


def _facteurs_co2() -> dict:
    """Facteurs ADEME kg eCO2/km (CO2_transports.tsv, CO2_voiture.tsv)."""
    global _FACTEURS_CO2
    if _FACTEURS_CO2 is not None:
        return _FACTEURS_CO2

    transports = pd.read_csv(CO2_TRANSPORTS_PATH, sep="\t")
    voitures = pd.read_csv(CO2_VOITURE_PATH, sep="\t")

    def facteur_transport(mot_cle):
        ligne = transports[transports["subsubcategory"].str.contains(mot_cle, na=False)]
        return float(ligne["total"].iloc[0])

    _FACTEURS_CO2 = {
        "avion_court": facteur_transport("Short haul"),
        "avion_moyen": facteur_transport("Medium haul"),
        "avion_long": facteur_transport("Long haul"),
        "train": facteur_transport("Train < 200"),
        "tgv": facteur_transport("TGV > 200"),
        "taxi": float(voitures[voitures["subcategory"] == "Car"]["total"].mean()),
        "tc": float(transports[transports["subcategory"] == "Bus"]["total"].mean()),
    }
    return _FACTEURS_CO2


def _co2eq_km(transport: str, distance_km) -> float | None:
    """Facteur kg eCO2/km selon transport et distance."""
    if pd.isna(distance_km):
        return None
    code = MAP_TRANSPORT_CO2.get(str(transport).strip().lower())
    if code is None:
        return None
    km = float(distance_km)
    f = _facteurs_co2()
    if code == "AVION":
        if km < 1000:
            return f["avion_court"]
        if km <= 3500:
            return f["avion_moyen"]
        return f["avion_long"]
    if code == "TRAIN":
        return f["train"] if km <= 200 else f["tgv"]
    if code == "TAXI":
        return f["taxi"]
    if code == "TC":
        return f["tc"]
    return None


def enrichir_impact_mission_co2(df_mission: ps.DataFrame) -> ps.DataFrame:
    """Ajoute CO2EQ (kg/km) et IMPACT (kg) = DISTANCE_KM × CO2EQ × (2 si aller-retour)."""
    if df_mission.empty:
        return df_mission

    etait_spark = hasattr(df_mission, "to_pandas")
    df = _en_pandas(df_mission)

    distance = pd.to_numeric(df["DISTANCE_KM"], errors="coerce")
    df["CO2EQ"] = [_co2eq_km(t, km) for t, km in zip(df["TRANSPORT"], distance)]

    ar = df["ALLER_RETOUR"].astype(str).str.lower().isin(["oui", "yes"])
    mult = np.where(ar, 2.0, 1.0)

    df["CO2EQ"] = pd.to_numeric(df["CO2EQ"], errors="coerce").round(6)
    df["IMPACT"] = (distance * df["CO2EQ"] * mult).round(4)

    return _en_pyspark_pandas(df, etait_spark)


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
        'DIM_MISSION':   ps.DataFrame(columns=[
            'ID_MISSION', 'TYPE_MISSION', 'VILLE_DEPART', 'PAYS_DEPART',
            'VILLE_DESTINATION', 'PAYS_DESTINATION', 'TRANSPORT', 'ALLER_RETOUR',
            'DISTANCE_KM', 'CO2EQ', 'IMPACT',
        ]),
    }
    
    # 2. Référentiel impact matériel (structure materiel_informatique_impact.csv)
    if os.path.exists(IMPACT_PATH):
        impact_par_modele, impact_par_type = _charger_referentiel_impact()
        print(
            f"[OK] Référentiel IMPACT chargé "
            f"({len(impact_par_modele)} modèles, {len(impact_par_type)} types avec défaut)"
        )
    else:
        print("[WARN] Fichier d'impact introuvable. Initialisation d'un référentiel vide.")
        impact_par_modele = ps.DataFrame(columns=["MODELE", "IMPACT"])
        impact_par_type = ps.DataFrame(columns=["TYPE", "IMPACT"])

    ref_distance = _charger_referentiel_distance()
    if not ref_distance.empty:
        print(f"[OK] Référentiel distance chargé ({len(ref_distance)} trajets).")
    else:
        print("[WARN] Référentiel distance vide ou absent.")

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
    return db_schema, impact_par_modele, impact_par_type, ref_distance


def etl(current_date, schema, impact_par_modele, impact_par_type, ref_distance):
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
            df_mission = pd.read_csv(mission_path, sep=";")
            df_mission = df_mission.drop_duplicates()
            df_mission = _filter_columns(df_mission, COLS_MISSION)
            df_mission = normalize_data(df_mission, ['TYPE_MISSION', 'TRANSPORT'])
            df_mission = standardize_timezone(df_mission, 'DATE_MISSION')
            df_mission, ref_distance = _joindre_distance_mission(df_mission, ref_distance)
            df_mission = enrichir_impact_mission_co2(df_mission)
            df_mission = df_mission.rename(columns={'DATE_MISSION': 'ID_DATE_MISSION'})
            df_mission = _ajouter_id_site(df_mission)

            missions_jour.append(ps.from_pandas(df_mission))
            print(f"    [OK] MISSION")

        # ── TRAITEMENT MATÉRIEL QUOTIDIEN ─────────────────────────────────────
        if os.path.exists(it_path):
            df_it = pd.read_csv(it_path, sep=";")
            df_it = df_it.drop_duplicates()
            df_it = _filter_columns(df_it, COLS_MATERIEL)
            df_it = standardize_timezone(df_it, 'DATE_ACHAT')
            df_it = df_it.rename(columns={'DATE_ACHAT': 'ID_DATE_ACHAT', 'ID_MATERIELINFO': 'ID_MATERIEL'})
            df_it = _ajouter_id_site(df_it)
            df_it = _joindre_impact_materiel(df_it, impact_par_modele, impact_par_type)

            info_jour.append(ps.from_pandas(df_it))
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

    print(f"\n{'─'*50}\nSchéma flocon incrémenté pour le jour donné.\n{'─'*50}\n")
    return schema, ref_distance


def export_schema_csv(
    schema: dict,
    label: str = "latest",
    skip_tables: set | None = None,
    copy_skipped_from: str | None = None,
) -> str:
    """Exporte toutes les tables du schéma en CSV (un fichier par table)."""
    skip_tables = skip_tables or set()
    out_dir = os.path.join(EXPORT_DIR, label)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[EXPORT CSV] → {out_dir}")
    if copy_skipped_from and skip_tables:
        src_dir = os.path.join(EXPORT_DIR, copy_skipped_from)
        for name in skip_tables:
            src = os.path.join(src_dir, f"{name}.csv")
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(out_dir, f"{name}.csv"))
    for name, df in schema.items():
        if name in skip_tables:
            continue
        dest = os.path.join(out_dir, f"{name}.csv")
        if len(df) == 0:
            pd.DataFrame(columns=df.columns).to_csv(dest, index=False)
            print(f"  {name}.csv (vide)")
            continue
        # Collecte côté driver (Spark SQL) — pas de worker pyspark.pandas
        df.to_spark().toPandas().to_csv(dest, index=False)
        print(f"  {name}.csv ({len(df)} lignes)")
    print()
    return out_dir


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    # Plage des fichiers MISSION_* / MATERIEL_INFORMATIQUE_* dans BDD_BGES (200 jours)
    current_date = datetime(2026, 4, 29)
    end_date = datetime(2026, 11, 14)
    delta = timedelta(days=1)

    schema, impact_par_modele, impact_par_type, ref_distance = initialiser_warehouse()
    export_schema_csv(schema, "init")

    skip_pers = {"DIM_PERSONNEL"} if EXPORT_SKIP_PERSONNEL_AFTER_INIT else set()

    while current_date <= end_date:
      print(f"\nLancement du processus ETL pour le jour : {current_date.strftime('%Y-%m-%d')}")
      schema, ref_distance = etl(
          current_date, schema, impact_par_modele, impact_par_type, ref_distance
      )
      day_label = current_date.strftime("%Y-%m-%d")
      export_schema_csv(schema, day_label, skip_tables=skip_pers, copy_skipped_from="init")
      export_schema_csv(schema, "latest", skip_tables=skip_pers, copy_skipped_from="init")

      current_date += delta

    export_schema_csv(schema, "final")
    print("Processus terminé avec succès.")
    print(f"CSV disponibles dans : {EXPORT_DIR}")

if __name__ == "__main__":
    main()