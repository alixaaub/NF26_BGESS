#!/usr/bin/env python3
"""
Génère BDD_BGES/BDD_BGES/referentiel_distance_ville.csv à partir des fichiers MISSION_*.txt.

Distance = à vol d'oiseau (ligne droite entre 2 villes géocodées), en km.
Pas de distance routière, pas de détour avion.
Usage : python build_referentiel_distance_ville.py
"""

import glob
import os
import time
import pandas as pd
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

BASE_DIR = os.path.join("BDD_BGES", "BDD_BGES")
OUTPUT = os.path.join(BASE_DIR, "referentiel_distance_ville.csv")
MISSION_GLOB = os.path.join(BASE_DIR, "**", "MISSION_*.txt")
DISTANCE_MEME_VILLE_KM = 8.0  # déplacement estimé au sein d'une même ville


def paire(ville, pays):
    return (str(ville).strip(), str(pays).strip())


def charger_trajets() -> tuple[list[tuple[str, str]], list[tuple[tuple[str, str], tuple[str, str]]]]:
    paths = glob.glob(MISSION_GLOB, recursive=True)
    if not paths:
        raise FileNotFoundError(f"Aucun fichier trouvé : {MISSION_GLOB}")

    df = pd.concat(
        [
            pd.read_csv(
                p,
                sep=";",
                usecols=["VILLE_DEPART", "PAYS_DEPART", "VILLE_DESTINATION", "PAYS_DESTINATION"],
                dtype=str,
            )
            for p in paths
        ],
        ignore_index=True,
    )

    trajets = set()
    villes = set()
    for _, row in df.iterrows():
        dep = paire(row["VILLE_DEPART"], row["PAYS_DEPART"])
        arr = paire(row["VILLE_DESTINATION"], row["PAYS_DESTINATION"])
        villes.add(dep)
        villes.add(arr)
        trajets.add((dep, arr))
    return sorted(villes), sorted(trajets)


def geocoder_villes(villes: list[tuple[str, str]]) -> dict[tuple[str, str], tuple[float, float]]:
    geolocator = Nominatim(user_agent="nf26_bges_referentiel_distance")
    coords = {}
    for ville, pays in villes:
        lieu = geolocator.geocode(f"{ville}, {pays}")
        time.sleep(1)  # politique d'utilisation Nominatim
        if lieu is None:
            print(f"[WARN] Géocodage impossible : {ville}, {pays}")
            continue
        coords[(ville, pays)] = (lieu.latitude, lieu.longitude)
        print(f"[OK] {ville}, {pays} -> {lieu.latitude:.4f}, {lieu.longitude:.4f}")
    return coords


def main():
    villes, trajets = charger_trajets()
    print(f"{len(villes)} villes distinctes, {len(trajets)} trajets distincts")

    coords = geocoder_villes(villes)
    if len(coords) < len(villes):
        print(f"[WARN] {len(villes) - len(coords)} ville(s) non géocodée(s)")

    lignes = []
    for (vd, pays_d), (va, pays_a) in trajets:
        if (vd, pays_d) not in coords or (va, pays_a) not in coords:
            continue
        if vd == va and pays_d == pays_a:
            km = DISTANCE_MEME_VILLE_KM
        else:
            km = geodesic(coords[(vd, pays_d)], coords[(va, pays_a)]).km  # vol d'oiseau
        lignes.append(
            {
                "VILLE_DEPART": vd,
                "PAYS_DEPART": pays_d,
                "VILLE_DESTINATION": va,
                "PAYS_DESTINATION": pays_a,
                "DISTANCE_KM": round(km, 2),
            }
        )

    out = pd.DataFrame(lignes)
    out.to_csv(OUTPUT, sep=";", index=False)
    print(f"{len(out)} paires écrites dans {OUTPUT}")


if __name__ == "__main__":
    main()
