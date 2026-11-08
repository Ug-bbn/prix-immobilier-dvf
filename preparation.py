"""Nettoie les ventes DVF 2021-2025, agrège les prix au m² par commune et entraîne le modèle d'estimation.

Usage : python preparation.py      (environ 500 Mo à télécharger la première fois)
Produit dans data/ : prix_communes.csv, communes.csv, rapport_nettoyage.csv, modele.joblib, performance.json
"""
import json
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

DATA = Path(__file__).parent / "data"
RAW = DATA / "raw"
ANNEES = range(2021, 2026)
URL = "https://files.data.gouv.fr/geo-dvf/latest/csv/{annee}/full.csv.gz"

LOGEMENTS = {"Maison", "Appartement"}
SURFACE_MIN = 9                  # surface minimale d'un logement décent (décret de 2002)
PRIX_M2_MIN, PRIX_M2_MAX = 300, 25_000
VENTES_MIN_PAR_COMMUNE = 5       # en dessous, une médiane ne veut pas dire grand-chose
FEATURES = ["longitude", "latitude", "surface", "pieces", "maison", "annee"]

COLONNES = {
    "id_mutation": str, "date_mutation": str, "nature_mutation": "category", "valeur_fonciere": float,
    "code_commune": str, "nom_commune": str, "code_departement": str, "id_parcelle": str, "lot1_numero": str,
    "type_local": "category", "surface_reelle_bati": float, "nombre_pieces_principales": float,
    "longitude": float, "latitude": float,
}


def telecharger(annee: int) -> Path:
    cible = RAW / f"dvf_{annee}.csv.gz"
    if not cible.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        partiel = cible.with_suffix(".part")
        urllib.request.urlretrieve(URL.format(annee=annee), partiel)
        partiel.replace(cible)
    return cible


def nettoyer(lignes: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Lignes DVF brutes -> une ligne par vente d'un logement unique, avec le nombre de ventes restant à chaque étape."""
    etapes = []

    def etape(nom, df):
        etapes.append((nom, df["id_mutation"].nunique()))
        return df

    df = etape("Mutations publiées", lignes)
    df = etape("Ventes classiques (hors VEFA, échanges, adjudications…)", df[df["nature_mutation"] == "Vente"])

    # un même prix ne doit couvrir qu'une transaction : plusieurs prix différents = plusieurs dispositions mêlées
    nb_prix = df.groupby("id_mutation")["valeur_fonciere"].nunique()
    df = etape("Un seul prix par mutation", df[df["id_mutation"].isin(nb_prix.index[nb_prix == 1])])

    # chaque logement est répété une fois par nature de culture du terrain : on dédoublonne
    locaux = df[df["type_local"].notna()].drop_duplicates(
        ["id_mutation", "type_local", "surface_reelle_bati", "nombre_pieces_principales", "id_parcelle", "lot1_numero"])
    logements = locaux[locaux["type_local"].isin(LOGEMENTS)]
    nb_logements = logements.groupby("id_mutation").size()
    avec_commerce = set(locaux.loc[locaux["type_local"].str.startswith("Local", na=False), "id_mutation"])
    uniques = set(nb_logements.index[nb_logements == 1]) - avec_commerce
    ventes = etape("Un seul logement, sans local commercial (dépendances admises)",
                   logements[logements["id_mutation"].isin(uniques)])

    ventes = ventes.rename(columns={"surface_reelle_bati": "surface", "nombre_pieces_principales": "pieces",
                                    "valeur_fonciere": "prix", "code_departement": "departement"})
    ventes = etape("Surface ≥ 9 m² et au moins une pièce",
                   ventes[(ventes["surface"] >= SURFACE_MIN) & (ventes["pieces"] >= 1)])
    ventes = etape("Coordonnées GPS connues", ventes.dropna(subset=["longitude", "latitude"]))

    ventes = ventes.assign(prix_m2=ventes["prix"] / ventes["surface"])
    ventes = etape(f"Prix au m² entre {PRIX_M2_MIN} et {PRIX_M2_MAX:,} €".replace(",", " "),
                   ventes[ventes["prix_m2"].between(PRIX_M2_MIN, PRIX_M2_MAX)])

    ventes = ventes.assign(annee=ventes["date_mutation"].str[:4].astype(int),
                           type=ventes["type_local"].astype(str))
    colonnes = ["annee", "code_commune", "nom_commune", "departement", "type", "surface", "pieces",
                "prix", "prix_m2", "longitude", "latitude"]
    return ventes[colonnes].reset_index(drop=True), etapes


def agreger(ventes: pd.DataFrame) -> pd.DataFrame:
    g = ventes.groupby(["annee", "code_commune", "type"]).agg(
        nom_commune=("nom_commune", "first"), departement=("departement", "first"),
        nb_ventes=("prix_m2", "size"), prix_m2_median=("prix_m2", "median")).reset_index()
    g = g[g["nb_ventes"] >= VENTES_MIN_PAR_COMMUNE]
    return g.assign(prix_m2_median=g["prix_m2_median"].round())


def communes(ventes: pd.DataFrame) -> pd.DataFrame:
    """Position de chaque commune : médiane des coordonnées de ses ventes."""
    return ventes.groupby("code_commune").agg(
        nom_commune=("nom_commune", "first"), departement=("departement", "first"),
        longitude=("longitude", "median"), latitude=("latitude", "median")).round(5).reset_index()


def entrainer(ventes: pd.DataFrame, graine: int = 0) -> tuple[dict, dict]:
    """Trois modèles sur le log du prix au m² : médiane, 10e et 90e centiles (fourchette d'estimation)."""
    X = ventes.assign(maison=(ventes["type"] == "Maison").astype(int))[FEATURES]
    y = np.log(ventes["prix_m2"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=graine)
    modeles = {}
    for nom, alpha in [("bas", 0.1), ("median", 0.5), ("haut", 0.9)]:
        modeles[nom] = HistGradientBoostingRegressor(
            # régularisé : sans cela, la fourchette colle aux données d'entraînement et devient trop étroite
            loss="quantile", quantile=alpha, max_iter=300, max_leaf_nodes=63, min_samples_leaf=100,
            early_stopping=True, random_state=graine,
        ).fit(X_train, y_train)

    reel, estime = np.exp(y_test), np.exp(modeles["median"].predict(X_test))
    ecart = np.abs(estime - reel) / reel
    dans_fourchette = (np.exp(modeles["bas"].predict(X_test)) <= reel) & (reel <= np.exp(modeles["haut"].predict(X_test)))
    perf = {
        "ventes_test": int(len(y_test)),
        "erreur_mediane_pct": round(float(np.median(ecart)) * 100, 1),
        "part_a_moins_de_20_pct": round(float((ecart < 0.2).mean()) * 100, 1),
        "couverture_fourchette_pct": round(float(dans_fourchette.mean()) * 100, 1),
    }
    return modeles, perf


def main() -> None:
    toutes, rapport = [], []
    for annee in ANNEES:
        lignes = pd.read_csv(telecharger(annee), usecols=list(COLONNES), dtype=COLONNES)
        ventes, etapes = nettoyer(lignes)
        del lignes
        rapport += [{"annee": annee, "etape": nom, "mutations": n} for nom, n in etapes]
        toutes.append(ventes)
        print(f"{annee} : {etapes[0][1]:>9} mutations -> {len(ventes):>8} ventes retenues")
    ventes = pd.concat(toutes, ignore_index=True)

    agreger(ventes).to_csv(DATA / "prix_communes.csv", index=False)
    communes(ventes).to_csv(DATA / "communes.csv", index=False)
    pd.DataFrame(rapport).to_csv(DATA / "rapport_nettoyage.csv", index=False)

    # shortcut: entraînement sur un échantillon de 1,5 million de ventes pour tenir en mémoire ; à élargir sur une machine plus grosse
    echantillon = ventes.sample(min(len(ventes), 1_500_000), random_state=0)
    modeles, perf = entrainer(echantillon)
    joblib.dump(modeles, DATA / "modele.joblib", compress=3)
    (DATA / "performance.json").write_text(json.dumps(perf, indent=2), encoding="utf-8")
    print("Modèle :", perf)


if __name__ == "__main__":
    main()
