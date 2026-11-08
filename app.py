import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from preparation import FEATURES

DATA = Path(__file__).parent / "data"
DERNIERE_ANNEE = 2025

st.set_page_config(page_title="Prix de l'immobilier en France", page_icon="🏠", layout="wide")


def nb(x, dec=0):
    """Format français : espace pour les milliers, virgule décimale."""
    return f"{x:,.{dec}f}".replace(",", " ").replace(".", ",")


@st.cache_data
def charger():
    prix = pd.read_csv(DATA / "prix_communes.csv", dtype={"code_commune": str, "departement": str})
    communes = pd.read_csv(DATA / "communes.csv", dtype={"code_commune": str, "departement": str})
    rapport = pd.read_csv(DATA / "rapport_nettoyage.csv")
    perf = json.loads((DATA / "performance.json").read_text(encoding="utf-8"))
    communes["libelle"] = communes["nom_commune"] + " (" + communes["departement"] + ")"
    return prix, communes.sort_values("libelle"), rapport, perf


@st.cache_resource
def modele():
    return joblib.load(DATA / "modele.joblib")


def couleur(rang: float) -> list[int]:
    """Bleu (moins cher) -> rouge (plus cher), selon le rang du prix parmi les communes."""
    return [int(40 + 215 * rang), int(110 * (1 - abs(rang - 0.5) * 2) + 60), int(220 * (1 - rang) + 35), 170]


prix, communes, rapport, perf = charger()
st.title("Prix de l'immobilier en France")
st.caption("Ventes de maisons et d'appartements 2021-2025 (DVF, DGFiP / Etalab), nettoyées et analysées en Python.")

carte, evolution, estimation, qualite = st.tabs(
    ["🗺️ Carte des prix", "📈 Évolution d'une commune", "🧮 Estimer un bien", "✅ Qualité des données"])

with carte:
    c1, c2 = st.columns(2)
    type_bien = c1.radio("Type de bien", ["Appartement", "Maison"], horizontal=True)
    annee = c2.select_slider("Année", options=sorted(prix["annee"].unique()), value=DERNIERE_ANNEE)

    vue = prix[(prix["type"] == type_bien) & (prix["annee"] == annee)].merge(
        communes[["code_commune", "longitude", "latitude"]], on="code_commune")
    vue = vue.assign(
        couleur=vue["prix_m2_median"].rank(pct=True).map(couleur),
        rayon=np.sqrt(vue["nb_ventes"]) * 150,
        prix_txt=vue["prix_m2_median"].map(lambda v: f"{nb(v)} €/m²"),
    )
    st.pydeck_chart(pdk.Deck(
        layers=[pdk.Layer("ScatterplotLayer", vue, get_position=["longitude", "latitude"],
                          get_fill_color="couleur", get_radius="rayon", radius_min_pixels=2, pickable=True)],
        initial_view_state=pdk.ViewState(latitude=46.6, longitude=2.4, zoom=5),
        tooltip={"text": "{nom_commune} ({departement})\n{prix_txt} — {nb_ventes} ventes"},
        map_style=None,
    ), height=600)
    st.caption(f"{nb(len(vue))} communes avec au moins 5 ventes. Bleu : moins cher, rouge : plus cher. "
               "La taille du point suit le nombre de ventes. Alsace, Moselle et Mayotte ne sont pas couvertes par DVF.")

    plus_cheres = vue.nlargest(10, "prix_m2_median")[["nom_commune", "departement", "prix_m2_median", "nb_ventes"]]
    st.subheader(f"Les 10 communes les plus chères ({type_bien.lower()}s, {annee})")
    st.dataframe(plus_cheres.rename(columns={"nom_commune": "Commune", "departement": "Dépt", "prix_m2_median": "Prix médian (€/m²)",
                                             "nb_ventes": "Ventes"}).style.format({"Prix médian (€/m²)": nb}), hide_index=True)

with evolution:
    libelle = st.selectbox("Commune", communes["libelle"], index=int((communes["libelle"] == "Rennes (35)").argmax()))
    code = communes.loc[communes["libelle"] == libelle, "code_commune"].item()
    serie = prix[prix["code_commune"] == code]
    if serie.empty:
        st.info("Pas assez de ventes dans cette commune (au moins 5 par an et par type) pour publier un prix médian.")
    else:
        tableau = serie.pivot_table(index="annee", columns="type", values="prix_m2_median")
        st.line_chart(tableau, x_label="Année", y_label="Prix médian (€/m²)")
        for type_bien in tableau.columns:
            s = tableau[type_bien].dropna()
            if len(s) >= 2:
                st.metric(f"{type_bien}s : {s.index[0]} → {s.index[-1]}", f"{nb(s.iloc[-1])} €/m²",
                          f"{nb((s.iloc[-1] / s.iloc[0] - 1) * 100, 1)} %")
        ventes = serie.pivot_table(index="annee", columns="type", values="nb_ventes", aggfunc="sum")
        st.caption("Nombre de ventes retenues : " + " · ".join(
            f"{a} : {', '.join(f'{nb(v)} {t.lower()}s' for t, v in ligne.dropna().items())}" for a, ligne in ventes.iterrows()))

with estimation:
    c1, c2 = st.columns(2)
    libelle = c1.selectbox("Commune ", communes["libelle"], index=int((communes["libelle"] == "Rennes (35)").argmax()))
    type_bien = c2.radio("Type de bien ", ["Appartement", "Maison"], horizontal=True)
    c3, c4 = st.columns(2)
    surface = c3.number_input("Surface habitable (m²)", 9, 1000, 65)
    pieces = c4.number_input("Nombre de pièces principales", 1, 20, 3)

    commune = communes[communes["libelle"] == libelle].iloc[0]
    X = pd.DataFrame([{"longitude": commune["longitude"], "latitude": commune["latitude"], "surface": surface,
                       "pieces": pieces, "maison": int(type_bien == "Maison"), "annee": DERNIERE_ANNEE}])[FEATURES]
    bas, median, haut = sorted(float(np.exp(modele()[k].predict(X)[0])) for k in ("bas", "median", "haut"))

    st.metric("Prix estimé", f"{nb(median * surface, 0)} €", help="Estimation au prix de 2025")
    st.write(f"Soit **{nb(median)} €/m²**. Fourchette probable : **{nb(bas * surface)} € à {nb(haut * surface)} €** "
             f"({nb(bas)} à {nb(haut)} €/m²).")
    reference = prix[(prix["code_commune"] == commune["code_commune"]) & (prix["type"] == type_bien)
                     & (prix["annee"] == DERNIERE_ANNEE)]
    if not reference.empty:
        st.caption(f"Pour comparaison, prix médian constaté en {DERNIERE_ANNEE} à {commune['nom_commune']} : "
                   f"{nb(reference['prix_m2_median'].item())} €/m² ({reference['nb_ventes'].item()} ventes).")

    st.divider()
    st.markdown(
        f"**Fiabilité du modèle**, mesurée sur {nb(perf['ventes_test'])} ventes réelles qu'il n'a jamais vues :\n"
        f"- erreur médiane : **{nb(perf['erreur_mediane_pct'], 1)} %** ;\n"
        f"- {nb(perf['part_a_moins_de_20_pct'], 1)} % des estimations à moins de 20 % du prix réel ;\n"
        f"- le vrai prix tombe dans la fourchette dans **{nb(perf['couverture_fourchette_pct'], 1)} %** des cas (cible : 80 %).\n\n"
        "Le modèle (gradient boosting, scikit-learn) ne connaît que la position, la surface, le nombre de pièces, "
        "le type de bien et l'année : ni l'état, ni l'étage, ni la vue. C'est un ordre de grandeur, pas une expertise.")

with qualite:
    st.subheader("Du fichier brut aux ventes exploitables")
    st.write("Les données DVF brutes ne sont pas utilisables telles quelles. Une même vente apparaît sur plusieurs "
             "lignes, avec le prix répété sur chacune. Un immeuble entier peut être vendu en un seul prix, et certaines "
             "ventes sont symboliques (1 €). Chaque étape du nettoyage est comptée :")
    tableau = rapport.pivot_table(index="etape", columns="annee", values="mutations", sort=False)
    tableau["Part restante (2025)"] = (tableau[DERNIERE_ANNEE] / tableau[DERNIERE_ANNEE].iloc[0] * 100).map(lambda v: f"{nb(v, 1)} %")
    tableau.columns = tableau.columns.map(str)
    st.dataframe(tableau.style.format(nb, subset=[c for c in tableau.columns if c.isdigit()]))
    st.caption("Source : Demandes de valeurs foncières géolocalisées, DGFiP / Etalab, data.gouv.fr.")
