import json
from pathlib import Path

import joblib
import numpy as np
import altair as alt
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
    alsace_moselle = json.loads((DATA / "alsace_moselle.geojson").read_text(encoding="utf-8"))
    communes["libelle"] = communes["nom_commune"] + " (" + communes["departement"] + ")"
    return prix, communes.sort_values("libelle"), rapport, perf, alsace_moselle


@st.cache_resource
def modele():
    return joblib.load(DATA / "modele.joblib")


# Classes fixes en €/m² : les couleurs gardent le même sens d'une année et d'un type de bien à l'autre
SEUILS = [1_500, 2_500, 3_500, 5_000, 8_000]
PALETTE = ["#2c7bb6", "#00a6ca", "#90eb9d", "#f9d057", "#f29e2e", "#d7191c"]
LIBELLES = [f"moins de {nb(SEUILS[0])} €"] + [f"{nb(a)} à {nb(b)} €" for a, b in zip(SEUILS, SEUILS[1:])]     + [f"plus de {nb(SEUILS[-1])} €"]


def rgb(hexa: str, alpha: int = 190) -> list[int]:
    return [int(hexa[i:i + 2], 16) for i in (1, 3, 5)] + [alpha]


def legende(classes: pd.Series) -> str:
    """Légende HTML : une pastille de couleur par classe de prix, avec le nombre de communes."""
    comptes = classes.value_counts()
    lignes = [f'<div style="display:flex;align-items:center;gap:8px;margin:5px 0">'
              f'<span style="width:16px;height:16px;border-radius:50%;background:{c};flex:none"></span>'
              f'<span>{lib}<br><small style="color:#888">{nb(comptes.get(i, 0))} communes</small></span></div>'
              for i, (c, lib) in reversed(list(enumerate(zip(PALETTE, LIBELLES))))]
    gris = ('<div style="display:flex;align-items:center;gap:8px;margin:12px 0 5px">'
            '<span style="width:16px;height:16px;background:#bbb;border:1px solid #777;flex:none"></span>'
            '<span>Pas de données publiques<br><small style="color:#888">Alsace-Moselle (livre foncier)</small></span></div>')
    taille = '<p style="margin-top:12px"><small style="color:#888">Taille du point : nombre de ventes</small></p>'
    return '<p style="font-weight:600;margin-bottom:4px">Prix médian au m²</p>' + "".join(lignes) + gris + taille


prix, communes, rapport, perf, alsace_moselle = charger()
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
        classe=np.searchsorted(SEUILS, vue["prix_m2_median"], side="right"),
        rayon=np.sqrt(vue["nb_ventes"]) * 150,
        prix_txt=vue["prix_m2_median"].map(lambda v: f"{nb(v)} €/m²"),
    )
    vue["couleur"] = vue["classe"].map(lambda i: rgb(PALETTE[i]))
    sans_donnees = [
        pdk.Layer("GeoJsonLayer", alsace_moselle, get_fill_color=[150, 150, 150, 90],
                  get_line_color=[90, 90, 90, 200], line_width_min_pixels=1),
        pdk.Layer("TextLayer", [{"position": [6.9, 48.55], "texte": "Pas de données publiques\n(livre foncier)"}],
                  get_position="position", get_text="texte", get_size=13, get_color=[60, 60, 60],
                  character_set="auto"),
    ]
    col_legende, col_carte = st.columns([1, 5])
    col_legende.markdown(legende(vue["classe"]), unsafe_allow_html=True)  # HTML généré ici, sans donnée utilisateur
    col_carte.pydeck_chart(pdk.Deck(
        layers=sans_donnees + [pdk.Layer("ScatterplotLayer", vue, get_position=["longitude", "latitude"],
                                         get_fill_color="couleur", get_radius="rayon", radius_min_pixels=2, pickable=True)],
        initial_view_state=pdk.ViewState(latitude=46.6, longitude=2.4, zoom=5),
        tooltip={"text": "{nom_commune} ({departement})\n{prix_txt} — {nb_ventes} ventes"},
        map_style=None,
    ), height=600)
    st.caption(f"{nb(len(vue))} communes avec au moins 5 ventes. En gris : le Bas-Rhin, le Haut-Rhin et la Moselle "
               "(Strasbourg, Mulhouse, Metz…), où les ventes sont inscrites au livre foncier, un registre de droit local "
               "tenu par le ministère de la Justice et non publié en open data. Mayotte est aussi absente.")

    plus_cheres = vue.nlargest(10, "prix_m2_median")[["nom_commune", "departement", "prix_m2_median", "nb_ventes"]]
    st.subheader(f"Les 10 communes les plus chères ({type_bien.lower()}s, {annee})")
    st.dataframe(plus_cheres.rename(columns={"nom_commune": "Commune", "departement": "Dépt", "prix_m2_median": "Prix médian (€/m²)",
                                             "nb_ventes": "Ventes"}).style.format({"Prix médian (€/m²)": nb}), hide_index=True)

with evolution:
    st.caption("Les communes du Bas-Rhin, du Haut-Rhin et de la Moselle n'apparaissent pas : leurs ventes ne sont pas publiées.")
    libelle = st.selectbox("Commune", communes["libelle"], index=int((communes["libelle"] == "Rennes (35)").argmax()))
    code = communes.loc[communes["libelle"] == libelle, "code_commune"].item()
    serie = prix[prix["code_commune"] == code]
    if serie.empty:
        st.info("Pas assez de ventes dans cette commune (au moins 5 par an et par type) pour publier un prix médian.")
    else:
        tableau = serie.pivot_table(index="annee", columns="type", values="prix_m2_median")
        donnees = tableau.reset_index().melt("annee", var_name="Type", value_name="prix").dropna()
        st.altair_chart(alt.Chart(donnees).mark_line(point=True).encode(
            x=alt.X("annee:O", title="Année", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("prix:Q", title="Prix médian (€/m²)",
                    axis=alt.Axis(labelExpr="replace(format(datum.value, ',.0f'), /,/g, ' ')")),
            color=alt.Color("Type:N", scale=alt.Scale(domain=["Appartement", "Maison"], range=["#2c7bb6", "#d7191c"])),
            tooltip=[alt.Tooltip("annee:O", title="Année"), "Type", alt.Tooltip("prix:Q", title="€/m²", format=",.0f")],
        ), width="stretch")
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
                   f"{nb(reference['prix_m2_median'].item())} €/m² ({nb(reference['nb_ventes'].item())} ventes).")

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
    tableau = rapport.pivot_table(index="etape", columns="annee", values="mutations", sort=False).rename_axis("Étape")
    tableau["Part restante (2025)"] = (tableau[DERNIERE_ANNEE] / tableau[DERNIERE_ANNEE].iloc[0] * 100).map(lambda v: f"{nb(v, 1)} %")
    tableau.columns = tableau.columns.map(str)
    st.dataframe(tableau.style.format(nb, subset=[c for c in tableau.columns if c.isdigit()]))
    st.caption("Source : Demandes de valeurs foncières géolocalisées, DGFiP / Etalab, data.gouv.fr.")
