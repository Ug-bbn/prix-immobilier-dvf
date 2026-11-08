import io

import pandas as pd
import pytest

import preparation as p

ENTETE = ("id_mutation,date_mutation,nature_mutation,valeur_fonciere,code_commune,nom_commune,code_departement,"
          "id_parcelle,lot1_numero,type_local,surface_reelle_bati,nombre_pieces_principales,longitude,latitude\n")
LIGNES = ENTETE + "\n".join([
    # M1 : maison répétée sur deux natures de culture + une dépendance -> 1 vente, 300 000 / 100 m²
    "M1,2025-01-03,Vente,300000,35238,Rennes,35,P1,,Maison,100,5,-1.68,48.10",
    "M1,2025-01-03,Vente,300000,35238,Rennes,35,P1,,Maison,100,5,-1.68,48.10",
    "M1,2025-01-03,Vente,300000,35238,Rennes,35,P1,12,Dépendance,,0,-1.68,48.10",
    # M2 : deux appartements vendus ensemble -> exclue (prix global, pas de prix au m² fiable)
    "M2,2025-02-01,Vente,500000,35238,Rennes,35,P2,1,Appartement,50,2,-1.68,48.11",
    "M2,2025-02-01,Vente,500000,35238,Rennes,35,P2,2,Appartement,60,3,-1.68,48.11",
    # M3 : vente en l'état futur d'achèvement -> exclue
    "M3,2025-03-01,Vente en l'état futur d'achèvement,200000,35238,Rennes,35,P3,1,Appartement,40,2,-1.68,48.12",
    # M4 : appartement vendu 1 € -> prix au m² aberrant, exclu
    "M4,2025-04-01,Vente,1,35238,Rennes,35,P4,1,Appartement,40,2,-1.68,48.13",
    # M5 : appartement + local commercial -> exclue
    "M5,2025-05-01,Vente,400000,35238,Rennes,35,P5,1,Appartement,70,3,-1.68,48.14",
    "M5,2025-05-01,Vente,400000,35238,Rennes,35,P5,2,Local industriel. commercial ou assimilé,80,0,-1.68,48.14",
    # M6 : appartement valide, 180 000 / 60 m²
    "M6,2025-06-01,Vente,180000,35238,Rennes,35,P6,7,Appartement,60,3,-1.68,48.15",
    # M7 : surface de 5 m² -> exclue
    "M7,2025-07-01,Vente,30000,35238,Rennes,35,P7,3,Appartement,5,1,-1.68,48.16",
])


@pytest.fixture
def resultat():
    return p.nettoyer(pd.read_csv(io.StringIO(LIGNES), dtype=p.COLONNES))


def test_seules_les_ventes_fiables_restent(resultat):
    ventes, _ = resultat
    assert sorted(ventes["prix_m2"]) == [3000, 3000]  # M1 et M6
    assert set(ventes["type"]) == {"Maison", "Appartement"}


def test_lignes_dupliquees_ne_font_qu_une_vente(resultat):
    ventes, _ = resultat
    assert (ventes["type"] == "Maison").sum() == 1


def test_rapport_compte_chaque_etape(resultat):
    _, etapes = resultat
    restant = dict(etapes)
    assert restant["Mutations publiées"] == 7
    assert restant["Ventes classiques (hors VEFA, échanges, adjudications…)"] == 6
    assert restant["Un seul logement, sans local commercial (dépendances admises)"] == 4  # M1, M4, M6, M7
    assert [n for _, n in etapes] == sorted([n for _, n in etapes], reverse=True)  # jamais de ventes ajoutées


def test_mediane_publiee_seulement_avec_assez_de_ventes():
    ventes = pd.DataFrame({
        "annee": 2025, "code_commune": ["A"] * 5 + ["B"] * 4, "nom_commune": "x", "departement": "35",
        "type": "Appartement", "prix_m2": [1000, 2000, 3000, 4000, 5000, 1, 1, 1, 1],
    })
    agrege = p.agreger(ventes)
    assert list(agrege["code_commune"]) == ["A"] and agrege["prix_m2_median"].item() == 3000


def test_modele_sur_donnees_synthetiques():
    # prix au m² qui dépend de la position : le modèle doit le retrouver
    import numpy as np
    rng = np.random.default_rng(0)
    n = 4000
    lon = rng.uniform(0, 5, n)  # prix au m² de 2 000 à 7 000 €
    ventes = pd.DataFrame({
        "longitude": lon, "latitude": rng.uniform(43, 50, n), "surface": rng.uniform(20, 150, n),
        "pieces": rng.integers(1, 6, n), "type": rng.choice(["Maison", "Appartement"], n), "annee": 2025,
        "prix_m2": 2000 + 1000 * lon + rng.normal(0, 100, n),
    })
    modeles, perf = p.entrainer(ventes)
    assert perf["erreur_mediane_pct"] < 10
    assert 70 <= perf["couverture_fourchette_pct"] <= 90  # fourchette 10 %-90 % : environ 80 % des cas
