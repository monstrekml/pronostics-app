"""Tests hors ligne des parseurs de données et des modèles.

    python -m pytest -q
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))

from basket_elo import EloNBA                       # noqa: E402
from foot_poisson import ModeleFoot                  # noqa: E402
from sources import (                                # noqa: E402
    analyser_espn, analyser_fdorg, analyser_openfootball, charger_correspondance,
    fusionner_resultats, normaliser_nom, saison_foot, saison_nba,
)

ECH = RACINE / "tests" / "echantillons"


def lire(nom):
    return json.loads((ECH / nom).read_text())


def test_saisons():
    assert saison_foot(date(2026, 10, 4)) == "2026-27"
    assert saison_foot(date(2027, 5, 30)) == "2026-27"
    assert saison_nba(date(2026, 10, 25)) == 2027
    assert saison_nba(date(2027, 4, 1)) == 2027


def test_correspondance_noms():
    corr = charger_correspondance()
    assert corr["Paris Saint-Germain FC"] == "Paris SG"
    assert corr["Manchester City FC"] == "Man City"
    # variante inconnue mais proche d'un nom connu
    assert normaliser_nom("Paris Saint Germain FC", corr) == "Paris SG"
    # promu jamais vu : nom nettoyé
    assert normaliser_nom("Real Oviedo CF", corr) in {"Real Oviedo", "Oviedo"}


def test_openfootball():
    res, cal = analyser_openfootball(lire("openfootball.json"), charger_correspondance())
    assert len(cal) == 4 and len(res) == 3
    assert res.iloc[0].to_dict() == {"Date": "2026-08-21", "HomeTeam": "Marseille", "AwayTeam": "Strasbourg",
                                     "FTHG": 4, "FTAG": 0}
    assert cal.iloc[3]["HomeTeam"] == "Paris FC"


def test_football_data_org_et_fusion():
    corr = charger_correspondance()
    res, cal = analyser_fdorg(lire("football_data_org.json"), corr, set(corr.values()))
    assert list(res["HomeTeam"]) == ["Marseille", "Paris SG"]
    assert cal.iloc[0]["Heure"] == "20:45"            # converti en heure de Paris
    base = res.iloc[:1]
    fusion = fusionner_resultats(base, res)
    assert len(fusion) == 2                            # un seul ajout, pas de doublon


def test_espn():
    df = analyser_espn(lire("espn_nba.json"))
    # play-in + playoffs + match à venir ; présaison et All-Star exclus
    assert list(df["id"]) == ["401810001", "401810002", "401810003"]
    assert df.iloc[0]["dom"] == "Sixers" and df.iloc[0]["pts_dom"] == 109
    assert df.iloc[1]["playoffs"] == 1 and df.iloc[1]["ext"] == "Warriors"
    a_venir = df.iloc[2]
    assert a_venir["termine"] == 0 and a_venir["dom"] == "Trailblazers" and a_venir["neutre"] == 1
    assert pd.isna(a_venir["pts_dom"])
    assert df.iloc[0]["date"] == "2026-04-15"          # date américaine (heure de New York)


def test_modele_foot_coherent():
    rng = np.random.default_rng(0)
    equipes = [f"E{i}" for i in range(10)]
    force = dict(zip(equipes, np.linspace(-0.4, 0.4, 10)))
    lignes = []
    jour = pd.Timestamp("2025-08-01")
    for k in range(400):
        d, e = rng.choice(equipes, 2, replace=False)
        lignes.append({"Date": jour + pd.Timedelta(days=k), "HomeTeam": d, "AwayTeam": e,
                       "FTHG": rng.poisson(np.exp(0.3 + force[d] - force[e])),
                       "FTAG": rng.poisson(np.exp(0.1 + force[e] - force[d]))})
    m = ModeleFoot().ajuster(pd.DataFrame(lignes))
    p = m.predire("E9", "E0")
    assert abs(p["p_dom"] + p["p_nul"] + p["p_ext"] - 1) < 1e-9
    assert p["p_dom"] > 0.6                            # la meilleure équipe à domicile est favorite
    assert m.classement_forces().iloc[0]["equipe"] in {"E9", "E8"}


def test_elo_retour_moyenne():
    elo = EloNBA()
    notes = elo.notes_pour_saison({"A": 1705, "B": 1305}, 2026, 2027)
    assert notes["A"] == 1655 and notes["B"] == 1355
    p = elo.predire(notes, "A", "B")
    assert 0.5 < p["p_dom"] < 1 and abs(p["p_dom"] + p["p_ext"] - 1) < 1e-12
