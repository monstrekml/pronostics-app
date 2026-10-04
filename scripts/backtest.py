"""
Évalue les modèles sur le passé et écrit les tableaux affichés dans l'onglet
« Fiabilité » de l'application (dossier resultats/). Relancé chaque lundi par
la GitHub Action.

Foot : walk-forward hebdomadaire depuis 2016/17 (le modèle ne voit jamais le futur).
NBA  : notes Elo calculées match après match, évaluées depuis 2005.

    python scripts/backtest.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from basket_elo import EloNBA                                   # noqa: E402
from donnees import charger_nba, charger_resultats_foot         # noqa: E402
from foot_poisson import ModeleFoot                             # noqa: E402
from metriques import resume, table_calibration                 # noqa: E402
from sources import LIGUES, RACINE                              # noqa: E402

SORTIE = RACINE / "resultats"


def walk_forward_foot(df: pd.DataFrame, debut: str = "2016-07-01", fenetre: int = 730) -> pd.DataFrame:
    lignes = []
    t = pd.Timestamp(debut)
    while t <= df["Date"].max():
        bloc = df[(df["Date"] >= t) & (df["Date"] < t + pd.Timedelta(days=7))]
        if len(bloc):
            histo = df[(df["Date"] < t) & (df["Date"] >= t - pd.Timedelta(days=fenetre))]
            m = ModeleFoot().ajuster(histo, date_ref=t)
            freq = np.bincount(histo["Resultat"], minlength=3) / len(histo)
            for r in bloc.itertuples():
                p = m.predire(r.HomeTeam, r.AwayTeam)
                lignes.append({"Date": r.Date, "Saison": r.Saison, "Domicile": r.HomeTeam, "Exterieur": r.AwayTeam,
                               "ButsDom": r.FTHG, "ButsExt": r.FTAG, "Resultat": r.Resultat,
                               "p1": p["p_dom"], "pN": p["p_nul"], "p2": p["p_ext"], "p_plus_2_5": p["p_plus_2_5"],
                               "ref1": freq[0], "refN": freq[1], "ref2": freq[2]})
        t += pd.Timedelta(days=7)
    return pd.DataFrame(lignes)


def confiance(p_max: np.ndarray, ok: np.ndarray, seuils) -> pd.DataFrame:
    return pd.DataFrame([{"confiance_min": s, "part_des_matchs": float((p_max >= s).mean()),
                          "n_matchs": int((p_max >= s).sum()), "precision": float(ok[p_max >= s].mean())}
                         for s in seuils if (p_max >= s).any()])


def foot() -> None:
    toutes, lignes = [], []
    for ligue, info in LIGUES.items():
        print(f"Foot {info['nom']}…")
        pred = walk_forward_foot(charger_resultats_foot(ligue))
        pred["Ligue"] = info["nom"]
        toutes.append(pred)
    pred = pd.concat(toutes, ignore_index=True)
    for nom, g in [*pred.groupby("Ligue"), ("Toutes", pred)]:
        y = g["Resultat"].to_numpy()
        lignes.append({"ligue": nom, **resume("Modèle Dixon-Coles", g[["p1", "pN", "p2"]].to_numpy(), y, ordinal=True)})
        lignes.append({"ligue": nom, **resume("Référence naïve", g[["ref1", "refN", "ref2"]].to_numpy(), y, ordinal=True)})
    pd.DataFrame(lignes).to_csv(SORTIE / "foot_metriques.csv", index=False)

    p = pred[["p1", "pN", "p2"]].to_numpy()
    ok = p.argmax(1) == pred["Resultat"].to_numpy()
    confiance(p.max(1), ok, [0.0, 0.4, 0.5, 0.6, 0.7, 0.8]).to_csv(SORTIE / "foot_confiance.csv", index=False)

    cal = []
    for col, idx, nom in (("p1", 0, "Victoire domicile"), ("pN", 1, "Nul"), ("p2", 2, "Victoire extérieur")):
        t = table_calibration(pred[col].to_numpy(), (pred["Resultat"] == idx).astype(int).to_numpy())
        t["issue"] = nom
        cal.append(t)
    pd.concat(cal).to_csv(SORTIE / "foot_calibration.csv", index=False)

    pred["ok"] = ok
    (pred.groupby(["Ligue", "Saison"])["ok"].mean().reset_index(name="precision")
     .to_csv(SORTIE / "foot_precision_saison.csv", index=False))


def nba() -> None:
    print("NBA…")
    matchs, _ = charger_nba()
    res, _ = EloNBA().parcourir(matchs)
    test = res[res["saison"] >= 2005].copy()
    y = test["victoire_dom"].to_numpy()
    naif = res.loc[res["saison"].between(1985, 2004), "victoire_dom"].mean()
    deux = lambda p: np.c_[1 - p, p]  # noqa: E731
    lignes = [
        {"periode": "2005-2015", **resume("Toujours l'équipe à domicile", deux(np.full(len(test), naif))[test.saison <= 2015],
                                          y[test.saison <= 2015])},
        {"periode": "2005-2015", **resume("Elo", deux(test.p_elo_dom.to_numpy())[test.saison <= 2015], y[test.saison <= 2015])},
        {"periode": "2005-2015", **resume("FiveThirtyEight (publié)", deux(test.p538_dom.to_numpy())[test.saison <= 2015],
                                          y[test.saison <= 2015])},
    ]
    recent = test[test.saison > 2015]
    if len(recent):
        yr = recent["victoire_dom"].to_numpy()
        periode = f"2016-{recent.saison.max()}"
        lignes += [
            {"periode": periode, **resume("Toujours l'équipe à domicile", deux(np.full(len(recent), naif)), yr)},
            {"periode": periode, **resume("Elo", deux(recent.p_elo_dom.to_numpy()), yr)},
        ]
    pd.DataFrame(lignes).to_csv(SORTIE / "nba_metriques.csv", index=False)

    p = test["p_elo_dom"].to_numpy()
    ok = (p > 0.5) == (y == 1)
    confiance(np.maximum(p, 1 - p), ok, [0.5, 0.6, 0.7, 0.8, 0.9]).to_csv(SORTIE / "nba_confiance.csv", index=False)
    table_calibration(p, y).to_csv(SORTIE / "nba_calibration.csv", index=False)
    test["ok"] = ok
    test.groupby("saison")["ok"].mean().reset_index(name="precision").to_csv(SORTIE / "nba_precision_saison.csv", index=False)


if __name__ == "__main__":
    SORTIE.mkdir(exist_ok=True)
    foot()
    nba()
    print("Résultats écrits dans", SORTIE)
