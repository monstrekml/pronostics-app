"""
Modèle de basket (NBA) : classement Elo + régression logistique.

Elo, en bref
------------
Chaque équipe a une note (≈1500 = équipe moyenne). Avant un match, l'écart de notes
donne une probabilité de victoire :

    P(A bat B) = 1 / (1 + 10 ** (-(Elo_A - Elo_B + avantage_domicile) / 400))

Après le match, le gagnant prend des points au perdant. Le transfert est d'autant
plus grand que le résultat était inattendu et que l'écart au score est large
(multiplicateur de marge de victoire, comme chez FiveThirtyEight).
Entre deux saisons, chaque note est ramenée d'un quart vers la moyenne, car les
effectifs changent.

Ensuite, une régression logistique combine l'écart Elo avec d'autres signaux
(jours de repos, back-to-back, forme récente, playoffs) pour voir si l'on gagne
en précision.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ELO_MOYEN = 1505


@dataclass
class EloNBA:
    k: float = 20.0
    avantage_domicile: float = 100.0
    report_saison: float = 0.75   # part de la note conservée d'une saison à l'autre
    elo_depart: float = 1300.0    # note d'une nouvelle franchise

    @staticmethod
    def proba(diff: np.ndarray | float) -> np.ndarray | float:
        return 1 / (1 + 10 ** (-np.asarray(diff) / 400))

    def parcourir(self, matchs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Rejoue tous les matchs dans l'ordre chronologique.

        Retourne le tableau des matchs enrichi des notes AVANT chaque match
        (donc sans fuite d'information), et les notes finales.
        """
        notes: dict[str, float] = {}
        saison_courante = None
        elo_dom = np.empty(len(matchs))
        elo_ext = np.empty(len(matchs))
        for i, r in enumerate(matchs.itertuples(index=False)):
            if r.saison != saison_courante:
                notes = {e: ELO_MOYEN + self.report_saison * (v - ELO_MOYEN) for e, v in notes.items()}
                saison_courante = r.saison
            ed = notes.get(r.dom, self.elo_depart)
            ee = notes.get(r.ext, self.elo_depart)
            elo_dom[i], elo_ext[i] = ed, ee
            hfa = 0 if r.neutre else self.avantage_domicile
            p = self.proba(ed + hfa - ee)
            marge = abs(r.marge_dom)
            ecart_gagnant = (ed + hfa - ee) if r.victoire_dom else (ee - ed - hfa)
            mult = ((marge + 3) ** 0.8) / (7.5 + 0.006 * ecart_gagnant)
            delta = self.k * mult * (r.victoire_dom - p)
            notes[r.dom] = ed + delta
            notes[r.ext] = ee - delta
        res = matchs.copy()
        res["elo_dom"] = elo_dom
        res["elo_ext"] = elo_ext
        res["diff_elo"] = elo_dom - elo_ext + np.where(res["neutre"] == 1, 0, self.avantage_domicile)
        res["p_elo_dom"] = self.proba(res["diff_elo"])
        res["ecart_prevu"] = res["diff_elo"] / 28  # ≈ 28 points Elo par point au score
        return res, notes

    def notes_pour_saison(self, notes: dict, derniere_saison: int, saison_cible: int) -> dict:
        """Applique le retour vers la moyenne si l'on prédit une saison qui n'a pas commencé."""
        for _ in range(max(0, saison_cible - derniere_saison)):
            notes = {e: ELO_MOYEN + self.report_saison * (v - ELO_MOYEN) for e, v in notes.items()}
        return notes

    def predire(self, notes: dict, dom: str, ext: str, neutre: bool = False) -> dict:
        diff = notes.get(dom, self.elo_depart) - notes.get(ext, self.elo_depart) + (0 if neutre else self.avantage_domicile)
        p = float(self.proba(diff))
        return {"p_dom": p, "p_ext": 1 - p, "ecart": diff / 28,
                "elo_dom": notes.get(dom, self.elo_depart), "elo_ext": notes.get(ext, self.elo_depart)}


def ajouter_variables(matchs: pd.DataFrame) -> pd.DataFrame:
    """Variables de contexte calculées uniquement avec le passé de chaque équipe."""
    lignes = []
    for cote, autre in (("dom", "ext"), ("ext", "dom")):
        t = matchs[["date", "saison", cote, "victoire_dom"]].copy()
        t["equipe"] = t[cote]
        t["victoire"] = t["victoire_dom"] if cote == "dom" else 1 - t["victoire_dom"]
        t["cote"] = cote
        t["idx"] = t.index
        lignes.append(t[["idx", "date", "saison", "equipe", "victoire", "cote"]])
    long = pd.concat(lignes).sort_values(["equipe", "date", "idx"])
    g = long.groupby(["equipe", "saison"])
    long["repos"] = g["date"].diff().dt.days.fillna(7).clip(lower=1, upper=7) - 1
    long["forme10"] = g["victoire"].transform(lambda s: s.shift().rolling(10, min_periods=3).mean()).fillna(0.5)
    larg = long.pivot(index="idx", columns="cote", values=["repos", "forme10"])
    larg.columns = [f"{a}_{b}" for a, b in larg.columns]
    res = matchs.join(larg)
    res["b2b_dom"] = (res["repos_dom"] == 0).astype(int)
    res["b2b_ext"] = (res["repos_ext"] == 0).astype(int)
    res["diff_repos"] = res["repos_dom"].clip(upper=3) - res["repos_ext"].clip(upper=3)
    res["diff_forme"] = res["forme10_dom"] - res["forme10_ext"]
    return res


VARIABLES = ["diff_elo", "diff_repos", "b2b_dom", "b2b_ext", "diff_forme", "playoffs", "neutre"]
