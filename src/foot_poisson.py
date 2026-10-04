"""
Modèle de football : Poisson avec correction de Dixon-Coles.

Idée
----
Le nombre de buts d'une équipe dans un match suit à peu près une loi de Poisson.
On estime pour chaque équipe :
  - une force d'attaque  (att)  : plus elle est haute, plus l'équipe marque ;
  - une faiblesse défensive (dfs) : plus elle est haute, plus l'équipe encaisse ;
  - un avantage du terrain commun à toutes les équipes.

    log(buts attendus domicile)   = mu + avantage_domicile + att[dom] + dfs[ext]
    log(buts attendus extérieur)  = mu                     + att[ext] + dfs[dom]

À partir des deux moyennes de buts attendus, on calcule la probabilité de chaque
score exact (0-0, 1-0, 2-1, ...), puis on en déduit 1N2, plus/moins de 2,5 buts, etc.

Deux améliorations classiques (Dixon & Coles, 1997) :
  1. Pondération temporelle : un match d'il y a deux ans compte moins qu'un match
     de la semaine dernière (décroissance exponentielle, paramètre xi).
  2. Correction rho des petits scores (0-0, 1-0, 0-1, 1-1), que la loi de Poisson
     indépendante prédit mal (elle sous-estime notamment les nuls 0-0 et 1-1).

Pour rester simple et rapide, l'ajustement se fait en deux étapes : d'abord les
forces des équipes (régression de Poisson pondérée, avec gradient analytique),
puis rho seul. C'est une approximation légère de l'estimation jointe d'origine.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

MAX_BUTS = 10  # on calcule les scores de 0-0 à 10-10


def _tau(x, y, lam, mu, rho):
    """Facteur de correction de Dixon-Coles pour les petits scores."""
    t = np.ones_like(lam, dtype=float)
    t = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return t


@dataclass
class ModeleFoot:
    xi: float = 0.0019          # décroissance temporelle par jour (demi-vie ≈ 1 an)
    ridge: float = 0.01         # petite régularisation pour stabiliser l'estimation
    min_matchs: int = 10        # en dessous, l'équipe est traitée comme un promu
    equipes: list = field(default_factory=list)
    att: dict = field(default_factory=dict)
    dfs: dict = field(default_factory=dict)
    mu: float = 0.0
    domicile: float = 0.0
    rho: float = 0.0
    att_promu: float = 0.0
    dfs_promu: float = 0.0

    # ------------------------------------------------------------------ apprentissage
    def ajuster(self, matchs: pd.DataFrame, date_ref: pd.Timestamp | None = None) -> "ModeleFoot":
        """matchs : colonnes Date, HomeTeam, AwayTeam, FTHG, FTAG (format football-data.co.uk)."""
        if date_ref is None:
            date_ref = matchs["Date"].max() + pd.Timedelta(days=1)
        age = (date_ref - matchs["Date"]).dt.days.to_numpy()
        w = np.exp(-self.xi * age)

        # Les équipes avec trop peu de matchs récents ne sont pas estimées
        recents = matchs[matchs["Date"] >= date_ref - pd.Timedelta(days=365)]
        nb = pd.concat([recents["HomeTeam"], recents["AwayTeam"]]).value_counts()
        self.equipes = sorted(nb[nb >= self.min_matchs].index)
        idx = {e: i for i, e in enumerate(self.equipes)}
        garde = matchs["HomeTeam"].isin(idx) & matchs["AwayTeam"].isin(idx)
        m = matchs[garde]
        w = w[garde.to_numpy()]
        h = m["HomeTeam"].map(idx).to_numpy()
        a = m["AwayTeam"].map(idx).to_numpy()
        gh = m["FTHG"].to_numpy(float)
        ga = m["FTAG"].to_numpy(float)
        n = len(self.equipes)

        def objectif(theta):
            mu, dom = theta[0], theta[1]
            att, dfs = theta[2:2 + n], theta[2 + n:]
            l_h = np.exp(mu + dom + att[h] + dfs[a])
            l_a = np.exp(mu + att[a] + dfs[h])
            # log-vraisemblance de Poisson pondérée (sans le terme constant log(y!))
            ll = np.sum(w * (gh * np.log(l_h) - l_h + ga * np.log(l_a) - l_a))
            pen = self.ridge * (att @ att + dfs @ dfs)
            r_h = w * (gh - l_h)   # résidus
            r_a = w * (ga - l_a)
            g = np.zeros_like(theta)
            g[0] = r_h.sum() + r_a.sum()
            g[1] = r_h.sum()
            g[2:2 + n] = np.bincount(h, r_h, n) + np.bincount(a, r_a, n) - 2 * self.ridge * att
            g[2 + n:] = np.bincount(a, r_h, n) + np.bincount(h, r_a, n) - 2 * self.ridge * dfs
            return -(ll - pen), -g

        theta0 = np.zeros(2 + 2 * n)
        theta0[0] = np.log(max(np.average(np.r_[gh, ga], weights=np.r_[w, w]), 0.1))
        res = minimize(objectif, theta0, jac=True, method="L-BFGS-B",
                       options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 2000})
        t = res.x
        self.mu, self.domicile = t[0], t[1]
        self.att = dict(zip(self.equipes, t[2:2 + n]))
        self.dfs = dict(zip(self.equipes, t[2 + n:]))

        # Correction rho (étape 2)
        l_h = np.exp(self.mu + self.domicile + t[2:2 + n][h] + t[2 + n:][a])
        l_a = np.exp(self.mu + t[2:2 + n][a] + t[2 + n:][h])
        x, y = gh.astype(int), ga.astype(int)

        def nll_rho(rho):
            tau = _tau(x, y, l_h, l_a, rho)
            if np.any(tau <= 0):
                return 1e10
            return -np.sum(w * np.log(tau))

        self.rho = minimize_scalar(nll_rho, bounds=(-0.2, 0.2), method="bounded").x

        # Valeur par défaut pour un promu : moyenne des 3 équipes les plus faibles
        force = sorted(self.equipes, key=lambda e: self.att[e] - self.dfs[e])[:3]
        self.att_promu = float(np.mean([self.att[e] for e in force]))
        self.dfs_promu = float(np.mean([self.dfs[e] for e in force]))
        return self

    # ------------------------------------------------------------------ prédiction
    def buts_attendus(self, dom: str, ext: str) -> tuple[float, float]:
        a_d, d_d = self.att.get(dom, self.att_promu), self.dfs.get(dom, self.dfs_promu)
        a_e, d_e = self.att.get(ext, self.att_promu), self.dfs.get(ext, self.dfs_promu)
        return (float(np.exp(self.mu + self.domicile + a_d + d_e)),
                float(np.exp(self.mu + a_e + d_d)))

    def matrice_scores(self, dom: str, ext: str) -> np.ndarray:
        """Matrice M[i, j] = P(domicile marque i buts, extérieur marque j buts)."""
        lh, la = self.buts_attendus(dom, ext)
        k = np.arange(MAX_BUTS + 1)
        m = np.outer(poisson.pmf(k, lh), poisson.pmf(k, la))
        m[0, 0] *= 1 - lh * la * self.rho
        m[0, 1] *= 1 + lh * self.rho
        m[1, 0] *= 1 + la * self.rho
        m[1, 1] *= 1 - self.rho
        return m / m.sum()

    def predire(self, dom: str, ext: str) -> dict:
        m = self.matrice_scores(dom, ext)
        lh, la = self.buts_attendus(dom, ext)
        total = np.add.outer(np.arange(MAX_BUTS + 1), np.arange(MAX_BUTS + 1))
        return {
            "buts_dom": lh,
            "buts_ext": la,
            "p_dom": float(np.tril(m, -1).sum()),
            "p_nul": float(np.trace(m)),
            "p_ext": float(np.triu(m, 1).sum()),
            "p_plus_2_5": float(m[total > 2.5].sum()),
            "p_les_deux_marquent": float(m[1:, 1:].sum()),
        }

    def scores_probables(self, dom: str, ext: str, n: int = 5) -> list[tuple[str, float]]:
        m = self.matrice_scores(dom, ext)
        ordre = np.dstack(np.unravel_index(np.argsort(-m, axis=None), m.shape))[0][:n]
        return [(f"{i}-{j}", float(m[i, j])) for i, j in ordre]

    def classement_forces(self) -> pd.DataFrame:
        df = pd.DataFrame({"equipe": self.equipes,
                           "attaque": [self.att[e] for e in self.equipes],
                           "defense": [-self.dfs[e] for e in self.equipes]})
        df["force"] = df["attaque"] + df["defense"]
        return df.sort_values("force", ascending=False).reset_index(drop=True)
