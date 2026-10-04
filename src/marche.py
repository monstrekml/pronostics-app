"""
Le modèle face au marché des paris.

Les cotes des bookmakers contiennent des informations que le modèle n'a pas
(blessures, compositions, transferts, argent misé). On les utilise de deux façons :

1. Probabilités du marché : 1 / cote, puis on retire la marge du bookmaker pour
   que la somme fasse 100 %.
2. Probabilités combinées : moyenne géométrique pondérée du modèle et du marché,
   renormalisée. Le poids du modèle est réglé sur les matchs archivés dès qu'il y
   en a assez ; avant cela, on prend 50/50.

Le bilan « modèle vs marché » n'utilise que des cotes relevées AVANT le match.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from metriques import log_loss, resume

MIN_MATCHS_REGLAGE = 150
POIDS_DEFAUT = 0.5


def combiner(p_modele: np.ndarray, p_marche: np.ndarray, poids_modele: float) -> np.ndarray:
    """Moyenne géométrique pondérée, ligne par ligne, renormalisée."""
    pm = np.clip(np.asarray(p_modele, float), 1e-6, 1)
    pk = np.clip(np.asarray(p_marche, float), 1e-6, 1)
    p = pm ** poids_modele * pk ** (1 - poids_modele)
    return p / p.sum(axis=1, keepdims=True)


def regler_poids(p_modele: np.ndarray, p_marche: np.ndarray, y: np.ndarray) -> float:
    if len(y) < MIN_MATCHS_REGLAGE:
        return POIDS_DEFAUT
    grille = np.round(np.arange(0, 1.01, 0.1), 2)
    pertes = [log_loss(combiner(p_modele, p_marche, w), y) for w in grille]
    return float(grille[int(np.argmin(pertes))])


def bilan(p_modele: np.ndarray, p_marche: np.ndarray, y: np.ndarray, ordinal: bool) -> tuple[pd.DataFrame, float]:
    w = regler_poids(p_modele, p_marche, y)
    lignes = [
        resume("Modèle seul", p_modele, y, ordinal),
        resume("Marché (bookmakers)", p_marche, y, ordinal),
        resume(f"Combiné ({w:.0%} modèle)", combiner(p_modele, p_marche, w), y, ordinal),
    ]
    return pd.DataFrame(lignes), w
