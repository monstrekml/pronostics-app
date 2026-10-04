"""
Métriques pour évaluer des prédictions probabilistes.

Un bon modèle de pronostic ne se juge PAS seulement à son taux de bonnes réponses :
il faut aussi vérifier que ses probabilités sont justes (calibration). Un modèle
qui annonce 70 % doit avoir raison environ 7 fois sur 10.
"""
import numpy as np
import pandas as pd

EPS = 1e-12


def precision(probas: np.ndarray, resultats: np.ndarray) -> float:
    """Taux de bonnes réponses : la classe la plus probable est-elle la bonne ?

    probas    : tableau (n, k) de probabilités
    resultats : tableau (n,) d'indices de la classe réelle (0..k-1)
    """
    return float(np.mean(np.argmax(probas, axis=1) == resultats))


def log_loss(probas: np.ndarray, resultats: np.ndarray) -> float:
    """Perte logarithmique : punit fortement les erreurs commises avec assurance.
    Plus c'est bas, mieux c'est."""
    p = probas[np.arange(len(resultats)), resultats]
    return float(-np.mean(np.log(np.clip(p, EPS, 1))))


def brier(probas: np.ndarray, resultats: np.ndarray) -> float:
    """Score de Brier multi-classes : erreur quadratique moyenne entre les
    probabilités et le résultat (0 = parfait). Plus c'est bas, mieux c'est."""
    cible = np.zeros_like(probas)
    cible[np.arange(len(resultats)), resultats] = 1
    return float(np.mean(np.sum((probas - cible) ** 2, axis=1)))


def rps(probas: np.ndarray, resultats: np.ndarray) -> float:
    """Ranked Probability Score : la métrique de référence pour le football 1N2,
    car elle tient compte de l'ordre des issues (domicile > nul > extérieur).
    Prédire « nul » quand l'équipe à domicile gagne est moins grave que prédire
    « victoire extérieur ». Plus c'est bas, mieux c'est."""
    cible = np.zeros_like(probas)
    cible[np.arange(len(resultats)), resultats] = 1
    cum_p = np.cumsum(probas, axis=1)[:, :-1]
    cum_c = np.cumsum(cible, axis=1)[:, :-1]
    return float(np.mean(np.sum((cum_p - cum_c) ** 2, axis=1) / (probas.shape[1] - 1)))


def resume(nom: str, probas: np.ndarray, resultats: np.ndarray, ordinal: bool = False) -> dict:
    ligne = {
        "modele": nom,
        "n_matchs": len(resultats),
        "precision": precision(probas, resultats),
        "log_loss": log_loss(probas, resultats),
        "brier": brier(probas, resultats),
    }
    if ordinal:
        ligne["rps"] = rps(probas, resultats)
    return ligne


def table_calibration(p_predite: np.ndarray, observe: np.ndarray, n_tranches: int = 10) -> pd.DataFrame:
    """Regroupe les prédictions par tranche de probabilité et compare la
    probabilité annoncée à la fréquence réellement observée."""
    tranches = np.linspace(0, 1, n_tranches + 1)
    idx = np.clip(np.digitize(p_predite, tranches) - 1, 0, n_tranches - 1)
    df = pd.DataFrame({"tranche": idx, "p": p_predite, "obs": observe})
    t = df.groupby("tranche").agg(p_moyenne=("p", "mean"), frequence_reelle=("obs", "mean"),
                                  n=("obs", "size")).reset_index()
    return t[t["n"] >= 20]
