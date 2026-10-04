"""
Assemblage des données pour l'application et les backtests.

Foot : pour chaque saison, on prend le fichier historique complet s'il existe,
sinon la version openfootball (saison en cours).
NBA  : historique FiveThirtyEight (jusqu'en 2015) + matchs ESPN (depuis 2015/16).
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from sources import DATA

DOSSIER_HISTO = DATA / "foot" / "historique"
DOSSIER_OF = DATA / "foot" / "openfootball"


def etat_mise_a_jour() -> dict:
    f = DATA / "derniere_mise_a_jour.json"
    return json.loads(f.read_text()) if f.exists() else {}


def _resultat_1n2(df: pd.DataFrame) -> pd.Series:
    return pd.Series(np.select([df.FTHG > df.FTAG, df.FTHG == df.FTAG], [0, 1], 2), index=df.index)


def charger_resultats_foot(ligue: str) -> pd.DataFrame:
    parts = {}
    for f in DOSSIER_HISTO.glob(f"{ligue}_*.csv"):
        d = pd.read_csv(f, usecols=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        parts[f.stem.split("_")[-1]] = d
    for f in DOSSIER_OF.glob(f"{ligue}_*_resultats.csv"):
        saison = f.stem.split("_")[1]
        if saison not in parts:          # l'historique complet est prioritaire
            parts[saison] = pd.read_csv(f)
    lignes = []
    for saison, d in parts.items():
        d = d.copy()
        d["Saison"] = saison
        lignes.append(d)
    df = pd.concat(lignes, ignore_index=True).dropna(subset=["FTHG", "FTAG"])
    df["Date"] = pd.to_datetime(df["Date"])
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    df["Resultat"] = _resultat_1n2(df)
    return df.sort_values("Date").reset_index(drop=True)


def saison_en_cours_foot(ligue: str) -> str | None:
    fichiers = sorted(DOSSIER_OF.glob(f"{ligue}_*_calendrier.csv"))
    return fichiers[-1].stem.split("_")[1] if fichiers else None


def charger_calendrier_foot(ligue: str) -> pd.DataFrame:
    """Calendrier complet de la saison en cours, avec le résultat quand il est connu."""
    saison = saison_en_cours_foot(ligue)
    if saison is None:
        return pd.DataFrame()
    cal = pd.read_csv(DOSSIER_OF / f"{ligue}_{saison}_calendrier.csv", keep_default_na=False)
    res = charger_resultats_foot(ligue)
    res = res[res["Saison"] == saison][["HomeTeam", "AwayTeam", "FTHG", "FTAG"]]
    cal = cal.merge(res, on=["HomeTeam", "AwayTeam"], how="left")
    cal["Date"] = pd.to_datetime(cal["Date"])
    cal["Joue"] = cal["FTHG"].notna()
    return cal.sort_values(["Date", "Heure"]).reset_index(drop=True)


def charger_nba() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retourne (matchs joués au format du modèle, matchs à venir)."""
    histo = pd.read_csv(DATA / "nba" / "historique_538.csv.gz")
    f = DATA / "nba" / "espn_matchs.csv"
    a_venir = pd.DataFrame()
    if f.exists():
        espn = pd.read_csv(f)
        joues = espn[espn["termine"] == 1].copy()
        joues["p538_dom"] = np.nan
        histo = pd.concat([histo, joues[histo.columns]], ignore_index=True)
        a_venir = espn[(espn["termine"] == 0) & (pd.to_datetime(espn["date"]).dt.date >= date.today())].copy()
    histo["date"] = pd.to_datetime(histo["date"])
    histo["pts_dom"] = histo["pts_dom"].astype(int)
    histo["pts_ext"] = histo["pts_ext"].astype(int)
    histo["victoire_dom"] = (histo["pts_dom"] > histo["pts_ext"]).astype(int)
    histo["marge_dom"] = histo["pts_dom"] - histo["pts_ext"]
    return histo.sort_values(["date", "dom"]).reset_index(drop=True), a_venir


def date_fichier(chemin: Path) -> float:
    return chemin.stat().st_mtime if chemin.exists() else 0.0


def charger_cotes(sport: str) -> pd.DataFrame:
    """Archive des cotes d'avant-match pour un championnat (« ligue-1 », « nba »…)."""
    f = DATA / "cotes" / "cotes.csv"
    if not f.exists():
        return pd.DataFrame()
    d = pd.read_csv(f)
    return d[d["sport"] == sport].reset_index(drop=True)
