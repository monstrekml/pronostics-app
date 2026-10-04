"""
Récupération des données « fraîches » et mise au format des modèles.

Sources utilisées (toutes gratuites) :

- Football, historique (2015/16 → 2025/26) : dépôt GitHub `datasets/football-datasets`
  (données d'origine football-data.co.uk). Fichiers déjà présents dans data/foot/historique.
- Football, saison en cours (résultats + calendrier) : projet ouvert `openfootball`
  sur GitHub, mis à jour par des bénévoles (délai possible de quelques jours).
- Football, option : API football-data.org (clé gratuite) pour des résultats plus frais.
- NBA, historique (1947 → 2015) : FiveThirtyEight, fichier data/nba/historique_538.csv.gz.
- NBA, depuis 2015/16 + calendrier : API publique (non officielle) d'ESPN.

Chaque fonction `analyser_*` transforme une réponse brute (JSON) en DataFrame :
elles sont séparées du téléchargement pour pouvoir être testées hors ligne.
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from datetime import date, datetime, timedelta
from difflib import get_close_matches
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
DATA = RACINE / "data"

LIGUES = {
    "premier-league": {"nom": "Premier League", "openfootball": "en.1", "fdorg": "PL"},
    "la-liga": {"nom": "Liga", "openfootball": "es.1", "fdorg": "PD"},
    "ligue-1": {"nom": "Ligue 1", "openfootball": "fr.1", "fdorg": "FL1"},
}

URL_HISTO_FOOT = "https://raw.githubusercontent.com/datasets/football-datasets/main/datasets/{ligue}/season-{saison}.csv"
URL_OPENFOOTBALL = "https://raw.githubusercontent.com/openfootball/football.json/master/{saison}/{code}.json"
URL_FDORG = "https://api.football-data.org/v4/competitions/{code}/matches?season={annee}"
URL_ESPN_NBA = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={jour}"

COLONNES_RESULTATS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
COLONNES_CALENDRIER = ["Date", "Heure", "Journee", "HomeTeam", "AwayTeam"]

# Noms ESPN -> noms FiveThirtyEight (les autres franchises ont le même nom)
NBA_NOMS = {"76ers": "Sixers", "Trail Blazers": "Trailblazers"}
NBA_FRANCHISES = {
    "Bucks", "Bulls", "Cavaliers", "Celtics", "Clippers", "Grizzlies", "Hawks", "Heat", "Hornets",
    "Jazz", "Kings", "Knicks", "Lakers", "Magic", "Mavericks", "Nets", "Nuggets", "Pacers", "Pelicans",
    "Pistons", "Raptors", "Rockets", "Sixers", "Spurs", "Suns", "Thunder", "Timberwolves",
    "Trailblazers", "Warriors", "Wizards",
}
NBA_TYPES_RETENUS = {2: 0, 3: 1, 5: 0}   # type de saison ESPN -> drapeau playoffs (1 = présaison, exclue)


# ---------------------------------------------------------------- utilitaires
def telecharger(url: str, entetes: dict | None = None, essais: int = 3, delai: float = 2.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "pronostics-pedagogique/1.0", **(entetes or {})})
    for i in range(essais):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except Exception:
            if i == essais - 1:
                raise
            time.sleep(delai * (i + 1))
    raise RuntimeError("inaccessible")


def saison_foot(jour: date) -> str:
    """Saison de football contenant ce jour, au format openfootball : '2026-27'."""
    debut = jour.year if jour.month >= 7 else jour.year - 1
    return f"{debut}-{str(debut + 1)[2:]}"


def saison_courte(saison: str) -> str:
    """'2026-27' -> '2627' (format des fichiers historiques)."""
    return saison[2:4] + saison[5:7]


def saison_nba(jour: date) -> int:
    """Saison NBA identifiée par l'année de fin (convention FiveThirtyEight et ESPN)."""
    return jour.year + 1 if jour.month >= 8 else jour.year


# ---------------------------------------------------------------- noms d'équipes (foot)
def charger_correspondance() -> dict[str, str]:
    f = DATA / "foot" / "correspondance_equipes.csv"
    if not f.exists():
        return {}
    d = pd.read_csv(f)
    return dict(zip(d["nom_source"], d["nom_modele"]))


def nettoyer_nom(nom: str) -> str:
    """Nom lisible pour une équipe absente de l'historique (souvent un promu)."""
    n = re.sub(r"\b(FC|AFC|CF|SC|AC|SAD|CD|UD|RC|ES)\b", "", nom)
    n = re.sub(r"\b(18|19|20)\d\d\b", "", n)
    return re.sub(r"\s+", " ", n).strip() or nom


def normaliser_nom(nom: str, correspondance: dict[str, str], connus: set[str] | None = None) -> str:
    if nom in correspondance:
        return correspondance[nom]
    proches = get_close_matches(nom, list(correspondance), n=1, cutoff=0.88)
    if proches:
        return correspondance[proches[0]]
    if connus:
        proches = get_close_matches(nettoyer_nom(nom), list(connus), n=1, cutoff=0.88)
        if proches:
            return proches[0]
    return nettoyer_nom(nom)


# ---------------------------------------------------------------- openfootball
def analyser_openfootball(brut: dict, correspondance: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retourne (résultats joués, calendrier complet) au format des modèles."""
    resultats, calendrier = [], []
    for m in brut.get("matches", []):
        dom = normaliser_nom(m["team1"], correspondance)
        ext = normaliser_nom(m["team2"], correspondance)
        calendrier.append({"Date": m["date"], "Heure": m.get("time", ""), "Journee": m.get("round", ""),
                           "HomeTeam": dom, "AwayTeam": ext})
        score = m.get("score")
        ft = score.get("ft") if isinstance(score, dict) else None
        if ft and len(ft) == 2:
            resultats.append({"Date": m["date"], "HomeTeam": dom, "AwayTeam": ext,
                              "FTHG": int(ft[0]), "FTAG": int(ft[1])})
    return (pd.DataFrame(resultats, columns=COLONNES_RESULTATS),
            pd.DataFrame(calendrier, columns=COLONNES_CALENDRIER))


# ---------------------------------------------------------------- football-data.org (option)
def analyser_fdorg(brut: dict, correspondance: dict[str, str], connus: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    resultats, calendrier = [], []
    paris = ZoneInfo("Europe/Paris")
    for m in brut.get("matches", []):
        dt = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00")).astimezone(paris)
        dom = normaliser_nom(m["homeTeam"].get("name") or "", correspondance, connus)
        ext = normaliser_nom(m["awayTeam"].get("name") or "", correspondance, connus)
        calendrier.append({"Date": dt.strftime("%Y-%m-%d"), "Heure": dt.strftime("%H:%M"),
                           "Journee": f"Matchday {m.get('matchday', '')}", "HomeTeam": dom, "AwayTeam": ext})
        ft = (m.get("score") or {}).get("fullTime") or {}
        if m.get("status") == "FINISHED" and ft.get("home") is not None:
            resultats.append({"Date": dt.strftime("%Y-%m-%d"), "HomeTeam": dom, "AwayTeam": ext,
                              "FTHG": int(ft["home"]), "FTAG": int(ft["away"])})
    return (pd.DataFrame(resultats, columns=COLONNES_RESULTATS),
            pd.DataFrame(calendrier, columns=COLONNES_CALENDRIER))


def fusionner_resultats(base: pd.DataFrame, complement: pd.DataFrame) -> pd.DataFrame:
    """Ajoute à `base` les matchs de `complement` qu'elle ne contient pas encore
    (clé : équipe à domicile + équipe à l'extérieur, unique dans une saison)."""
    if complement.empty:
        return base
    cle = ["HomeTeam", "AwayTeam"]
    manquants = complement.merge(base[cle], on=cle, how="left", indicator=True)
    manquants = manquants[manquants["_merge"] == "left_only"].drop(columns="_merge")
    return pd.concat([base, manquants], ignore_index=True).sort_values("Date").reset_index(drop=True)


# ---------------------------------------------------------------- ESPN (NBA)
def _score(concurrent: dict) -> int:
    s = concurrent.get("score")
    if isinstance(s, dict):
        s = s.get("value", s.get("displayValue"))
    return int(float(s))


def analyser_espn(brut: dict) -> pd.DataFrame:
    """Un match NBA par ligne. Les matchs à venir ont des scores vides."""
    est = ZoneInfo("America/New_York")
    lignes = []
    for ev in brut.get("events", []):
        type_saison = (ev.get("season") or {}).get("type")
        if type_saison not in NBA_TYPES_RETENUS:
            continue
        comp = (ev.get("competitions") or [{}])[0]
        equipes = {c.get("homeAway"): c for c in comp.get("competitors", [])}
        if "home" not in equipes or "away" not in equipes:
            continue
        noms = {k: NBA_NOMS.get(v["team"].get("name"), v["team"].get("name")) for k, v in equipes.items()}
        if not (noms["home"] in NBA_FRANCHISES and noms["away"] in NBA_FRANCHISES):
            continue   # All-Star Game, équipes invitées…
        statut = ((comp.get("status") or ev.get("status") or {}).get("type") or {})
        termine = bool(statut.get("completed"))
        dt = datetime.fromisoformat(ev["date"].replace("Z", "+00:00")).astimezone(est)
        lignes.append({
            "id": ev.get("id"),
            "date": dt.strftime("%Y-%m-%d"),
            "heure_paris": dt.astimezone(ZoneInfo("Europe/Paris")).strftime("%d/%m %H:%M"),
            "saison": int((ev.get("season") or {}).get("year")),
            "playoffs": NBA_TYPES_RETENUS[type_saison],
            "neutre": int(bool(comp.get("neutralSite"))),
            "dom": noms["home"],
            "ext": noms["away"],
            "pts_dom": _score(equipes["home"]) if termine else None,
            "pts_ext": _score(equipes["away"]) if termine else None,
            "termine": int(termine),
        })
    return pd.DataFrame(lignes)


def jours(debut: date, fin: date):
    j = debut
    while j <= fin:
        yield j
        j += timedelta(days=1)


def recuperer_espn(debut: date, fin: date, travailleurs: int = 4) -> tuple[pd.DataFrame, list[str]]:
    """Télécharge tous les jours entre deux dates (requêtes parallèles, modérées)."""
    from concurrent.futures import ThreadPoolExecutor

    def un_jour(j: date):
        try:
            return analyser_espn(json.loads(telecharger(URL_ESPN_NBA.format(jour=j.strftime("%Y%m%d"))))), None
        except Exception as e:  # noqa: BLE001
            return None, f"{j}: {e}"

    morceaux, erreurs = [], []
    with ThreadPoolExecutor(travailleurs) as ex:
        for df, err in ex.map(un_jour, jours(debut, fin)):
            if err:
                erreurs.append(err)
            elif df is not None and len(df):
                morceaux.append(df)
    return (pd.concat(morceaux, ignore_index=True) if morceaux else pd.DataFrame()), erreurs
