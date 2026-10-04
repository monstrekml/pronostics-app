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


def telecharger_json_avec_quota(url: str) -> tuple[object, str | None]:
    """Comme `telecharger`, mais renvoie aussi le quota restant annoncé par l'API."""
    req = urllib.request.Request(url, headers={"User-Agent": "pronostics-pedagogique/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read()), r.headers.get("x-requests-remaining")


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


# ---------------------------------------------------------------- cotes (The Odds API, offre gratuite)
URL_ODDS = ("https://api.the-odds-api.com/v4/sports/{sport}/odds/"
            "?apiKey={cle}&regions=eu&markets=h2h&oddsFormat=decimal&dateFormat=iso")
ODDS_SPORTS = {
    "premier-league": "soccer_epl",
    "la-liga": "soccer_spain_la_liga",
    "ligue-1": "soccer_france_ligue_one",
    "nba": "basketball_nba",
}
COLONNES_COTES = ["sport", "id_cotes", "date", "debut_utc", "dom", "ext", "p_dom", "p_nul", "p_ext",
                  "cote_dom", "cote_nul", "cote_ext", "marge", "n_bookmakers", "releve_utc"]


def _simplifier(nom: str) -> str:
    import unicodedata
    n = unicodedata.normalize("NFKD", nom).encode("ascii", "ignore").decode().lower()
    n = re.sub(r"[^a-z0-9 ]", " ", n)
    n = re.sub(r"\b(fc|afc|cf|sc|ac|sad|cd|ud|rc|rcd|es|club|de|the|and|stade|real)\b", " ", n)
    return re.sub(r"\s+", " ", n).strip()


def associer_equipe(nom: str, candidats: set[str], correspondance: dict[str, str]) -> str | None:
    """Retrouve, parmi les équipes du championnat, celle qui correspond à un nom
    venant d'une autre source (« Paris Saint Germain » -> « Paris SG »)."""
    if nom in candidats:
        return nom
    if correspondance.get(nom) in candidats:
        return correspondance[nom]
    cible = _simplifier(nom)
    meilleur, score = None, 0.0
    from difflib import SequenceMatcher
    for cand in candidats:
        variantes = [cand] + [k for k, v in correspondance.items() if v == cand]
        for v in variantes:
            s = _simplifier(v)
            r = SequenceMatcher(None, cible, s).ratio()
            if s and (s in cible or cible in s):
                r = max(r, 0.9)
            if r > score:
                meilleur, score = cand, r
    return meilleur if score >= 0.75 else None


def _nba_depuis_nom_complet(nom: str) -> str | None:
    """« Golden State Warriors » -> « Warriors », « Philadelphia 76ers » -> « Sixers »."""
    vers_espn = {v: k for k, v in NBA_NOMS.items()}                  # Sixers -> 76ers
    for franchise in sorted(NBA_FRANCHISES, key=lambda f: -len(vers_espn.get(f, f))):
        if nom.endswith(vers_espn.get(franchise, franchise)):
            return franchise
    return None


def analyser_cotes(brut: list, ligue: str, candidats: set[str] | None = None,
                   correspondance: dict[str, str] | None = None, releve: str = "") -> tuple[pd.DataFrame, list[str]]:
    """Une ligne par match : probabilités implicites moyennes (marge retirée) et meilleures cotes.

    Retourne aussi la liste des noms d'équipes qui n'ont pas pu être associés."""
    lignes, inconnus = [], []
    est_nba = ligue == "nba"
    fuseau = ZoneInfo("America/New_York" if est_nba else "Europe/Paris")
    for ev in brut or []:
        if est_nba:
            dom, ext = _nba_depuis_nom_complet(ev["home_team"]), _nba_depuis_nom_complet(ev["away_team"])
        else:
            dom = associer_equipe(ev["home_team"], candidats or set(), correspondance or {})
            ext = associer_equipe(ev["away_team"], candidats or set(), correspondance or {})
        if not dom or not ext or dom == ext:
            inconnus += [n for n, v in ((ev["home_team"], dom), (ev["away_team"], ext)) if not v]
            continue
        probas, meilleures = [], {"dom": 0.0, "nul": 0.0, "ext": 0.0}
        for bk in ev.get("bookmakers", []):
            marche = next((m for m in bk.get("markets", []) if m.get("key") == "h2h"), None)
            if not marche:
                continue
            prix = {}
            for o in marche.get("outcomes", []):
                if o["name"] == ev["home_team"]:
                    prix["dom"] = float(o["price"])
                elif o["name"] == ev["away_team"]:
                    prix["ext"] = float(o["price"])
                elif o["name"].lower() == "draw":
                    prix["nul"] = float(o["price"])
            attendues = ("dom", "ext") if est_nba else ("dom", "nul", "ext")
            if not all(k in prix and prix[k] > 1 for k in attendues):
                continue
            impl = {k: 1 / prix[k] for k in attendues}
            total = sum(impl.values())
            probas.append({k: v / total for k, v in impl.items()} | {"marge": total - 1})
            for k in attendues:
                meilleures[k] = max(meilleures[k], prix[k])
        if not probas:
            continue
        moy = lambda k: sum(p.get(k, 0.0) for p in probas) / len(probas)  # noqa: E731
        dt = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")).astimezone(fuseau)
        lignes.append({
            "sport": ligue, "id_cotes": ev.get("id"), "date": dt.strftime("%Y-%m-%d"),
            "debut_utc": ev["commence_time"], "dom": dom, "ext": ext,
            "p_dom": moy("dom"), "p_nul": None if est_nba else moy("nul"), "p_ext": moy("ext"),
            "cote_dom": meilleures["dom"], "cote_nul": None if est_nba else meilleures["nul"],
            "cote_ext": meilleures["ext"], "marge": moy("marge"), "n_bookmakers": len(probas),
            "releve_utc": releve,
        })
    return pd.DataFrame(lignes, columns=COLONNES_COTES), inconnus


def fusionner_cotes(archive: pd.DataFrame, nouvelles: pd.DataFrame, maintenant_utc: datetime) -> pd.DataFrame:
    """Garde le relevé le plus récent de chaque match tant qu'il n'a pas commencé :
    après le coup d'envoi, la dernière cote archivée reste figée (≈ cote de clôture)."""
    # les matchs déjà commencés sont ignorés : leurs cotes « en direct » ne sont plus des cotes d'avant-match
    if not nouvelles.empty:
        debut = pd.to_datetime(nouvelles["debut_utc"], utc=True)
        nouvelles = nouvelles[debut > pd.Timestamp(maintenant_utc)]
    if nouvelles.empty:
        return archive
    cle = ["sport", "date", "dom", "ext"]
    if archive.empty:
        return nouvelles.copy()
    tout = pd.concat([archive, nouvelles], ignore_index=True)
    return (tout.sort_values("releve_utc").drop_duplicates(cle, keep="last")
            .sort_values(["sport", "date", "dom"]).reset_index(drop=True))
