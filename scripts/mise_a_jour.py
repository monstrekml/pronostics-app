"""
Met à jour toutes les données de l'application. Lancé chaque jour par la GitHub Action
(.github/workflows/mise_a_jour.yml), ou à la main :

    python scripts/mise_a_jour.py              # foot + NBA
    python scripts/mise_a_jour.py --foot       # foot seulement
    python scripts/mise_a_jour.py --nba        # NBA seulement

Option : définir la variable d'environnement FOOTBALL_DATA_API_KEY (clé gratuite sur
football-data.org) pour compléter les résultats d'openfootball, parfois en retard.

Le script ne s'arrête jamais sur une source en panne : l'erreur est notée dans
data/derniere_mise_a_jour.json et l'application affiche les dernières données valides.
"""
import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sources import (  # noqa: E402
    DATA, LIGUES, URL_ESPN_NBA, URL_FDORG, URL_HISTO_FOOT, URL_OPENFOOTBALL, analyser_fdorg, analyser_openfootball,
    charger_correspondance, fusionner_resultats, recuperer_espn, saison_courte, saison_foot, telecharger,
)

ETAT = DATA / "derniere_mise_a_jour.json"
DEBUT_ESPN = date(2015, 10, 1)   # l'historique FiveThirtyEight s'arrête à la saison 2014-15


def lire_etat() -> dict:
    return json.loads(ETAT.read_text()) if ETAT.exists() else {}


def maj_foot(etat: dict) -> None:
    aujourdhui = date.today()
    saison = saison_foot(aujourdhui)
    corr = charger_correspondance()
    cle_api = os.environ.get("FOOTBALL_DATA_API_KEY")
    dossier_histo = DATA / "foot" / "historique"
    dossier_of = DATA / "foot" / "openfootball"
    connus = set(corr.values())

    for ligue, info in LIGUES.items():
        rapport = {"heure": maintenant()}
        try:
            # 1. Saison précédente : le jeu de données historique complet est-il paru ?
            prec = saison_foot(aujourdhui - timedelta(days=365))
            f = dossier_histo / f"{ligue}_{saison_courte(prec)}.csv"
            if not f.exists():
                try:
                    f.write_bytes(telecharger(URL_HISTO_FOOT.format(ligue=ligue, saison=saison_courte(prec))))
                    rapport["historique_ajoute"] = prec
                except Exception:  # noqa: BLE001
                    pass   # pas encore publié : on garde la version openfootball

            # 2. Saison en cours : openfootball
            brut = json.loads(telecharger(URL_OPENFOOTBALL.format(saison=saison, code=info["openfootball"])))
            res, cal = analyser_openfootball(brut, corr)
            rapport["source_resultats"] = "openfootball"

            # 3. Option : football-data.org, souvent plus à jour
            if cle_api:
                try:
                    brut2 = json.loads(telecharger(URL_FDORG.format(code=info["fdorg"], annee=saison[:4]),
                                                   entetes={"X-Auth-Token": cle_api}))
                    res2, cal2 = analyser_fdorg(brut2, corr, connus)
                    avant = len(res)
                    res = fusionner_resultats(res, res2)
                    rapport["ajouts_football_data_org"] = len(res) - avant
                    rapport["source_resultats"] = "openfootball + football-data.org"
                    if len(cal2) >= len(cal):
                        cal = cal2
                except Exception as e:  # noqa: BLE001
                    rapport["erreur_football_data_org"] = str(e)[:200]

            sc = saison_courte(saison)
            res.to_csv(dossier_of / f"{ligue}_{sc}_resultats.csv", index=False)
            cal.to_csv(dossier_of / f"{ligue}_{sc}_calendrier.csv", index=False)
            rapport.update(matchs_joues=len(res), matchs_au_calendrier=len(cal),
                           dernier_resultat=res["Date"].max() if len(res) else None, statut="ok")
        except Exception as e:  # noqa: BLE001
            rapport.update(statut="erreur", erreur=str(e)[:300])
        etat.setdefault("foot", {})[ligue] = rapport
        print(f"foot {ligue} :", rapport)


def maj_nba(etat: dict, jours_avant: int = 14) -> None:
    f = DATA / "nba" / "espn_matchs.csv"
    rapport = {"heure": maintenant()}
    try:
        if f.exists():
            ancien = pd.read_csv(f)
            joues = ancien[ancien["termine"] == 1]
            debut = (pd.to_datetime(joues["date"]).max().date() - timedelta(days=3)) if len(joues) else DEBUT_ESPN
            ancien = ancien[pd.to_datetime(ancien["date"]).dt.date < debut]
        else:
            ancien, debut = pd.DataFrame(), DEBUT_ESPN
            print("Premier lancement : récupération de toutes les saisons NBA depuis 2015 (quelques minutes)…")
        fin = date.today() + timedelta(days=jours_avant)
        # Test rapide avant de lancer des centaines de requêtes
        telecharger(URL_ESPN_NBA.format(jour=date.today().strftime("%Y%m%d")), essais=2)
        nouveau, erreurs = recuperer_espn(debut, fin)
        if len(erreurs) > 0.2 * max((fin - debut).days, 1):
            raise RuntimeError(f"{len(erreurs)} jours en échec, ex. {erreurs[:2]}")
        tout = pd.concat([ancien, nouveau], ignore_index=True)
        if len(tout):
            tout = tout.drop_duplicates("id", keep="last").sort_values(["date", "dom"])
            tout.to_csv(f, index=False)
        rapport.update(statut="ok", jours_interroges=(fin - debut).days + 1, jours_en_echec=len(erreurs),
                       matchs_total=len(tout), dernier_resultat=tout.loc[tout.termine == 1, "date"].max() if len(tout) else None)
    except Exception as e:  # noqa: BLE001
        rapport.update(statut="erreur", erreur=str(e)[:300])
    etat["nba"] = rapport
    print("nba :", rapport)


def maintenant() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--foot", action="store_true")
    ap.add_argument("--nba", action="store_true")
    args = ap.parse_args()
    tout = not (args.foot or args.nba)
    etat = lire_etat()
    if args.foot or tout:
        maj_foot(etat)
    if args.nba or tout:
        maj_nba(etat)
    etat["derniere_execution"] = maintenant()
    ETAT.write_text(json.dumps(etat, indent=2, ensure_ascii=False, default=str))
    statuts = [v.get("statut") for v in etat.get("foot", {}).values()] + [etat.get("nba", {}).get("statut")]
    if statuts and all(s == "erreur" for s in statuts if s):
        sys.exit("Toutes les sources sont en échec.")


if __name__ == "__main__":
    main()
