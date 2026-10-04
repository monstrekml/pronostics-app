"""
Construit la table de correspondance entre les noms d'équipes d'openfootball
(« Paris Saint-Germain FC ») et ceux de l'historique (« Paris SG »).

Méthode : on aligne les matchs des deux sources sur les saisons communes
(même date + même score) et chaque appariement compte comme un « vote ».
Le nom historique qui récolte le plus de votes l'emporte. Quelques équipes,
absentes des saisons communes, sont ajoutées à la main.

À relancer seulement si de nouvelles équipes posent problème :
    python scripts/construire_correspondance.py
"""
import collections
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sources import DATA, LIGUES, URL_OPENFOOTBALL, telecharger  # noqa: E402

SAISONS = [f"{a}-{str(a + 1)[2:]}" for a in range(2015, 2026)]
MANUEL = {
    "Hull City AFC": "Hull",
    "ES Troyes AC": "Troyes",
    "RC Deportivo La Coruña": "La Coruna",
    "Real Racing Club de Santander": "Racing Santander",
    "Coventry City FC": "Coventry",
    "Le Mans FC": "Le Mans",
}


def main() -> None:
    votes = collections.defaultdict(collections.Counter)
    for ligue, info in LIGUES.items():
        for saison in SAISONS:
            histo = DATA / "foot" / "historique" / f"{ligue}_{saison[2:4]}{saison[5:7]}.csv"
            if not histo.exists():
                continue
            fd = pd.read_csv(histo)
            fd["Date"] = pd.to_datetime(fd["Date"]).dt.strftime("%Y-%m-%d")
            groupes = fd.groupby(["Date", "FTHG", "FTAG"]).groups
            try:
                brut = json.loads(telecharger(URL_OPENFOOTBALL.format(saison=saison, code=info["openfootball"])))
            except Exception as e:  # noqa: BLE001
                print(f"  {ligue} {saison} : indisponible ({e})")
                continue
            for m in brut.get("matches", []):
                ft = (m.get("score") or {}).get("ft") if isinstance(m.get("score"), dict) else None
                if not ft:
                    continue
                for i in groupes.get((m["date"], ft[0], ft[1]), []):
                    votes[m["team1"]][fd.at[i, "HomeTeam"]] += 1
                    votes[m["team2"]][fd.at[i, "AwayTeam"]] += 1
            print(f"  {ligue} {saison} : ok")

    lignes = [{"nom_source": k, "nom_modele": c.most_common(1)[0][0]}
              for k, c in votes.items() if c.most_common(1)[0][1] >= 4]
    lignes += [{"nom_source": k, "nom_modele": v} for k, v in MANUEL.items()]
    df = pd.DataFrame(lignes).drop_duplicates("nom_source", keep="last").sort_values("nom_source")
    sortie = DATA / "foot" / "correspondance_equipes.csv"
    df.to_csv(sortie, index=False)
    print(f"{len(df)} correspondances écrites dans {sortie}")


if __name__ == "__main__":
    main()
