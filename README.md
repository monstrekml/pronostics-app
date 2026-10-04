# Pronostics foot & NBA : application Streamlit mise à jour automatiquement

Application web qui affiche les **probabilités** des prochains matchs de Premier League, Liga,
Ligue 1 et NBA. Les données se mettent à jour toutes seules deux fois par jour, gratuitement.

> Ce sont des probabilités, pas des certitudes. Sur 11 000 matchs de foot testés, le modèle
> trouve le bon résultat 1N2 environ 53 fois sur 100 ; en NBA, environ 67 fois sur 100.
> L'onglet « Fiabilité » de l'app le montre en détail.

## Ce que fait l'app

| Onglet | Football | NBA |
|---|---|---|
| Prochains matchs | 1/N/2, +2,5 buts, score le plus probable | probabilité de victoire, écart prévu |
| Simulateur | n'importe quelle affiche, grille des scores exacts, comparaison avec des cotes | idem, avec option terrain neutre |
| Forces / Classement | attaque et défense de chaque équipe | notes Elo des 30 franchises |
| Bilan de la saison | pronostics d'avant-match comparés aux vrais résultats | idem |
| Fiabilité | backtest sur 10 saisons, calibration, bilan face au marché | backtest depuis 2005, bilan face au marché |

Avec la clé The Odds API, l'onglet « Prochains matchs » propose trois jeux de probabilités :
**Modèle** (notre calcul), **Marché** (cotes moyennes des bookmakers, marge retirée) et
**Combiné** (mélange des deux, dont le poids est réglé automatiquement sur les matchs archivés).

## Comment les données restent à jour

```
GitHub Action (06h et 18h UTC)
   ├─ openfootball (GitHub)        → résultats + calendrier foot de la saison en cours
   ├─ football-data.org (option)   → résultats foot plus frais
   ├─ The Odds API (option)        → cotes 1N2 des prochains matchs, archivées
   └─ API ESPN                     → résultats + calendrier NBA
        ↓  commit des fichiers dans data/
Streamlit Community Cloud redémarre l'app → nouvelles probabilités
```

Le modèle est réentraîné à chaque ouverture de l'app avec les derniers résultats (calcul en
une fraction de seconde, mis en cache une heure).

**Délais à connaître**
- openfootball est tenu par des bénévoles : les résultats peuvent arriver avec quelques jours
  de retard. L'app le signale (« X matchs déjà joués n'ont pas encore de résultat »). Pour des
  résultats plus frais, ajoute la clé gratuite football-data.org (étape 4 ci-dessous).
- L'API ESPN est publique mais non officielle : elle peut changer sans prévenir. Si elle tombe,
  l'app continue d'afficher les dernières données et le signale dans la barre latérale.
- Ce n'est pas du « temps réel » pendant les matchs : les probabilités sont recalculées après
  chaque journée, ce qui suffit pour ce type de modèle.

---

## Déploiement gratuit, pas à pas

### 1. Créer le dépôt GitHub

Le dépôt doit être **public** : les GitHub Actions y sont gratuites et illimitées.

```bash
cd pronostics_app
git init -b main
git add .
git commit -m "Première version de l'app de pronostics"
git remote add origin https://github.com/<ton-compte>/pronostics-app.git
git push -u origin main
```

### 2. Autoriser l'Action à enregistrer les données

Sur GitHub : **Settings → Actions → General → Workflow permissions** → coche
**Read and write permissions** → Save.

### 3. Lancer la première mise à jour

Onglet **Actions** → **Mise à jour des données** → **Run workflow** (coche « Relancer aussi
le backtest »).

Le premier lancement récupère toutes les saisons NBA depuis 2015 : compte 5 à 10 minutes.
Les suivants prennent moins d'une minute.

### 4. (Facultatif) Clé football-data.org

1. Crée un compte gratuit sur football-data.org et récupère ta clé API.
2. GitHub : **Settings → Secrets and variables → Actions → New repository secret**,
   nom `FOOTBALL_DATA_API_KEY`, valeur : ta clé.

### 4 bis. (Facultatif) Cotes des bookmakers : clé The Odds API

1. Crée un compte gratuit sur the-odds-api.com (offre gratuite : 500 crédits par mois).
2. Ajoute le secret `ODDS_API_KEY` comme à l'étape 4.

Chaque mise à jour consomme 1 crédit par championnat (4 en tout), soit environ 250 crédits par
mois avec deux mises à jour par jour : on reste dans l'offre gratuite. Évite de multiplier les
lancements manuels. Le nombre de crédits restants s'affiche dans la barre latérale de l'app.

L'offre gratuite ne donne pas l'historique des cotes : l'app **archive elle-même** chaque relevé
(`data/cotes/cotes.csv`). Seules les cotes relevées avant le coup d'envoi sont gardées. Cet
archivage alimente, dans l'onglet Fiabilité, le bilan « modèle face au marché ».

### 5. Publier l'app sur Streamlit Community Cloud

1. Va sur share.streamlit.io et connecte-toi avec ton compte GitHub.
2. **Create app** → choisis le dépôt, la branche `main` et le fichier `app.py`.
3. Choisis une adresse (par exemple `pronostics-alphonsine.streamlit.app`) → **Deploy**.

C'est en ligne. À chaque commit de l'Action, l'app se met à jour toute seule.

**Bon à savoir sur l'offre gratuite**
- Une app sans visite pendant un moment se met en veille ; elle redémarre en quelques
  secondes à la visite suivante.
- GitHub suspend les tâches planifiées d'un dépôt sans activité pendant 60 jours. Les commits
  de l'Action comptent comme activité, donc ça ne devrait pas arriver ; sinon, il suffit de
  les réactiver dans l'onglet Actions.

---

## Lancer en local

```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt pytest

python scripts/mise_a_jour.py      # récupérer les données du jour
python scripts/backtest.py         # recalculer l'onglet Fiabilité (~20 s)
python -m pytest -q                # tests hors ligne
streamlit run app.py               # ouvre http://localhost:8501
```

## Organisation du code

```
app.py                         l'application Streamlit
src/
  foot_poisson.py              modèle foot (Poisson + Dixon-Coles, pondération temporelle)
  basket_elo.py                modèle NBA (Elo avec marge de victoire)
  sources.py                   téléchargement et lecture des sources (openfootball, ESPN, cotes…)
  marche.py                    probabilités du marché, combinaison et bilan modèle vs marché
  donnees.py                   assemblage historique + saison en cours
  metriques.py                 précision, log loss, Brier, RPS, calibration
scripts/
  mise_a_jour.py               lancé par l'Action : nouvelles données
  backtest.py                  évaluation des modèles sur le passé
  construire_correspondance.py noms d'équipes openfootball → noms de l'historique
data/
  foot/historique/             saisons 2015/16 → 2025/26 (football-data.co.uk via GitHub)
  foot/openfootball/           saison en cours (mise à jour automatique)
  foot/correspondance_equipes.csv
  nba/historique_538.csv.gz    NBA 1947 → 2015 (FiveThirtyEight)
  nba/espn_matchs.csv          NBA depuis 2015/16 (créé par la première mise à jour)
  cotes/cotes.csv              cotes d'avant-match archivées (créé avec la clé The Odds API)
  derniere_mise_a_jour.json    état de la dernière mise à jour (affiché dans l'app)
resultats/                     tableaux de l'onglet Fiabilité
tests/                         tests des parseurs avec des échantillons de réponses
.github/workflows/             la mise à jour automatique
```

## Problèmes fréquents

| Symptôme | Cause probable | Solution |
|---|---|---|
| L'Action échoue à l'étape « Enregistrer » | Permissions en lecture seule | Étape 2 ci-dessus |
| L'app NBA affiche des notes de 2015 | Première mise à jour pas encore lancée | Étape 3 |
| Une nouvelle équipe promue a un nom bizarre | Nom absent de la table de correspondance | L'ajouter dans `data/foot/correspondance_equipes.csv` |
| Encadré « dernière récupération a échoué » | Source momentanément indisponible | Rien à faire, la prochaine mise à jour réessaie |

---

*Projet pédagogique. Les paris sportifs comportent un risque de perte et d'addiction ; cette
application n'est pas un outil de mise.*
