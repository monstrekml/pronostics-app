"""
Application Streamlit : probabilités de résultats pour le football et la NBA.

Lancer en local :
    streamlit run app.py
"""
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from basket_elo import EloNBA                                          # noqa: E402
from donnees import (                                                  # noqa: E402
    charger_calendrier_foot, charger_nba, charger_resultats_foot, date_fichier, etat_mise_a_jour,
    saison_en_cours_foot,
)
from foot_poisson import ModeleFoot                                    # noqa: E402
from metriques import log_loss                                         # noqa: E402
from sources import DATA, LIGUES, RACINE                               # noqa: E402

st.set_page_config(page_title="Pronostics foot & NBA", page_icon="⚽", layout="wide")

RESULTATS = RACINE / "resultats"
FENETRE_JOURS = 730
JOURS = ["lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."]
POURCENT = st.column_config.ProgressColumn(format="%.0f %%", min_value=0, max_value=100, width="small")


# =============================================================== données (en cache)
def cle_donnees() -> float:
    """Change dès que la mise à jour quotidienne a écrit de nouveaux fichiers."""
    return date_fichier(DATA / "derniere_mise_a_jour.json")


@st.cache_data(ttl=3600, show_spinner=False)
def foot_donnees(ligue: str, cle: float):
    return charger_resultats_foot(ligue), charger_calendrier_foot(ligue)


@st.cache_resource(ttl=3600, show_spinner="Entraînement du modèle…")
def foot_modele(ligue: str, jour: date, cle: float) -> ModeleFoot:
    res, _ = foot_donnees(ligue, cle)
    t = pd.Timestamp(jour)
    histo = res[(res["Date"] < t) & (res["Date"] >= t - pd.Timedelta(days=FENETRE_JOURS))]
    return ModeleFoot().ajuster(histo, date_ref=t)


@st.cache_data(ttl=3600, show_spinner="Calcul du bilan de la saison…")
def foot_bilan_saison(ligue: str, cle: float) -> pd.DataFrame:
    """Pronostics « d'avant-match » pour chaque match déjà joué cette saison."""
    res, _ = foot_donnees(ligue, cle)
    saison = saison_en_cours_foot(ligue)
    jouees = res[res["Saison"] == saison]
    if jouees.empty:
        return pd.DataFrame()
    lignes = []
    t = jouees["Date"].min().normalize()
    while t <= jouees["Date"].max():
        bloc = jouees[(jouees["Date"] >= t) & (jouees["Date"] < t + pd.Timedelta(days=7))]
        if len(bloc):
            histo = res[(res["Date"] < t) & (res["Date"] >= t - pd.Timedelta(days=FENETRE_JOURS))]
            m = ModeleFoot().ajuster(histo, date_ref=t)
            for r in bloc.itertuples():
                p = m.predire(r.HomeTeam, r.AwayTeam)
                lignes.append({"Date": r.Date, "Domicile": r.HomeTeam, "Extérieur": r.AwayTeam,
                               "Score": f"{r.FTHG}-{r.FTAG}", "Resultat": r.Resultat,
                               "p1": p["p_dom"], "pN": p["p_nul"], "p2": p["p_ext"]})
        t += pd.Timedelta(days=7)
    return pd.DataFrame(lignes)


@st.cache_data(ttl=3600, show_spinner="Calcul des notes Elo…")
def nba_donnees(cle: float):
    matchs, a_venir = charger_nba()
    elo = EloNBA()
    res, notes = elo.parcourir(matchs)
    return res, notes, a_venir


@st.cache_data(ttl=3600, show_spinner=False)
def lire_resultat(nom: str) -> pd.DataFrame | None:
    f = RESULTATS / nom
    return pd.read_csv(f) if f.exists() else None


# =============================================================== composants
def bandeau_avertissement() -> None:
    st.caption("Ces chiffres sont des **probabilités**, jamais des certitudes. Même quand le modèle est sûr "
               "à 80 %, il se trompe environ une fois sur six. Projet pédagogique : ce n'est pas un conseil de pari.")


def comparer_cotes(probas: list[float], libelles: list[str], cle: str) -> None:
    with st.expander("Comparer avec les cotes d'un bookmaker"):
        cols = st.columns(len(libelles))
        cotes = [c.number_input(f"Cote {lib}", min_value=1.01, value=None, step=0.05, key=f"{cle}_{lib}",
                                placeholder="ex. 2.10") for c, lib in zip(cols, libelles)]
        if all(cotes):
            impl = 1 / np.array(cotes)
            marge = impl.sum() - 1
            df = pd.DataFrame({
                "Issue": libelles, "Cote": cotes,
                "Probabilité bookmaker": impl / impl.sum() * 100,
                "Probabilité modèle": np.array(probas) * 100,
                "Espérance (%)": (np.array(probas) * np.array(cotes) - 1) * 100,
            })
            st.dataframe(df, hide_index=True, column_config={
                "Probabilité bookmaker": POURCENT, "Probabilité modèle": POURCENT,
                "Espérance (%)": st.column_config.NumberColumn(format="%+.1f")})
            st.caption(f"Marge du bookmaker : {marge:.1%}. Une espérance positive veut seulement dire que le modèle "
                       "et le bookmaker ne sont pas d'accord ; le plus souvent, c'est le bookmaker qui a raison, "
                       "car il connaît les blessures, les compositions et l'argent misé.")


def graphique_calibration(df: pd.DataFrame, couleur: str | None = None) -> alt.Chart:
    base = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color="#999").encode(
        x="x", y="y")
    enc = dict(x=alt.X("p_moyenne:Q", title="Probabilité annoncée", scale=alt.Scale(domain=[0, 1])),
               y=alt.Y("frequence_reelle:Q", title="Fréquence observée", scale=alt.Scale(domain=[0, 1])),
               tooltip=[alt.Tooltip("p_moyenne:Q", format=".0%"), alt.Tooltip("frequence_reelle:Q", format=".0%"), "n:Q"])
    if couleur:
        enc["color"] = alt.Color(f"{couleur}:N", title=None, legend=alt.Legend(orient="bottom"))
    points = alt.Chart(df).mark_line(point=True).encode(**enc)
    return (base + points).properties(height=320)


def fiabilite(prefixe: str) -> None:
    met = lire_resultat(f"{prefixe}_metriques.csv")
    if met is None:
        st.info("Les résultats du backtest ne sont pas encore calculés (lancer `python scripts/backtest.py`).")
        return
    st.markdown("Le modèle a été testé sur des milliers de matchs passés, **sans jamais voir le résultat à "
                "l'avance**. Voici ce qu'il vaut vraiment.")
    colonnes = {"precision": st.column_config.NumberColumn("Bons pronostics", format="%.1f %%"),
                "log_loss": st.column_config.NumberColumn("Log loss ↓", format="%.3f"),
                "brier": st.column_config.NumberColumn("Brier ↓", format="%.3f"),
                "rps": st.column_config.NumberColumn("RPS ↓", format="%.3f"),
                "n_matchs": st.column_config.NumberColumn("Matchs"), "modele": "Modèle", "ligue": "Ligue",
                "periode": "Période"}
    met = met.assign(precision=met["precision"] * 100)
    st.dataframe(met, hide_index=True, column_config=colonnes)
    st.caption("↓ = plus c'est bas, mieux c'est. La référence naïve sert de point de comparaison : "
               "un modèle utile doit la battre nettement.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Quand le modèle est sûr de lui, a-t-il raison ?**")
        conf = lire_resultat(f"{prefixe}_confiance.csv")
        conf = conf.assign(confiance_min=(conf["confiance_min"] * 100).round().astype(int).astype(str) + " % et plus",
                           part_des_matchs=conf["part_des_matchs"] * 100, precision=conf["precision"] * 100)
        st.dataframe(conf, hide_index=True, column_config={
            "confiance_min": "Confiance du modèle", "part_des_matchs": st.column_config.NumberColumn("Part des matchs", format="%.0f %%"),
            "n_matchs": "Matchs", "precision": st.column_config.NumberColumn("Bons pronostics", format="%.1f %%")})
    with c2:
        st.markdown("**Calibration** : quand il annonce 60 %, ça arrive-t-il 6 fois sur 10 ?")
        cal = lire_resultat(f"{prefixe}_calibration.csv")
        st.altair_chart(graphique_calibration(cal, "issue" if "issue" in cal else None), width="stretch")

    sais = lire_resultat(f"{prefixe}_precision_saison.csv")
    if sais is not None:
        st.markdown("**Taux de bons pronostics par saison**")
        sais = sais.assign(precision=sais["precision"] * 100)
        x = "Saison:N" if "Saison" in sais else "saison:O"
        enc = dict(x=alt.X(x, title=None), y=alt.Y("precision:Q", title="Bons pronostics (%)", scale=alt.Scale(zero=False)))
        if "Ligue" in sais:
            enc["color"] = alt.Color("Ligue:N", legend=alt.Legend(orient="bottom", title=None))
        st.altair_chart(alt.Chart(sais).mark_line(point=True).encode(**enc).properties(height=280),
                        width="stretch")


# =============================================================== page football
def page_foot(ligue: str) -> None:
    cle = cle_donnees()
    res, cal = foot_donnees(ligue, cle)
    aujourdhui = date.today()
    modele = foot_modele(ligue, aujourdhui, cle)
    nom = LIGUES[ligue]["nom"]
    st.header(f"⚽ {nom}")
    bandeau_avertissement()

    onglets = st.tabs(["Prochains matchs", "Simulateur", "Forces des équipes", "Bilan de la saison", "Fiabilité"])

    # ---- prochains matchs
    with onglets[0]:
        if cal.empty:
            st.info("Pas de calendrier disponible pour la saison en cours.")
        else:
            dernier = res["Date"].max().date()
            en_retard = cal[(~cal["Joue"]) & (cal["Date"].dt.date < aujourdhui)]
            if len(en_retard):
                st.warning(f"{len(en_retard)} match(s) déjà joué(s) n'ont pas encore de résultat dans la source "
                           f"(dernier résultat connu : {dernier:%d/%m/%Y}). Le modèle les intégrera dès leur publication.")
            horizon = st.select_slider("Horizon", options=[7, 14, 30, 60], value=14, format_func=lambda j: f"{j} jours")
            a_venir = cal[(~cal["Joue"]) & (cal["Date"].dt.date >= aujourdhui) &
                          (cal["Date"].dt.date <= aujourdhui + timedelta(days=horizon))]
            if a_venir.empty:
                st.info("Aucun match prévu sur cette période.")
            else:
                lignes = []
                for r in a_venir.itertuples():
                    p = modele.predire(r.HomeTeam, r.AwayTeam)
                    score = modele.scores_probables(r.HomeTeam, r.AwayTeam, 1)[0]
                    issues = {"1": p["p_dom"], "N": p["p_nul"], "2": p["p_ext"]}
                    lignes.append({"Date": f"{JOURS[r.Date.weekday()]} {r.Date:%d/%m}", "Heure": r.Heure, "Domicile": r.HomeTeam,
                                   "Extérieur": r.AwayTeam, "1": p["p_dom"] * 100, "N": p["p_nul"] * 100,
                                   "2": p["p_ext"] * 100, "+2,5 buts": p["p_plus_2_5"] * 100,
                                   "Score le plus probable": f"{score[0]} ({score[1]:.0%})",
                                   "Tendance": max(issues, key=issues.get)})
                st.dataframe(pd.DataFrame(lignes), hide_index=True, width="stretch",
                             column_config={"1": POURCENT, "N": POURCENT, "2": POURCENT, "+2,5 buts": POURCENT})
                st.caption(f"Modèle entraîné sur {len(res[res['Date'] >= pd.Timestamp(aujourdhui) - pd.Timedelta(days=FENETRE_JOURS)])} "
                           f"matchs des deux dernières saisons, les plus récents comptant davantage.")

    # ---- simulateur
    with onglets[1]:
        equipes = sorted(set(cal["HomeTeam"]) | set(cal["AwayTeam"])) if not cal.empty else modele.equipes
        c1, c2 = st.columns(2)
        dom = c1.selectbox("Équipe à domicile", equipes, index=0, key=f"dom_{ligue}")
        ext = c2.selectbox("Équipe à l'extérieur", equipes, index=min(1, len(equipes) - 1), key=f"ext_{ligue}")
        if dom == ext:
            st.warning("Choisis deux équipes différentes.")
        else:
            inconnues = [e for e in (dom, ext) if e not in modele.equipes]
            if inconnues:
                st.info(f"{', '.join(inconnues)} : trop peu de matchs récents dans ce championnat (promu). "
                        "Le modèle lui attribue le niveau moyen des équipes les plus faibles.")
            p = modele.predire(dom, ext)
            m = st.columns(5)
            m[0].metric(f"Victoire {dom}", f"{p['p_dom']:.0%}")
            m[1].metric("Match nul", f"{p['p_nul']:.0%}")
            m[2].metric(f"Victoire {ext}", f"{p['p_ext']:.0%}")
            m[3].metric("Plus de 2,5 buts", f"{p['p_plus_2_5']:.0%}")
            m[4].metric("Les deux marquent", f"{p['p_les_deux_marquent']:.0%}")
            st.markdown(f"Buts attendus : **{p['buts_dom']:.2f}** pour {dom}, **{p['buts_ext']:.2f}** pour {ext}.")

            mat = modele.matrice_scores(dom, ext)[:6, :6]
            df_mat = pd.DataFrame([{"Buts " + dom: i, "Buts " + ext: j, "p": mat[i, j]}
                                   for i in range(6) for j in range(6)])
            base = alt.Chart(df_mat).encode(
                x=alt.X(f"Buts {ext}:O", axis=alt.Axis(orient="top", labelAngle=0)), y=alt.Y(f"Buts {dom}:O"))
            carte = base.mark_rect().encode(color=alt.Color("p:Q", scale=alt.Scale(scheme="blues"), legend=None),
                                            tooltip=[alt.Tooltip("p:Q", format=".1%", title="Probabilité")])
            texte = base.mark_text(fontSize=12).encode(
                text=alt.Text("p:Q", format=".0%"),
                color=alt.condition(alt.datum.p > 0.07, alt.value("white"), alt.value("#333")))
            st.markdown("**Probabilité de chaque score exact**")
            st.altair_chart((carte + texte).properties(height=300, width=360))
            comparer_cotes([p["p_dom"], p["p_nul"], p["p_ext"]], ["1", "N", "2"], f"foot_{ligue}")

    # ---- forces
    with onglets[2]:
        forces = modele.classement_forces()
        actuelles = set(cal["HomeTeam"]) if not cal.empty else set(forces["equipe"])
        forces = forces[forces["equipe"].isin(actuelles)]
        st.markdown("Chaque point est une équipe. **À droite** : elle marque plus que la moyenne. "
                    "**En haut** : elle encaisse moins que la moyenne. Les promus récents n'apparaissent pas "
                    "tant qu'ils n'ont pas joué assez de matchs.")
        pts = alt.Chart(forces).encode(
            x=alt.X("attaque:Q", title="Attaque →"), y=alt.Y("defense:Q", title="Défense →"),
            tooltip=["equipe", alt.Tooltip("attaque:Q", format="+.2f"), alt.Tooltip("defense:Q", format="+.2f")])
        st.altair_chart((pts.mark_circle(size=90) + pts.mark_text(dx=8, align="left", fontSize=11).encode(text="equipe"))
                        .properties(height=480), width="stretch")

    # ---- bilan
    with onglets[3]:
        bilan = foot_bilan_saison(ligue, cle)
        if bilan.empty:
            st.info("Aucun match joué cette saison pour l'instant.")
        else:
            p = bilan[["p1", "pN", "p2"]].to_numpy()
            y = bilan["Resultat"].to_numpy()
            ok = p.argmax(1) == y
            c = st.columns(3)
            c[0].metric("Matchs joués", len(bilan))
            c[1].metric("Bons pronostics 1N2", f"{ok.mean():.0%}")
            c[2].metric("Log loss (↓)", f"{log_loss(p, y):.3f}", help="Vers 0,99 en moyenne sur 10 saisons ; "
                        "1,10 correspond à des probabilités tirées au hasard.")
            vue = bilan.assign(
                Date=bilan["Date"].dt.strftime("%d/%m"),
                Pronostic=np.array(["1", "N", "2"])[p.argmax(1)],
                Confiance=p.max(1) * 100,
                Résultat=np.array(["1", "N", "2"])[y],
                Correct=np.where(ok, "✅", "❌"),
            )[["Date", "Domicile", "Extérieur", "Score", "Pronostic", "Confiance", "Résultat", "Correct"]]
            st.dataframe(vue.iloc[::-1], hide_index=True, width="stretch",
                         column_config={"Confiance": POURCENT})
            st.caption("Pronostics recalculés comme ils l'auraient été avant chaque journée, "
                       "avec uniquement les matchs déjà joués à ce moment-là.")

    with onglets[4]:
        fiabilite("foot")


# =============================================================== page NBA
def page_nba() -> None:
    cle = cle_donnees()
    res, notes, a_venir = nba_donnees(cle)
    elo = EloNBA()
    derniere_saison = int(res["saison"].max())
    st.header("🏀 NBA")
    bandeau_avertissement()

    if res["date"].max() < pd.Timestamp(date.today() - timedelta(days=400)):
        st.warning("Les données NBA récentes n'ont pas encore été récupérées : les notes datent de "
                   f"{res['date'].max():%Y}. Lance la GitHub Action « Mise à jour des données » "
                   "(voir le README) pour charger les saisons depuis 2015.")

    saison_cible = int(a_venir["saison"].min()) if len(a_venir) else derniere_saison
    notes_a_jour = elo.notes_pour_saison(notes, derniere_saison, saison_cible)

    onglets = st.tabs(["Prochains matchs", "Simulateur", "Classement Elo", "Bilan de la saison", "Fiabilité"])

    with onglets[0]:
        if a_venir.empty:
            st.info("Aucun match programmé dans les 14 prochains jours (ou calendrier pas encore récupéré).")
        else:
            lignes = []
            for r in a_venir.sort_values(["date", "heure_paris"]).itertuples():
                p = elo.predire(notes_a_jour, r.dom, r.ext, bool(r.neutre))
                fav = r.dom if p["p_dom"] >= 0.5 else r.ext
                lignes.append({"Date (heure de Paris)": r.heure_paris, "Domicile": r.dom, "Extérieur": r.ext,
                               "Victoire domicile": p["p_dom"] * 100, "Victoire extérieur": p["p_ext"] * 100,
                               "Écart prévu": f"{fav} de {abs(p['ecart']):.0f} pts"})
            st.dataframe(pd.DataFrame(lignes), hide_index=True, width="stretch",
                         column_config={"Victoire domicile": POURCENT, "Victoire extérieur": POURCENT})
            if saison_cible > derniere_saison:
                st.caption("Nouvelle saison : les notes de la saison passée ont été ramenées d'un quart vers la "
                           "moyenne, car les effectifs changent pendant l'intersaison.")

    with onglets[1]:
        franchises = sorted(notes_a_jour, key=lambda e: -notes_a_jour[e])
        franchises = [f for f in franchises if f in set(res.loc[res.saison == derniere_saison, "dom"])] or franchises
        c1, c2, c3 = st.columns([2, 2, 1])
        dom = c1.selectbox("Équipe à domicile", sorted(franchises), key="nba_dom")
        ext = c2.selectbox("Équipe à l'extérieur", sorted(franchises), index=1, key="nba_ext")
        neutre = c3.checkbox("Terrain neutre")
        if dom == ext:
            st.warning("Choisis deux équipes différentes.")
        else:
            p = elo.predire(notes_a_jour, dom, ext, neutre)
            m = st.columns(3)
            m[0].metric(f"Victoire {dom}", f"{p['p_dom']:.0%}", help=f"Elo {p['elo_dom']:.0f}")
            m[1].metric(f"Victoire {ext}", f"{p['p_ext']:.0%}", help=f"Elo {p['elo_ext']:.0f}")
            m[2].metric("Écart prévu", f"{p['ecart']:+.1f} pts", help=f"Positif = avantage {dom}")
            comparer_cotes([p["p_dom"], p["p_ext"]], [dom, ext], "nba")

    with onglets[2]:
        st.markdown("Note Elo de chaque franchise (1505 = équipe moyenne). Une équipe qui a 100 points de plus "
                    "que son adversaire gagne environ 64 % du temps sur terrain neutre.")
        df_n = pd.DataFrame({"Franchise": list(notes_a_jour), "Elo": list(notes_a_jour.values())})
        df_n = df_n[df_n["Franchise"].isin(set(res.loc[res.saison == derniere_saison, "dom"]))]
        st.altair_chart(alt.Chart(df_n).mark_bar().encode(
            x=alt.X("Elo:Q", scale=alt.Scale(domain=[1200, df_n["Elo"].max() + 30])),
            y=alt.Y("Franchise:N", sort="-x", title=None),
            color=alt.condition(alt.datum.Elo >= 1505, alt.value("#1f5f8b"), alt.value("#c0c7cf")),
            tooltip=["Franchise", alt.Tooltip("Elo:Q", format=".0f")]).properties(height=620),
            width="stretch")

    with onglets[3]:
        saison = res[res["saison"] == derniere_saison]
        if saison.empty or derniere_saison <= 2015:
            st.info("Pas encore de matchs récents dans les données.")
        else:
            ok = (saison["p_elo_dom"] > 0.5) == (saison["victoire_dom"] == 1)
            c = st.columns(2)
            c[0].metric(f"Matchs de la saison {derniere_saison - 1}-{str(derniere_saison)[2:]}", len(saison))
            c[1].metric("Bons pronostics", f"{ok.mean():.0%}")
            vue = saison.assign(
                Date=saison["date"].dt.strftime("%d/%m/%Y"), Score=saison["pts_dom"].astype(str) + "-" + saison["pts_ext"].astype(str),
                Favori=np.where(saison["p_elo_dom"] > 0.5, saison["dom"], saison["ext"]),
                Confiance=np.maximum(saison["p_elo_dom"], 1 - saison["p_elo_dom"]) * 100,
                Correct=np.where(ok, "✅", "❌"))[["Date", "dom", "ext", "Score", "Favori", "Confiance", "Correct"]]
            st.dataframe(vue.iloc[::-1].rename(columns={"dom": "Domicile", "ext": "Extérieur"}), hide_index=True,
                         width="stretch", column_config={"Confiance": POURCENT})

    with onglets[4]:
        fiabilite("nba")


# =============================================================== mise en page
def barre_laterale() -> tuple[str, str | None]:
    with st.sidebar:
        st.title("Pronostics")
        sport = st.radio("Sport", ["Football", "NBA"], horizontal=True)
        ligue = None
        if sport == "Football":
            ligue = st.selectbox("Championnat", list(LIGUES), format_func=lambda l: LIGUES[l]["nom"])
        st.divider()
        etat = etat_mise_a_jour()
        st.markdown("**Fraîcheur des données**")
        if etat.get("derniere_execution"):
            dt = datetime.fromisoformat(etat["derniere_execution"].replace("Z", "+00:00"))
            st.caption(f"Dernière mise à jour automatique : {dt:%d/%m/%Y à %H:%M} UTC")
        info = (etat.get("foot", {}).get(ligue) if ligue else etat.get("nba")) or {}
        if info.get("statut") == "erreur":
            st.warning("La dernière récupération a échoué : les données affichées sont les dernières disponibles.")
            st.caption(f"Détail : {info.get('erreur', '')[:150]}")
        if info.get("dernier_resultat"):
            st.caption(f"Dernier résultat connu : {pd.Timestamp(info['dernier_resultat']):%d/%m/%Y}")
        if info.get("source_resultats"):
            st.caption(f"Source : {info['source_resultats']}")
        st.divider()
        with st.expander("Comment ça marche ?"):
            st.markdown(
                "- **Football** : modèle de Poisson (Dixon-Coles). Chaque équipe a une force d'attaque et de "
                "défense estimée sur ses matchs récents ; on en déduit la probabilité de chaque score.\n"
                "- **NBA** : notes Elo. Le vainqueur prend des points au perdant, d'autant plus que la victoire "
                "était inattendue et large.\n"
                "- Les données sont mises à jour **chaque jour** par une tâche automatique.\n"
                "- Onglet **Fiabilité** : performances mesurées sur 10 saisons de matchs passés.")
    return sport, ligue


def main() -> None:
    sport, ligue = barre_laterale()
    if sport == "Football":
        page_foot(ligue)
    else:
        page_nba()


main()
