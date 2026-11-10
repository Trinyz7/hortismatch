"""HortisMatch - interface de démonstration (Streamlit).

Lancer :  streamlit run app.py
"""
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

import storage
from matcher import (DEFAULT_WEIGHTS, LANGS, SITES, SKILLS, Consultant, Mission, SemanticEncoder,
                     document_language, extract_languages, extract_profile, extract_site,
                     extract_experience, extract_skills, load_consultants, load_missions, match, read_document)

st.set_page_config(page_title="HortisMatch", page_icon="🎯", layout="wide")
st.markdown(f"<style>{(Path(__file__).parent / 'style.css').read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True)

# Couleurs système Apple : (couleur principale, dégradé de la barre).
PALETTE = {
    "blue": ("#0A84FF", "linear-gradient(90deg, #5AC8FA, #0A84FF)"),
    "indigo": ("#5E5CE6", "linear-gradient(90deg, #A29BFE, #5E5CE6)"),
    "green": ("#30D158", "linear-gradient(90deg, #A8E063, #30D158)"),
    "orange": ("#FF9F0A", "linear-gradient(90deg, #FFD60A, #FF9F0A)"),
    "pink": ("#FF375F", "linear-gradient(90deg, #FF8FA3, #FF375F)"),
}


def _vars(color: str, pct: float, delay: float) -> str:
    c, g = PALETTE[color]
    return f"--c:{c};--g:{g};--w:{max(0.0, min(pct, 100.0)):.1f}%;--d:{delay:.2f}s"


def counter(value: float, delay: float = 0.0) -> str:
    """Nombre entier animé de 0 à value (CSS pur, sans JavaScript)."""
    return f'<span class="hm-count" style="--to:{round(value)};--d:{delay:.2f}s"></span>'


def bar(pct: float, color: str = "blue", delay: float = 0.0) -> str:
    return f'<div class="hm-track"><div class="hm-fill" style="{_vars(color, pct, delay)}"></div></div>'


def kpi_tiles(tiles: list[dict]) -> None:
    """Tuiles façon widgets iOS : libellé, chiffre animé, barre qui se remplit."""
    parts = []
    for i, t in enumerate(tiles):
        d = 0.08 * i
        parts.append(
            f'<div class="hm-kpi" style="{_vars(t["color"], t["pct"], d)}">'
            f'<div class="hm-kpi-label"><span class="hm-dot"></span>{escape(t["label"])}</div>'
            f'<div class="hm-kpi-value">{counter(t["value"], d)}<span class="hm-unit">{escape(t.get("unit", ""))}</span></div>'
            f'{bar(t["pct"], t["color"], d + 0.1)}'
            f'<div class="hm-kpi-foot">{escape(t.get("foot", ""))}</div></div>')
    st.html(f'<div class="hm-kpis">{"".join(parts)}</div>')


def skill_tags(groups: list[tuple[str, str, str, list[str]]]) -> str:
    """Lignes « libellé + pastilles » : (libellé, icône, couleur, compétences)."""
    rows = []
    for label, icon, color, skills in groups:
        if not skills:
            continue
        chips = "".join(f'<span class="hm-chip">{escape(sk)}</span>' for sk in skills)
        rows.append(
            f'<div class="hm-tagrow hm-{icon}" style="--c:{PALETTE[color][0]}">'
            f'<span class="hm-tagkey"><span class="hm-ico" aria-hidden="true"></span>'
            f'{label}</span><div class="hm-chips">{chips}</div></div>')
    return f'<div class="hm-tags">{"".join(rows)}</div>'


def score_color(score: float) -> str:
    return "green" if score >= 75 else "blue" if score >= 50 else "orange" if score >= 30 else "pink"


@st.cache_resource
def get_encoder():
    return SemanticEncoder()


# Consultants fictifs + CV importés et conservés dans la base SQLite.
consultants = load_consultants() + storage.load_imported()
missions = load_missions()
encoder = get_encoder()

st.title("HortisMatch")
st.caption("Prototype : associer le bon consultant à la bonne mission, avec un score expliqué. "
           f"Données fictives · moteur sémantique : {encoder.backend}")

# ---------------------------------------------------------------- Barre latérale
with st.sidebar:
    st.header("Pondération du score")
    st.caption("Ajustez l'importance de chaque critère.")
    w = {
        "sem": st.slider("Proximité du CV avec la mission", 0.0, 1.0, DEFAULT_WEIGHTS["sem"], 0.05),
        "must": st.slider("Compétences obligatoires", 0.0, 1.0, DEFAULT_WEIGHTS["must"], 0.05),
        "senior": st.slider("Séniorité", 0.0, 1.0, DEFAULT_WEIGHTS["senior"], 0.05),
        "dispo": st.slider("Disponibilité", 0.0, 1.0, DEFAULT_WEIGHTS["dispo"], 0.05),
        "lang": st.slider("Langues et site", 0.0, 1.0, DEFAULT_WEIGHTS["lang"], 0.05),
    }
    st.divider()
    st.header("Ajouter un CV")
    up = st.file_uploader("PDF, DOCX ou TXT", type=["pdf", "docx", "txt"])
    if up is not None and st.button("Analyser"):
        data = up.getvalue()
        text = read_document(data, up.name)
        langs = extract_languages(text) | ({document_language(text)} - {None})
        st.session_state.draft = {
            "fichier_nom": up.name, "fichier": data, "texte": text,
            **extract_profile(text, up.name, data if up.name.lower().endswith(".pdf") else None),
            "site": extract_site(text),
            **dict(zip(("annees", "annees_detail"), extract_experience(text))),
            "langues": sorted(langs),
            "competences": sorted(extract_skills(text)),
        }

    # Étape de validation : l'outil propose, la personne corrige avant l'enregistrement.
    draft = st.session_state.get("draft")
    if draft:
        with st.form("valider_cv", border=False):
            st.caption(f"{len(draft['competences'])} compétences détectées dans « {draft['fichier_nom']} ». "
                       "Vérifiez le profil avant de l'enregistrer.")
            nom = st.text_input("Nom", draft["nom"] or "", placeholder="Prénom Nom")
            titre = st.text_input("Titre", draft["titre"] or "", placeholder="Ex. : Data Engineer")
            sites = list(SITES)
            site = st.selectbox("Site", sites, index=sites.index(draft["site"]) if draft["site"] else 0,
                                help="Détecté depuis l'adresse en tête du CV." if draft["site"]
                                else "Aucune ville reconnue dans le CV : à choisir.")
            c1, c2 = st.columns(2)
            annees = c1.number_input("Expérience (ans)", 0, 45, int(draft["annees"]),
                                     help=f"Expérience professionnelle détectée : {draft['annees_detail']}. "
                                          "L'âge, la formation et le poste recherché ne sont pas comptés.")
            dispo = c2.number_input("Dispo (jours)", 0, 365, 0)
            langues = st.multiselect("Langues", list(LANGS), default=draft["langues"])
            ok, cancel = st.columns(2)
            if ok.form_submit_button("Enregistrer", type="primary"):
                new = Consultant(id="draft", nom=nom.strip() or draft["fichier_nom"].rsplit(".", 1)[0],
                                 titre=titre.strip() or "Profil importé",
                                 site=site, annees_experience=int(annees), disponible_dans_jours=int(dispo),
                                 cv=draft["texte"], langues=set(langues))
                storage.save_cv(new, draft["fichier_nom"], draft["fichier"])
                del st.session_state.draft
                st.toast(f"{new.nom} est enregistré dans la base.")
                st.rerun()
            if cancel.form_submit_button("Annuler"):
                del st.session_state.draft
                st.rerun()

    stored = storage.list_imported()
    if stored:
        with st.expander(f"CV enregistrés ({len(stored)})"):
            for cv in stored:
                st.markdown(f"**{escape(cv['nom'])}** · {escape(cv['site'])}<br>"
                            f"<span style='opacity:.6;font-size:.8rem'>{escape(cv['fichier_nom'])} · "
                            f"{cv['importe_le'][:10]} · {cv['taille'] // 1024} Ko</span>", unsafe_allow_html=True)
                d1, d2 = st.columns(2)
                fname, fdata = storage.get_file(cv["id"])
                d1.download_button("Fichier", fdata, file_name=fname, key=f"dl_{cv['id']}")
                if d2.button("Supprimer", key=f"del_{cv['id']}"):
                    storage.delete_cv(cv["id"])
                    st.rerun()

tab_match, tab_profils = st.tabs(["Matching", "Profils extraits"])

# ---------------------------------------------------------------- Matching
with tab_match:
    labels = {f"{m.titre} — {m.client}": m for m in missions}
    labels["✏️ Saisir une nouvelle mission…"] = None
    choice = st.selectbox("Mission", list(labels))
    mission = labels[choice]

    if mission is None:
        desc = st.text_area("Collez la fiche de mission", height=180,
                            placeholder="Ex. : Nous recherchons un data analyst Power BI pour le contrôle de gestion…")
        detected = sorted(extract_skills(desc)) if desc else []
        must = st.multiselect("Compétences obligatoires (détectées automatiquement, modifiables)",
                              sorted(SKILLS), default=detected)
        c1, c2, c3 = st.columns(3)
        site = c1.selectbox("Site", ["Genève", "Lausanne", "Paris"])
        start = c2.number_input("Début dans (jours)", 0, 180, 15)
        senior = c3.number_input("Années d'expérience minimum", 0, 20, 3)
        mission = Mission("custom", "Nouvelle mission", "Saisie libre", site, int(start), int(senior),
                          desc or "", must, [], sorted(extract_languages(desc)) if desc else [])
    else:
        with st.expander("Voir la fiche de mission", expanded=False):
            st.write(mission.description)
            st.write("**Obligatoires :** " + ", ".join(mission.obligatoires))
            st.write("**Souhaitées :** " + ", ".join(mission.souhaitees))
            st.write(f"**Site :** {mission.site} · **Début :** dans {mission.debut_dans_jours} jours · "
                     f"**Séniorité :** {mission.seniorite_min} ans · **Langues :** {', '.join(mission.langues)}")

    if mission.description or mission.obligatoires:
        results = match(mission, consultants, encoder, w)
        top_n = st.radio("Afficher", [3, 5, len(results)], horizontal=True,
                         format_func=lambda n: "Tous" if n == len(results) else f"Top {n}")

        # Tableau de bord de la mission.
        top3 = results[:3]
        dispo30 = sum(r.consultant.disponible_dans_jours <= 30 for r in results)
        must_top = top3[0].detail["must"] * 100 if top3 else 0
        kpi_tiles([
            {"label": "Meilleur score", "value": results[0].score, "unit": "/100", "pct": results[0].score,
             "color": score_color(results[0].score), "foot": results[0].consultant.nom},
            {"label": "Moyenne du top 3", "value": sum(r.score for r in top3) / len(top3), "unit": "/100",
             "pct": sum(r.score for r in top3) / len(top3), "color": "indigo",
             "foot": "Qualité de la short-list"},
            {"label": "Obligatoires couvertes", "value": must_top, "unit": "%", "pct": must_top,
             "color": "blue", "foot": "Par le n° 1 du classement"},
            {"label": "Disponibles sous 30 j", "value": dispo30, "unit": f"/{len(results)}",
             "pct": 100 * dispo30 / len(results), "color": "orange", "foot": "Profils mobilisables vite"},
        ])

        criteria = [("sem", "Proximité du CV", "blue"), ("must", "Compétences obligatoires", "indigo"),
                    ("senior", "Séniorité", "green"), ("dispo", "Disponibilité", "orange"),
                    ("lang", "Langues et site", "pink")]
        total_w = sum(w.values()) or 1.0
        for rank, r in enumerate(results[:top_n], 1):
            c = r.consultant
            with st.container(border=True, key=f"card_{rank}"):
                a, b = st.columns([3, 1])
                a.subheader(f"{rank}. {c.nom}")
                a.caption(f"{c.titre} · {c.site} · {c.annees_experience} ans d'expérience · "
                          f"disponible {'maintenant' if c.disponible_dans_jours == 0 else f'dans {c.disponible_dans_jours} jours'}")
                d = 0.06 * (rank - 1)
                with b:
                    st.html(f'<div class="hm-score"><div class="hm-score-label">Score</div>'
                            f'<div class="hm-score-value">{counter(r.score, d)}<span class="hm-unit">/100</span></div>'
                            f'{bar(r.score, score_color(r.score), d)}</div>')
                a.html(skill_tags([("Couvre", "check", "green", r.couvertes),
                                   ("Manque", "alert", "orange", r.manquantes),
                                   ("Atouts", "plus", "indigo", r.bonus)]))
                with a.expander("Détail du score"):
                    rows = []
                    for i, (k, name, color) in enumerate(criteria):
                        note = r.detail[k] * 100
                        pts = 95 * w[k] * r.detail[k] / total_w
                        rows.append(
                            f'<div class="hm-row" style="--c:{PALETTE[color][0]};--d:{0.07 * i:.2f}s">'
                            f'<div class="hm-row-head"><span class="hm-row-name">{name}</span>'
                            f'<span class="hm-row-meta">poids {100 * w[k] / total_w:.0f} % · '
                            f'<b>+{pts:.1f} pts</b></span></div>'
                            f'<div style="display:flex;align-items:center;gap:12px">'
                            f'<div style="flex:1">{bar(note, color, 0.07 * i)}</div>'
                            f'<span class="hm-row-meta" style="min-width:42px;text-align:right">'
                            f'{counter(note, 0.07 * i)} %</span></div></div>')
                    st.html(f'<div class="hm-rows">{"".join(rows)}</div>')
        st.info("L'outil recommande, un business manager décide. Chaque score est expliqué.")
    else:
        st.write("Collez une fiche de mission pour lancer le matching.")

# ---------------------------------------------------------------- Profils
with tab_profils:
    st.subheader("Ce que l'outil a extrait de chaque CV")
    st.caption("Les consultants pourraient vérifier et corriger leur profil (transparence nLPD / RGPD).")
    st.dataframe(pd.DataFrame([{
        "Consultant": c.nom, "Titre": c.titre, "Site": c.site, "Expérience (ans)": c.annees_experience,
        "Langues": ", ".join(sorted(c.langues)), "Nb compétences": len(c.competences),
        "Compétences": ", ".join(sorted(c.competences)),
    } for c in consultants]), hide_index=True, width="stretch")
