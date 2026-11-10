# HortisMatch – prototype

Associer en quelques secondes le bon consultant à la bonne mission, avec un **score sur 100 expliqué**.
Prototype réalisé par Anthony Nguyen (Mastère Data Engineer & IA, EFREI Paris) pour illustrer un sujet d'alternance.

> Toutes les données (consultants, missions, clients) sont **fictives**, sauf le profil de l'auteur.

## Ce que fait le prototype

| Onglet | Rôle |
| --- | --- |
| Matching | Classe les consultants pour une mission, avec ce qu'ils couvrent, ce qui manque et le détail du score. On peut aussi coller une nouvelle fiche de mission. |
| Profils extraits | Montre ce que l'outil a compris de chaque CV (transparence nLPD / RGPD). |

On peut aussi **importer un CV** (PDF, DOCX, TXT) depuis la barre latérale. L'outil propose le nom, le site (d'après l'adresse en tête du CV), l'expérience et les langues ; on vérifie, puis le profil et le fichier d'origine sont enregistrés dans une base SQLite locale, `data/hortismatch.db` (module `storage.py`). Les CV enregistrés se retrouvent à chaque lancement et peuvent être téléchargés ou supprimés.

## Le score

```
Score = 95 × (0,45 proximité du CV + 0,30 compétences obligatoires + 0,10 séniorité
              + 0,10 disponibilité + 0,05 langues et site)
      + 5 × part des compétences souhaitées couvertes
```

Les poids se règlent en direct dans la barre latérale.

- **Proximité du CV** : similarité sémantique entre le CV et la fiche de mission. Le prototype utilise un modèle multilingue `sentence-transformers` s'il est installé, sinon un TF-IDF (aucun téléchargement).
- **Compétences** : extraites avec un référentiel d'environ 60 compétences et leurs synonymes FR/EN (à aligner sur ESCO dans la version cible).

## Lancer

```bash
pip install -r requirements.txt
streamlit run app.py
```

Pour le moteur sémantique multilingue (optionnel, ~500 Mo) :

```bash
pip install sentence-transformers
```

Test rapide en ligne de commande : `python matcher.py`

## Structure

```
app.py               interface Streamlit
matcher.py           extraction des CV, score hybride
storage.py           base SQLite des CV importés
style.css            thème de l'interface
data/consultants.json  12 profils fictifs
data/missions.json     8 missions fictives
```

## Limites et suite

- Les données sont fictives : la qualité du classement reste à valider sur l'historique réel des staffings.
- Étapes suivantes : référentiel ESCO, PostgreSQL + pgvector, API FastAPI, apprentissage des poids sur l'historique (learning to rank), pseudonymisation des CV, hébergement en Suisse.
