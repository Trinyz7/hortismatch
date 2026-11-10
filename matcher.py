"""
HortisMatch - moteur de matching consultants / missions (prototype).

Score hybride et explicable :
    Score = w_sem * similarité sémantique
          + w_must * couverture des compétences obligatoires
          + w_senior * adéquation de séniorité
          + w_dispo * disponibilité
          + w_lang * langues et site

La similarité sémantique utilise sentence-transformers (modèle multilingue)
si le paquet est installé, sinon un TF-IDF (aucun téléchargement nécessaire).
"""
from __future__ import annotations

import json
import re
from datetime import date
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent / "data"

DEFAULT_WEIGHTS = {"sem": 0.45, "must": 0.30, "senior": 0.10, "dispo": 0.10, "lang": 0.05}

# --------------------------------------------------------------------------
# Référentiel de compétences : nom canonique -> (catégorie, synonymes)
# Dans la version cible, ce référentiel est aligné sur ESCO.
# --------------------------------------------------------------------------
SKILLS: dict[str, tuple[str, list[str]]] = {
    # Langages
    "Python": ("Langage", ["python"]),
    "SQL": ("Langage", ["sql", "t-sql", "pl/sql", "plsql"]),
    "Java": ("Langage", ["java"]),
    "Kotlin": ("Langage", ["kotlin"]),
    "JavaScript": ("Langage", ["javascript", "js", "typescript", "ts"]),
    "PHP": ("Langage", ["php"]),
    "C#": ("Langage", ["c#", ".net", "dotnet"]),
    "ABAP": ("Langage", ["abap"]),
    "R": ("Langage", ["langage r", "rstudio"]),
    # Data & IA
    "Pandas": ("Data & IA", ["pandas"]),
    "Machine Learning": ("Data & IA", ["machine learning", "apprentissage automatique", "scikit-learn", "sklearn"]),
    "Deep Learning": ("Data & IA", ["deep learning", "pytorch", "tensorflow", "keras"]),
    "NLP": ("Data & IA", ["nlp", "traitement du langage", "natural language processing"]),
    "IA générative": ("Data & IA", ["ia générative", "generative ai", "genai", "llm", "rag"]),
    "Data Visualisation": ("Data & IA", ["data visualisation", "dataviz", "data visualization"]),
    "Statistiques": ("Data & IA", ["statistiques", "statistics", "modélisation statistique"]),
    "Power BI": ("BI", ["power bi", "powerbi", "dax"]),
    "Tableau": ("BI", ["tableau software", "tableau desktop"]),
    "Qlik": ("BI", ["qlik", "qlikview", "qlik sense"]),
    "Excel avancé": ("BI", ["excel avancé", "vba", "power query"]),
    "Data Warehouse": ("Data Engineering", ["data warehouse", "entrepôt de données", "datawarehouse"]),
    "ETL": ("Data Engineering", ["etl", "elt", "talend", "informatica", "ssis"]),
    "Airflow": ("Data Engineering", ["airflow", "prefect", "dagster"]),
    "Spark": ("Data Engineering", ["spark", "pyspark", "databricks"]),
    "Data Lake": ("Data Engineering", ["data lake", "data lakes", "datalake", "lakehouse"]),
    "Kafka": ("Data Engineering", ["kafka", "streaming temps réel"]),
    "dbt": ("Data Engineering", ["dbt"]),
    "API REST": ("Data Engineering", ["api rest", "api", "web services", "fastapi"]),
    # Bases de données
    "PostgreSQL": ("Base de données", ["postgresql", "postgres"]),
    "MySQL": ("Base de données", ["mysql", "mariadb"]),
    "Oracle": ("Base de données", ["oracle"]),
    "MongoDB": ("Base de données", ["mongodb", "nosql"]),
    # Cloud & DevOps
    "Docker": ("Cloud & DevOps", ["docker", "conteneur"]),
    "Kubernetes": ("Cloud & DevOps", ["kubernetes", "k8s"]),
    "CI/CD": ("Cloud & DevOps", ["ci/cd", "github actions", "gitlab ci", "jenkins"]),
    "Git": ("Cloud & DevOps", ["git", "github", "gitlab"]),
    "Azure": ("Cloud & DevOps", ["azure"]),
    "AWS": ("Cloud & DevOps", ["aws", "amazon web services"]),
    "Linux": ("Cloud & DevOps", ["linux", "unix"]),
    # Développement
    "React": ("Développement", ["react", "react.js", "reactjs"]),
    "Vue.js": ("Développement", ["vue.js", "vuejs", "vue 3"]),
    "Node.js": ("Développement", ["node.js", "nodejs"]),
    "Symfony": ("Développement", ["symfony"]),
    "Android": ("Développement", ["android"]),
    "iOS": ("Développement", ["ios", "swift"]),
    "UX/UI": ("Développement", ["ux", "ui", "figma", "ergonomie"]),
    # Métier & méthodes
    "Business Analysis": ("Métier", ["business analysis", "business analyst", "analyste métier", "recueil des besoins", "spécifications fonctionnelles"]),
    "Gestion de projet": ("Métier", ["gestion de projet", "chef de projet", "project management", "pmo"]),
    "Agile / Scrum": ("Métier", ["agile", "scrum", "kanban", "product owner"]),
    "Tests / Recette": ("Métier", ["recette", "tests fonctionnels", "uat", "qa"]),
    "Banque privée": ("Secteur", ["banque privée", "private banking", "gestion de fortune", "wealth management"]),
    "Banque": ("Secteur", ["banque", "bancaire", "banking"]),
    "Finance / Contrôle de gestion": ("Secteur", ["contrôle de gestion", "finance", "comptabilité", "reporting financier"]),
    "Médias": ("Secteur", ["médias", "media", "audiovisuel"]),
    "Secteur public": ("Secteur", ["secteur public", "administration publique", "état de genève"]),
    "Olympic Banking System": ("Progiciel", ["olympic", "olympic banking"]),
    "Avaloq": ("Progiciel", ["avaloq"]),
    "Temenos T24": ("Progiciel", ["temenos", "t24"]),
    "SAP": ("Progiciel", ["sap", "s/4hana"]),
    "Conformité / LBA": ("Métier", ["conformité", "compliance", "lba", "aml", "kyc", "finma"]),
    "Protection des données": ("Métier", ["nlpd", "rgpd", "gdpr", "lpd", "protection des données"]),
}

LANGS = {
    "Français": ["français", "french", "francais"],
    "Anglais": ["anglais", "english"],
    "Allemand": ["allemand", "german", "deutsch"],
    "Italien": ["italien", "italian"],
    "Espagnol": ["espagnol", "spanish", "español"],
    "Portugais": ["portugais", "portuguese"],
    "Néerlandais": ["néerlandais", "dutch"],
    "Arabe": ["arabe", "arabic"],
    "Chinois": ["chinois", "mandarin", "chinese"],
    "Japonais": ["japonais", "japanese"],
    "Vietnamien": ["vietnamien", "vietnamese"],
}


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text.lower())
    return re.sub(r"\s+", " ", text)


def _alias_pattern(alias: str) -> re.Pattern:
    # Limites de mot robustes aux symboles (c#, ci/cd, .net, node.js)
    return re.compile(r"(?<![\w])" + re.escape(alias) + r"(?![\w])")


_PATTERNS = {name: [_alias_pattern(a) for a in aliases] for name, (_, aliases) in SKILLS.items()}
_LANG_PATTERNS = {name: [_alias_pattern(a) for a in aliases] for name, aliases in LANGS.items()}


def extract_skills(text: str) -> set[str]:
    t = _norm(text)
    found = {name for name, pats in _PATTERNS.items() if any(p.search(t) for p in pats)}
    # "Banque privée" implique "Banque"
    if "Banque privée" in found:
        found.add("Banque")
    return found


def extract_languages(text: str) -> set[str]:
    t = _norm(text)
    return {name for name, pats in _LANG_PATTERNS.items() if any(p.search(t) for p in pats)}


# Sites Hortis et indices de localisation (ville, variantes, codes postaux, communes proches).
SITES = {
    "Genève": ["genève", "geneve", "geneva", "genf", "carouge", "lancy", "meyrin", "vernier", "onex",
               "plan-les-ouates", "annemasse", r"ch-12\d\d", r"12\d\d genève"],
    "Lausanne": ["lausanne", "renens", "pully", "morges", "prilly", "epalinges", "vaud", r"ch-10\d\d",
                 r"10\d\d lausanne"],
    "Paris": ["paris", "île-de-france", "ile-de-france", "idf", "villejuif", "boulogne-billancourt", "nanterre",
              "la défense", "la defense", "saint-denis", "montreuil", "issy-les-moulineaux", "levallois",
              "neuilly-sur-seine", "ivry-sur-seine", "vitry-sur-seine", "créteil", "versailles",
              r"7[5789]\d{3}", r"9[1-5]\d{3}"],
}
_SITE_PATTERNS = {
    site: [re.compile(r"(?<![\w-])" + (a if "\\d" in a else re.escape(a)) + r"(?![\w-])") for a in aliases]
    for site, aliases in SITES.items()}


def extract_site(text: str) -> str | None:
    """Devine le site du consultant.

    L'adresse est presque toujours dans l'en-tête du CV : une mention dans les
    premières lignes pèse bien plus lourd que les lieux cités plus bas (écoles,
    anciens employeurs).
    """
    t = _norm(text)
    head = len(t) * 0.15 + 300
    scores: dict[str, float] = {}
    for site, pats in _SITE_PATTERNS.items():
        for p in pats:
            for m in p.finditer(t):
                scores[site] = scores.get(site, 0) + (5 if m.start() < head else 1)
    return max(scores, key=scores.get) if scores else None


_FR_WORDS = {"le", "la", "les", "des", "et", "en", "du", "pour", "avec", "dans", "sur", "une", "au", "aux"}
_EN_WORDS = {"the", "and", "of", "to", "in", "for", "with", "on", "at", "as", "by", "from", "an"}


def document_language(text: str) -> str | None:
    """Langue de rédaction du CV : quelqu'un qui écrit son CV en français parle français."""
    words = re.findall(r"[a-zà-ÿ]+", _norm(text))
    if len(words) < 40:
        return None
    fr = sum(w in _FR_WORDS for w in words)
    en = sum(w in _EN_WORDS for w in words)
    if max(fr, en) < 0.04 * len(words):
        return None
    return "Français" if fr >= en else "Anglais"


# Mots qui ne peuvent pas faire partie d'un nom : titres de rubrique, intitulés de poste.
_NOT_NAME = {
    "cv", "curriculum", "vitae", "resume", "résumé", "profil", "profile", "contact", "coordonnées", "about",
    "compétences", "competences", "skills", "hard", "soft", "langues", "languages", "certifications",
    "formation", "formations", "éducation", "education", "expérience", "expériences", "experience", "experiences",
    "projets", "projects", "centres", "centre", "intérêt", "intérêts", "interets", "loisirs", "hobbies",
    "références", "references", "summary", "objectif", "parcours", "diplômes", "outils", "atouts", "qualités",
    "data", "engineer", "développeur", "développeuse", "developer", "consultant", "consultante", "analyst",
    "ingénieur", "ingénieure", "alternance", "alternant", "alternante", "stage", "stagiaire", "étudiant",
    "étudiante", "manager", "chef", "projet", "mastère", "master", "bachelor", "web", "full", "stack",
}

# Mots qui signalent un intitulé de poste ou de profil.
_ROLE_WORDS = re.compile(
    r"\b(engineer|ingénieure?|développeu(?:r|se)|developer|dev|analyste?|analyst|consultante?|scientist|"
    r"architecte?|chef de projet|project manager|product owner|po|scrum master|data|devops|administrat(?:eur|rice)|"
    r"responsable|manager|directeur|directrice|étudiante?|alternante?|stagiaire|business analyst|"
    r"full[- ]?stack|front[- ]?end|back[- ]?end|ux|ui|designer|technicien(?:ne)?)\b")

_CONTACT = re.compile(r"@|https?:|www\.|linkedin|github|\+?\d[\d .]{7,}")


def _as_name(line: str) -> str | None:
    """Renvoie la ligne mise en forme si elle ressemble à « Prénom Nom », sinon None."""
    line = re.sub(r"\s+", " ", line).strip()
    words = line.split()
    if not 2 <= len(words) <= 4 or len(line) > 40:
        return None
    # Lettres, tirets et apostrophes uniquement : exclut listes de compétences, e-mails, dates…
    if not re.fullmatch(r"[A-Za-zÀ-ÖØ-öø-ÿ'’ -]+", line):
        return None
    if any(w.lower().strip("'’") in _NOT_NAME for w in re.split(r"[\s'’-]+", line)) or extract_skills(line):
        return None
    if not all(w[0].isupper() for w in words):
        return None
    return " ".join("-".join(p.capitalize() for p in w.split("-")) if w.isupper() else w for w in words)


def _hint_tokens(filename: str, text: str) -> set[str]:
    """Mots du nom de fichier et de l'adresse e-mail : ils contiennent souvent le nom de la personne."""
    sources = [filename] + re.findall(r"([\w.+-]+)@", text)
    words = set()
    for s in sources:
        words |= {w for w in re.findall(r"[a-zà-ÿ]{3,}", _norm(s)) if w not in _NOT_NAME}
    return words - {"pdf", "docx", "txt", "pro", "contact", "mail", "gmail"}


def _pdf_blocks(data: bytes) -> tuple[list[dict], float]:
    """Titres de la première page, en recollant les lignes d'un même titre (« Anthony » + « NGUYEN »).

    Renvoie les groupes {texte, taille, y, rubrique} et la hauteur de la page.
    """
    import pymupdf
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        if not doc.page_count:
            return [], 1.0
        page = doc[0]
        lines = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                text = "".join(s["text"] for s in line["spans"]).strip()
                if text:
                    x0, y0, x1, y1 = line["bbox"]
                    lines.append({"text": text, "size": round(max(s["size"] for s in line["spans"]), 1),
                                  "x0": x0, "x1": x1, "y0": y0, "y1": y1})
        height = page.rect.height
    groups: list[dict] = []
    for ln in sorted(lines, key=lambda l: (l["y0"], l["x0"])):
        for g in groups:
            same_size = abs(g["size"] - ln["size"]) < 0.5
            overlap = min(g["x1"], ln["x1"]) - max(g["x0"], ln["x0"]) > 0
            close = 0 <= ln["y0"] - g["y1"] < ln["size"] * 0.9
            if same_size and overlap and close and len(g["lines"]) < 3:
                g["lines"].append(ln["text"])
                g.update(x0=min(g["x0"], ln["x0"]), x1=max(g["x1"], ln["x1"]), y1=ln["y1"])
                break
        else:
            groups.append({**ln, "lines": [ln["text"]]})
    # Une taille utilisée par 3 groupes ou plus = style des rubriques (FORMATION, LANGUES…), ou du corps de texte.
    counts: dict[float, int] = {}
    for g in groups:
        counts[g["size"]] = counts.get(g["size"], 0) + 1
    body = sorted(l["size"] for l in lines)[len(lines) // 2] if lines else 0
    out = [{"text": " ".join(g["lines"]), "titre_text": " · ".join(g["lines"]), "size": g["size"], "y": g["y0"],
            "rubrique": counts[g["size"]] >= 3, "titre": g["size"] >= body * 1.3}
           for g in groups]
    return sorted(out, key=lambda g: (-g["size"], g["y"])), height


def extract_profile(text: str, filename: str = "", pdf_data: bytes | None = None) -> dict:
    """Devine le nom (« Prénom Nom ») et le titre du CV.

    Pour un PDF, on s'appuie sur la mise en page : le nom et le titre sont les plus gros textes du
    haut de la première page, avec un style que n'ont pas les rubriques. L'ordre du texte brut,
    lui, suit les colonnes et n'est pas fiable. Le nom de fichier et l'e-mail confirment le nom.
    """
    hints = _hint_tokens(filename, text)

    def score(name: str) -> int:
        return len(hints & set(_norm(name).replace("-", " ").split()))

    nom = titre = None
    blocks, height = (_pdf_blocks(pdf_data) if pdf_data else ([], 1.0))
    if blocks:
        names = [(b, n) for b in blocks if (n := _as_name(b["text"]))]
        # Un nom confirmé par le fichier ou l'e-mail l'emporte ; sinon le plus gros titre hors rubrique.
        names.sort(key=lambda bn: (-score(bn[1]), bn[0]["rubrique"], -bn[0]["size"], bn[0]["y"]))
        if names and (score(names[0][1]) or not names[0][0]["rubrique"]):
            name_block, nom = names[0]
        else:
            name_block = None
        for b in blocks:  # déjà triés du plus gros au plus petit
            if (b is not name_block and b["titre"] and not b["rubrique"] and b["y"] < height * 0.35
                    and not _CONTACT.search(b["text"]) and len(b["text"]) <= 90 and not _as_name(b["text"])):
                titre = b["titre_text"]
                break

    # Texte brut (DOCX, TXT, ou PDF sans mise en page exploitable).
    lines = [re.sub(r"\s+", " ", l).strip() for l in text.splitlines() if l.strip()][:40]
    if not nom:
        candidates = [n for n in map(_as_name, lines) if n]
        # Prénom et nom sur deux lignes consécutives.
        candidates += [n for a, b in zip(lines, lines[1:]) if " " not in a and " " not in b
                       and (n := _as_name(f"{a} {b}"))]
        if candidates:
            nom = max(candidates, key=lambda n: (score(n), -candidates.index(n)))
    if not titre:
        for l in lines[:15]:
            if (_ROLE_WORDS.search(_norm(l)) and len(l) <= 80 and not _CONTACT.search(l)
                    and not l.endswith(":") and l.count(",") < 2 and not _as_name(l)):
                titre = l
                break
    return {"nom": nom, "titre": _tidy_title(titre) if titre else None}


_ACRONYMS = {"IA", "AI", "BI", "IT", "PO", "UX", "UI", "SAP", "RH", "ERP", "CRM", "QA", "SQL", "API", "MBA",
             "EFREI", "EPITA", "ESGI", "HEC", "ESSEC", "EPFL", "HEIG", "HES", "IUT", "BTS", "DUT", "CDI", "CDD"}
_SMALL = {"de", "du", "des", "en", "et", "à", "a", "la", "le", "les", "au", "aux", "d", "l", "of", "and", "the", "in"}


def _tidy_title(title: str) -> str:
    """« ÉTUDIANT EN MASTÈRE DATA ENGINEER » -> « Étudiant en Mastère Data Engineer » (sigles conservés)."""
    title = re.sub(r"\s+", " ", title).strip(" -–·|")
    letters = [c for c in title if c.isalpha()]
    if not letters or sum(c.isupper() for c in letters) / len(letters) < 0.8:
        return title

    def fix(word: str, first: bool) -> str:
        core = word.strip(",.;:()&-–'’")
        if core.upper() in _ACRONYMS or not core:
            return word
        if not first and core.lower() in _SMALL:
            return word.lower()
        # « L’EFREI » : élision + sigle
        m = re.match(r"^([A-Za-zÀ-ÿ])['’](.+)$", word)
        if m and m.group(2).strip(",.").upper() in _ACRONYMS:
            return m.group(1).lower() + "’" + m.group(2)
        return word[:1].upper() + word[1:].lower()

    return " ".join(fix(w, i == 0) for i, w in enumerate(title.split()))


def extract_name(text: str, filename: str = "", pdf_data: bytes | None = None) -> str | None:
    return extract_profile(text, filename, pdf_data)["nom"]


# --------------------------------------------------------------------------
# Expérience professionnelle
# --------------------------------------------------------------------------
_MONTHS = {
    "janvier": 1, "janv": 1, "jan": 1, "january": 1, "février": 2, "fevrier": 2, "févr": 2, "fev": 2, "fév": 2,
    "feb": 2, "february": 2, "mars": 3, "mar": 3, "march": 3, "avril": 4, "avr": 4, "apr": 4, "april": 4,
    "mai": 5, "may": 5, "juin": 6, "jun": 6, "june": 6, "juillet": 7, "juil": 7, "jul": 7, "july": 7,
    "août": 8, "aout": 8, "aug": 8, "august": 8, "septembre": 9, "sept": 9, "sep": 9, "september": 9,
    "octobre": 10, "oct": 10, "october": 10, "novembre": 11, "nov": 11, "november": 11,
    "décembre": 12, "decembre": 12, "déc": 12, "dec": 12, "december": 12,
}
_M = "|".join(sorted(map(re.escape, _MONTHS), key=len, reverse=True))
_NOW = r"présent|present|aujourd['’]hui|actuel(?:lement)?|en cours|now|current|ce jour"
_RANGE = re.compile(
    rf"(?<![\w])(?:(?P<m1>{_M})\.?\s*)?(?:(?P<m1n>\d{{1,2}})/)?(?P<y1>(?:19|20)\d\d)?\s*(?:-|–|—|à|au|to|>)\s*"
    rf"(?:(?P<m2>{_M})\.?\s*)?(?:(?P<m2n>\d{{1,2}})/)?(?P<y2>(?:19|20)\d\d|{_NOW})(?![\w])")
_DURATION = re.compile(r"(?<![\w])(\d{1,2}(?:[.,]5)?)\s*(mois|months?|semaines?|weeks?|ans?|années?|years?)(?![\w])")
_EXPLICIT = re.compile(
    r"(\d{1,2})\s*\+?\s*(?:ans|années|years)\s+(?:d['’]\s*)?(?:expérience|experience|of experience)"
    r"|(?:expérience|experience)\s+(?:professionnelle\s+)?(?:de|of)\s+(\d{1,2})\s*\+?\s*(?:ans|années|years)")
# Une ligne de poste : contrat, statut ou intitulé.
_JOB = re.compile(
    r"\b(stage|stagiaire|intern(?:ship)?|alternan(?:ce|t|te)|apprenti(?:ssage)?|cdi|cdd|freelance|indépendante?|"
    r"service civique|intérim|consultante?|développeu(?:r|se)|developer|engineer|ingénieure? (?:data|logiciel|"
    r"études|d['’]affaires|commercial|système|réseau|cloud)|analyste?|analyst|chef de projet|project manager|"
    r"product owner|manager|responsable|directeu(?:r|rice)|technicien(?:ne)?|assistante?|chargée? d)\b")
# Une ligne de formation : à ne pas compter, même avec des dates.
_SCHOOL = re.compile(
    r"\b(cycle|mast[èe]re|master|msc|bachelor|licence|bts|dut|but|bac|baccalaur[ée]at|pr[ée]pa|pr[ée]paratoire|"
    r"[ée]cole|university|universit[ée]|dipl[ôo]me|semestre|\d(?:e|[èe]me|er|re) ann[ée]e|cours|formation|"
    r"exchange|erasmus|[ée]tudes?|student|[ée]tudiante?)\b")
# Ce que la personne cherche, pas ce qu'elle a fait.
_WISH = re.compile(r"\b(recherche|cherche|looking|seeking|disponible|à partir|début|rythme|souhaite)\b")


def _months_of_duration(n: str, unit: str) -> float:
    n = float(n.replace(",", "."))
    if unit.startswith(("semaine", "week")):
        return n / 4.33
    if unit.startswith(("an", "year")):
        return n * 12
    return n


def _range_to_interval(m: re.Match, today: date) -> tuple[int, int] | None:
    """Plage de dates -> (début, fin) en mois absolus (année × 12 + mois)."""
    y2 = m["y2"]
    if re.fullmatch(_NOW, y2):
        end_y, end_m = today.year, today.month
    else:
        end_y = int(y2)
        end_m = _MONTHS.get((m["m2"] or "").lower()) or (int(m["m2n"]) if m["m2n"] else 12)
    start_y = int(m["y1"]) if m["y1"] else end_y
    if m["m1"] or m["m1n"]:
        start_m = _MONTHS.get((m["m1"] or "").lower()) or int(m["m1n"])
    elif m["y1"]:
        start_m = 1
    else:
        return None  # « - 2024 » sans début : inexploitable
    start, end = start_y * 12 + start_m, end_y * 12 + end_m
    if not 0 <= end - start <= 12 * 45:
        return None
    return start, end + 1  # mois de fin inclus


def extract_experience(text: str, today: date | None = None) -> tuple[int, str]:
    """Années d'expérience professionnelle et explication courte.

    1. Une phrase explicite (« 8 ans d'expérience ») fait foi.
    2. Sinon, on additionne les postes : stages, alternances, CDI, service civique…
       - une durée écrite (« Stage de 7 mois ») est prise telle quelle ;
       - sinon la plage de dates (« Janv. 2019 – Présent »), en fusionnant les périodes qui se chevauchent.
       Les lignes de formation, l'âge (« 22 ans ») et les loisirs (« 16 ans en club ») sont ignorés,
       tout comme le poste recherché (« Stage de 12 semaines, début avant juillet »).
    """
    today = today or date.today()
    t = _norm(text)
    explicit = [int(a or b) for a, b in _EXPLICIT.findall(t)]
    if explicit:
        years = max(explicit)
        return years, f"« {years} ans d'expérience » écrit dans le CV"

    lines = [_norm(l).strip() for l in text.splitlines() if l.strip()]
    now = today.year * 12 + today.month
    entries: list[tuple[int, str, object]] = []  # (n° de ligne, "durée" | "dates", valeur)
    undated = 0
    for i, line in enumerate(lines):
        prev = lines[i - 1] if i else ""
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        # Poste recherché, y compris quand la phrase continue à la ligne suivante (« … - » / « début avant juillet »).
        if _SCHOOL.search(line) or _WISH.search(line) or (re.search(r"[-–,:]$", line) and _WISH.search(nxt)):
            continue
        ranges = list(_RANGE.finditer(line))
        durations = _DURATION.findall(line)
        if not ranges and not durations:
            # « 2023 : Stagiaire chez X » : un poste daté d'une seule année, durée inconnue.
            if re.match(r"(?:19|20)\d\d\s*[:–-]", line) and _JOB.search(line):
                undated += 1
            continue
        # Le poste est sur la ligne, ou sur la ligne voisine (« Data Analyst – Banque X » puis les dates).
        job_near = any(_JOB.search(n) and not _SCHOOL.search(n) for n in (prev, nxt))
        if not _JOB.search(line) and not (ranges and job_near):
            continue
        if durations:
            n, unit = durations[0]
            entries.append((i, "durée", _months_of_duration(n, unit)))
            continue
        for r in ranges:
            iv = _range_to_interval(r, today)
            if iv and iv[0] <= now:  # une période future (« 2026 – 2028 ») n'est pas de l'expérience
                entries.append((i, "dates", (iv[0], min(iv[1], now + 1))))
                break

    # Une durée et des dates sur deux lignes voisines décrivent le même poste : on le compte une fois.
    jobs: list[dict] = []
    for i, kind, value in entries:
        last = jobs[-1] if jobs else None
        if last and i - last["ligne"] == 1 and kind not in last:
            last[kind], last["ligne"] = value, i
        else:
            jobs.append({"ligne": i, kind: value})
    written = sum(j["durée"] for j in jobs if "durée" in j)
    intervals = [j["dates"] for j in jobs if "durée" not in j]
    found = len(jobs)

    merged = 0
    for start, end in sorted(intervals):
        if merged and start < last_end:
            if end > last_end:
                merged += end - last_end
                last_end = end
        else:
            merged += end - start
            last_end = end
    months = written + merged
    if not found and not undated:
        return 0, "aucune expérience datée trouvée"
    years = min(45, int(months / 12 + 0.5))
    detail = f"{months:.0f} mois" if months < 24 else f"{months / 12:.1f} ans".replace(".", ",")
    detail = f"{detail} cumulés sur {found} poste{'s' if found > 1 else ''}" if found else "0 mois daté"
    if undated:
        detail += f", + {undated} poste{'s' if undated > 1 else ''} sans durée indiquée"
    return years, detail


def extract_years(text: str) -> int | None:
    """Compatibilité : années d'expérience, ou None si rien n'est trouvé."""
    years, detail = extract_experience(text)
    return years if not detail.startswith("aucune") else None


def read_document(path_or_bytes, filename: str = "") -> str:
    """Lit un PDF, un DOCX ou un TXT et renvoie le texte brut."""
    name = (filename or str(path_or_bytes)).lower()
    if name.endswith(".pdf"):
        import pymupdf
        doc = pymupdf.open(stream=path_or_bytes, filetype="pdf") if isinstance(path_or_bytes, bytes) else pymupdf.open(path_or_bytes)
        return "\n".join(page.get_text() for page in doc)
    if name.endswith(".docx"):
        import io
        import docx  # python-docx
        f = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
        return "\n".join(p.text for p in docx.Document(f).paragraphs)
    if isinstance(path_or_bytes, bytes):
        return path_or_bytes.decode("utf-8", errors="ignore")
    return Path(path_or_bytes).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Modèle de données
# --------------------------------------------------------------------------
@dataclass
class Consultant:
    id: str
    nom: str
    titre: str
    site: str
    annees_experience: int
    disponible_dans_jours: int
    cv: str
    competences: set[str] = field(default_factory=set)
    langues: set[str] = field(default_factory=set)

    def __post_init__(self):
        self.competences = self.competences or extract_skills(self.cv)
        self.langues = self.langues or extract_languages(self.cv)


@dataclass
class Mission:
    id: str
    titre: str
    client: str
    site: str
    debut_dans_jours: int
    seniorite_min: int
    description: str
    obligatoires: list[str]
    souhaitees: list[str]
    langues: list[str]
    retenu: str | None = None  # consultant retenu historiquement (pour l'évaluation)


def load_consultants() -> list[Consultant]:
    raw = json.loads((DATA_DIR / "consultants.json").read_text(encoding="utf-8"))
    return [Consultant(**c) for c in raw]


def load_missions() -> list[Mission]:
    raw = json.loads((DATA_DIR / "missions.json").read_text(encoding="utf-8"))
    return [Mission(**m) for m in raw]


# --------------------------------------------------------------------------
# Similarité sémantique
# --------------------------------------------------------------------------
class SemanticEncoder:
    """sentence-transformers si disponible, sinon TF-IDF."""

    MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, force_tfidf: bool = False):
        self.backend = "tfidf"
        self._model = None
        if not force_tfidf:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.MODEL)
                self.backend = f"embeddings ({self.MODEL})"
            except Exception:
                self._model = None

    def similarities(self, query: str, docs: list[str]) -> np.ndarray:
        if self._model is not None:
            emb = self._model.encode([query] + docs, normalize_embeddings=True)
            return emb[1:] @ emb[0]
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode")
        m = vec.fit_transform([query] + docs)
        return cosine_similarity(m[0], m[1:]).ravel()


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
@dataclass
class MatchResult:
    consultant: Consultant
    score: float
    detail: dict[str, float]
    couvertes: list[str]
    manquantes: list[str]
    bonus: list[str]
    explication: str


def _seniority(years: int, minimum: int) -> float:
    if minimum <= 0:
        return 1.0
    return float(min(1.0, years / minimum))


def _availability(dispo_days: int, start_days: int) -> float:
    late = max(0, dispo_days - start_days)
    return float(max(0.0, 1 - late / 60))  # 0 si plus de 2 mois de retard


def _lang_site(c: Consultant, m: Mission) -> float:
    langs = 1.0 if not m.langues else len(set(m.langues) & c.langues) / len(m.langues)
    site = 1.0 if c.site == m.site else 0.5
    return 0.7 * langs + 0.3 * site


def match(mission: Mission, consultants: list[Consultant], encoder: SemanticEncoder,
          weights: dict[str, float] | None = None) -> list[MatchResult]:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    total_w = sum(w.values()) or 1.0
    query = f"{mission.titre}. {mission.description} Compétences : {', '.join(mission.obligatoires + mission.souhaitees)}"
    sims = encoder.similarities(query, [c.cv for c in consultants])
    # Normalisation relative : le meilleur candidat sémantique vaut 1
    sem = sims / sims.max() if sims.max() > 0 else sims

    results = []
    for c, s in zip(consultants, sem):
        must = set(mission.obligatoires)
        couvertes = sorted(must & c.competences)
        manquantes = sorted(must - c.competences)
        bonus = sorted(set(mission.souhaitees) & c.competences)
        detail = {
            "sem": float(s),
            "must": len(couvertes) / len(must) if must else 1.0,
            "senior": _seniority(c.annees_experience, mission.seniorite_min),
            "dispo": _availability(c.disponible_dans_jours, mission.debut_dans_jours),
            "lang": _lang_site(c, mission),
        }
        # 95 points pour les critères pondérés, 5 points pour les compétences souhaitées
        base = sum(w[k] * detail[k] for k in w) / total_w
        extra = len(bonus) / len(mission.souhaitees) if mission.souhaitees else 1.0
        score = 95 * base + 5 * extra
        results.append(MatchResult(c, round(score, 1), detail, couvertes, manquantes, bonus,
                                   _explain(c, mission, couvertes, manquantes, bonus, detail)))
    return sorted(results, key=lambda r: r.score, reverse=True)


def _explain(c, m, couvertes, manquantes, bonus, d) -> str:
    parts = []
    if couvertes:
        parts.append("couvre " + ", ".join(couvertes))
    if manquantes:
        parts.append("manque " + ", ".join(manquantes))
    if bonus:
        parts.append("bonus " + ", ".join(bonus))
    if d["senior"] < 1:
        parts.append(f"{c.annees_experience} an(s) d'expérience pour {m.seniorite_min} demandé(s)")
    late = c.disponible_dans_jours - m.debut_dans_jours
    parts.append("disponible à temps" if late <= 0 else f"disponible {late} jours après le début")
    text = " ; ".join(parts)
    return text[:1].upper() + text[1:] + "."


if __name__ == "__main__":
    consultants, missions = load_consultants(), load_missions()
    enc = SemanticEncoder()
    print(f"Moteur sémantique : {enc.backend}\n")
    for m in missions:
        print(f"=== {m.titre} ({m.client}) ===")
        for r in match(m, consultants, enc)[:3]:
            print(f"  {r.score:5.1f}  {r.consultant.nom:<22} {r.explication}")
        print()
