"""Stockage des CV importés dans une base SQLite locale (data/hortismatch.db).

Chaque import conserve le fichier d'origine, le texte extrait et le profil validé.
Un même fichier réimporté met à jour le profil existant au lieu de créer un doublon.
"""
import hashlib
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from matcher import DATA_DIR, Consultant

DB_PATH = DATA_DIR / "hortismatch.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cv_importes (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    nom                   TEXT NOT NULL,
    titre                 TEXT NOT NULL,
    site                  TEXT NOT NULL,
    annees_experience     INTEGER NOT NULL,
    disponible_dans_jours INTEGER NOT NULL,
    langues               TEXT NOT NULL,   -- séparées par des virgules
    cv_texte              TEXT NOT NULL,
    fichier_nom           TEXT NOT NULL,
    fichier               BLOB NOT NULL,   -- le PDF / DOCX / TXT d'origine
    fichier_sha256        TEXT NOT NULL UNIQUE,
    importe_le            TEXT NOT NULL
)
"""


def _connect(path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(path or DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    return conn


def save_cv(consultant: Consultant, filename: str, data: bytes, path: Path | None = None) -> int:
    """Enregistre (ou met à jour) un CV importé et renvoie son identifiant."""
    digest = hashlib.sha256(data).hexdigest()
    row = (consultant.nom, consultant.titre, consultant.site, consultant.annees_experience,
           consultant.disponible_dans_jours, ",".join(sorted(consultant.langues)), consultant.cv,
           filename, data, digest, datetime.now().isoformat(timespec="seconds"))
    with closing(_connect(path)) as conn, conn:
        conn.execute("""
            INSERT INTO cv_importes (nom, titre, site, annees_experience, disponible_dans_jours, langues,
                                     cv_texte, fichier_nom, fichier, fichier_sha256, importe_le)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(fichier_sha256) DO UPDATE SET
                nom = excluded.nom, titre = excluded.titre, site = excluded.site,
                annees_experience = excluded.annees_experience,
                disponible_dans_jours = excluded.disponible_dans_jours, langues = excluded.langues,
                fichier_nom = excluded.fichier_nom, importe_le = excluded.importe_le
        """, row)
        return conn.execute("SELECT id FROM cv_importes WHERE fichier_sha256 = ?", (digest,)).fetchone()[0]


def load_imported(path: Path | None = None) -> list[Consultant]:
    with closing(_connect(path)) as conn:
        rows = conn.execute("SELECT * FROM cv_importes ORDER BY id").fetchall()
    return [Consultant(id=f"db{r['id']}", nom=r["nom"], titre=r["titre"], site=r["site"],
                       annees_experience=r["annees_experience"], disponible_dans_jours=r["disponible_dans_jours"],
                       cv=r["cv_texte"], langues=set(filter(None, r["langues"].split(","))))
            for r in rows]


def list_imported(path: Path | None = None) -> list[dict]:
    """Métadonnées des CV stockés (sans le fichier), pour l'affichage."""
    with closing(_connect(path)) as conn:
        rows = conn.execute("""SELECT id, nom, titre, site, fichier_nom, importe_le, length(fichier) AS taille
                               FROM cv_importes ORDER BY id""").fetchall()
    return [dict(r) for r in rows]


def get_file(cv_id: int, path: Path | None = None) -> tuple[str, bytes] | None:
    with closing(_connect(path)) as conn:
        r = conn.execute("SELECT fichier_nom, fichier FROM cv_importes WHERE id = ?", (cv_id,)).fetchone()
    return (r["fichier_nom"], r["fichier"]) if r else None


def delete_cv(cv_id: int, path: Path | None = None) -> None:
    with closing(_connect(path)) as conn, conn:
        conn.execute("DELETE FROM cv_importes WHERE id = ?", (cv_id,))
