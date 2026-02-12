# coding: utf-8
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable, List, Tuple, Any


# ============================================================
# Domain models
# ============================================================

@dataclass(frozen=True)
class StorageRoot:
    id: int
    name: str
    root_path: str  # stored as TEXT in DB


@dataclass(frozen=True)
class Patient:
    ipp: str
    folder_name: str
    storage_root_id: int


@dataclass(frozen=True)
class Session:
    id: int
    ipp: str
    session_index: int
    folder_name: str
    session_date: Optional[str] = None  # "YYYY-MM-DD" (optional)
    notes: Optional[str] = None


@dataclass(frozen=True)
class Artifact:
    id: int
    session_id: int
    data_type: str
    rel_path: str
    label: Optional[str] = None


@dataclass(frozen=True)
class ArtifactMeta:
    id: int
    artifact_id: int
    section: Optional[str]
    key: str
    value: Optional[str]
    value_type: Optional[str] = None

# ============================================================
# Adapter pattern: convert stored TEXT paths into Path objects
# ============================================================

class PathAdapter:
    """Adapter between DB TEXT paths and pathlib.Path."""
    @staticmethod
    def to_db(path: Path | str) -> str:
        return str(path)

    @staticmethod
    def from_db(path_str: str) -> Path:
        return Path(path_str)


# ============================================================
# SQLite connection / schema
# ============================================================

class SQLiteConnectionFactory:
    def __init__(self, db_path: Path | str):
        self._db_path = Path(db_path)

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self._db_path)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON;")
        return con


class SchemaManager:
    def __init__(self, con: sqlite3.Connection):
        self._con = con

    def ensure_schema(self) -> None:
        self._con.executescript(
            """
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS storage_root (
              id         INTEGER PRIMARY KEY AUTOINCREMENT,
              name       TEXT NOT NULL UNIQUE,
              root_path  TEXT NOT NULL,
              created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS patient (
              ipp             TEXT PRIMARY KEY,
              folder_name     TEXT NOT NULL,
              storage_root_id INTEGER NOT NULL,
              created_at      TEXT NOT NULL DEFAULT (datetime('now')),
              updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
              FOREIGN KEY (storage_root_id) REFERENCES storage_root(id)
            );

            CREATE TABLE IF NOT EXISTS session (
              id            INTEGER PRIMARY KEY AUTOINCREMENT,
              ipp           TEXT NOT NULL,
              session_index INTEGER NOT NULL,
              folder_name   TEXT NOT NULL,
              session_date  TEXT,
              notes         TEXT,
              created_at    TEXT NOT NULL DEFAULT (datetime('now')),
              updated_at    TEXT NOT NULL DEFAULT (datetime('now')),
              FOREIGN KEY (ipp) REFERENCES patient(ipp) ON DELETE CASCADE,
              UNIQUE (ipp, session_index),
              UNIQUE (ipp, folder_name)
            );

            CREATE INDEX IF NOT EXISTS idx_session_ipp ON session(ipp);

            CREATE TABLE IF NOT EXISTS artifact (
              id         INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id INTEGER NOT NULL,
              data_type  TEXT NOT NULL,
              rel_path   TEXT NOT NULL,
              label      TEXT,
              created_at TEXT NOT NULL DEFAULT (datetime('now')),
              FOREIGN KEY (session_id) REFERENCES session(id) ON DELETE CASCADE,
              UNIQUE (session_id, data_type, rel_path)
            );

            CREATE INDEX IF NOT EXISTS idx_artifact_session ON artifact(session_id);
            CREATE INDEX IF NOT EXISTS idx_artifact_type ON artifact(data_type);
            
            CREATE TABLE IF NOT EXISTS artifact_metadata (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            artifact_id  INTEGER NOT NULL,
            section      TEXT,                 -- ex: "TRIAL_INFO"
            key          TEXT NOT NULL,        -- ex: "Task"
            value        TEXT,                 -- valeur brute string
            value_type   TEXT,                 -- "str" | "int" | "float" | "bool" | "datetime" | "csv" ...
            created_at   TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (artifact_id) REFERENCES artifact(id) ON DELETE CASCADE,
            UNIQUE (artifact_id, section, key)
            );

            CREATE INDEX IF NOT EXISTS idx_meta_artifact ON artifact_metadata(artifact_id);
            CREATE INDEX IF NOT EXISTS idx_meta_key ON artifact_metadata(section, key);

            
            """
        )
        self._con.commit()


# ============================================================
# Repositories (CRUD)
# ============================================================

class StorageRootRepository:
    def __init__(self, con: sqlite3.Connection, path_adapter: PathAdapter = PathAdapter()):
        self._con = con
        self._path = path_adapter

    # ----------------------------
    # CREATE / UPDATE (by name)
    # ----------------------------
    def upsert(self, name: str, root_path: Path | str) -> StorageRoot:
        root_path_str = self._path.to_db(Path(root_path))
        self._con.execute(
            """
            INSERT INTO storage_root(name, root_path)
            VALUES(?, ?)
            ON CONFLICT(name) DO UPDATE SET root_path = excluded.root_path
            """,
            (name, root_path_str),
        )
        self._con.commit()
        return self.get_by_name(name)

    # ----------------------------
    # READ
    # ----------------------------
    def get_by_name(self, name: str) -> StorageRoot:
        row = self._con.execute(
            "SELECT id, name, root_path FROM storage_root WHERE name=?",
            (name,),
        ).fetchone()
        if row is None:
            raise KeyError(f"StorageRoot not found: name={name}")
        return StorageRoot(id=row["id"], name=row["name"], root_path=row["root_path"])

    def get_by_id(self, root_id: int) -> StorageRoot:
        row = self._con.execute(
            "SELECT id, name, root_path FROM storage_root WHERE id=?",
            (root_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"StorageRoot not found: id={root_id}")
        return StorageRoot(id=row["id"], name=row["name"], root_path=row["root_path"])

    def list_all(self) -> List[StorageRoot]:
        rows = self._con.execute(
            "SELECT id, name, root_path FROM storage_root ORDER BY name"
        ).fetchall()
        return [StorageRoot(id=r["id"], name=r["name"], root_path=r["root_path"]) for r in rows]

    # ----------------------------
    # DELETE
    # ----------------------------
    def delete(self, name: str) -> None:
        self._con.execute("DELETE FROM storage_root WHERE name=?", (name,))
        self._con.commit()

    # ----------------------------
    # UPDATE (by id) with UNIQUE-safe rename
    # ----------------------------
    def update_name_by_id(self, root_id: int, new_name: str) -> StorageRoot:
        """
        Rename a storage root using its id.

        - If new_name is already used by another row, raise ValueError (clearer than sqlite3.IntegrityError).
        - If root_id does not exist, raise KeyError.
        """
        # 1) ensure target exists
        _ = self.get_by_id(root_id)

        # 2) check UNIQUE(name) conflict
        existing = self._con.execute(
            "SELECT id FROM storage_root WHERE name=?",
            (new_name,),
        ).fetchone()

        if existing is not None and int(existing["id"]) != int(root_id):
            raise ValueError(f"Cannot rename storage_root id={root_id} to '{new_name}': name already used by id={existing['id']}")

        # 3) do update
        self._con.execute(
            """
            UPDATE storage_root
            SET name = ?
            WHERE id = ?
            """,
            (new_name, root_id),
        )
        self._con.commit()

        return self.get_by_id(root_id)




class PatientRepository:
    def __init__(self, con: sqlite3.Connection):
        self._con = con

    def upsert(self, ipp: str, folder_name: str, storage_root_id: int) -> Patient:
        self._con.execute(
            """
            INSERT INTO patient(ipp, folder_name, storage_root_id)
            VALUES(?, ?, ?)
            ON CONFLICT(ipp) DO UPDATE SET
              folder_name = excluded.folder_name,
              storage_root_id = excluded.storage_root_id,
              updated_at = datetime('now')
            """,
            (ipp, folder_name, storage_root_id),
        )
        self._con.commit()
        return self.get(ipp)

    def get(self, ipp: str) -> Patient:
        row = self._con.execute(
            "SELECT ipp, folder_name, storage_root_id FROM patient WHERE ipp=?",
            (ipp,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Patient not found: ipp={ipp}")
        return Patient(ipp=row["ipp"], folder_name=row["folder_name"], storage_root_id=row["storage_root_id"])

    def list_all(self) -> List[Patient]:
        rows = self._con.execute(
            "SELECT ipp, folder_name, storage_root_id FROM patient ORDER BY ipp"
        ).fetchall()
        return [Patient(ipp=r["ipp"], folder_name=r["folder_name"], storage_root_id=r["storage_root_id"]) for r in rows]

    def delete(self, ipp: str) -> None:
        self._con.execute("DELETE FROM patient WHERE ipp=?", (ipp,))
        self._con.commit()


class SessionRepository:
    def __init__(self, con: sqlite3.Connection):
        self._con = con

    def create_or_update(
        self,
        ipp: str,
        session_index: int,
        folder_name: str,
        session_date: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Session:
        self._con.execute(
            """
            INSERT INTO session(ipp, session_index, folder_name, session_date, notes)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(ipp, session_index) DO UPDATE SET
              folder_name = excluded.folder_name,
              session_date = excluded.session_date,
              notes = excluded.notes,
              updated_at = datetime('now')
            """,
            (ipp, session_index, folder_name, session_date, notes),
        )
        self._con.commit()
        return self.get_by_index(ipp, session_index)

    def get_by_index(self, ipp: str, session_index: int) -> Session:
        row = self._con.execute(
            """
            SELECT id, ipp, session_index, folder_name, session_date, notes
            FROM session
            WHERE ipp=? AND session_index=?
            """,
            (ipp, session_index),
        ).fetchone()
        if row is None:
            raise KeyError(f"Session not found: ipp={ipp}, session_index={session_index}")
        return Session(
            id=row["id"],
            ipp=row["ipp"],
            session_index=row["session_index"],
            folder_name=row["folder_name"],
            session_date=row["session_date"],
            notes=row["notes"],
        )

    def list_by_patient(self, ipp: str) -> List[Session]:
        rows = self._con.execute(
            """
            SELECT id, ipp, session_index, folder_name, session_date, notes
            FROM session
            WHERE ipp=?
            ORDER BY session_index
            """,
            (ipp,),
        ).fetchall()
        return [
            Session(
                id=r["id"],
                ipp=r["ipp"],
                session_index=r["session_index"],
                folder_name=r["folder_name"],
                session_date=r["session_date"],
                notes=r["notes"],
            )
            for r in rows
        ]

    def delete_by_index(self, ipp: str, session_index: int) -> None:
        self._con.execute(
            "DELETE FROM session WHERE ipp=? AND session_index=?",
            (ipp, session_index),
        )
        self._con.commit()


class ArtifactRepository:
    def __init__(self, con: sqlite3.Connection):
        self._con = con

    def upsert(self, session_id: int, data_type: str, rel_path: str, label: Optional[str] = None) -> Artifact:
        self._con.execute(
            """
            INSERT INTO artifact(session_id, data_type, rel_path, label)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(session_id, data_type, rel_path) DO UPDATE SET
              label = excluded.label
            """,
            (session_id, data_type, rel_path, label),
        )
        self._con.commit()
        return self.get(session_id, data_type, rel_path)

    def get(self, session_id: int, data_type: str, rel_path: str) -> Artifact:
        row = self._con.execute(
            """
            SELECT id, session_id, data_type, rel_path, label
            FROM artifact
            WHERE session_id=? AND data_type=? AND rel_path=?
            """,
            (session_id, data_type, rel_path),
        ).fetchone()
        if row is None:
            raise KeyError(f"Artifact not found: session_id={session_id}, data_type={data_type}, rel_path={rel_path}")
        return Artifact(
            id=row["id"],
            session_id=row["session_id"],
            data_type=row["data_type"],
            rel_path=row["rel_path"],
            label=row["label"],
        )

    def list_by_session(self, session_id: int) -> List[Artifact]:
        rows = self._con.execute(
            """
            SELECT id, session_id, data_type, rel_path, label
            FROM artifact
            WHERE session_id=?
            ORDER BY data_type, rel_path
            """,
            (session_id,),
        ).fetchall()
        return [
            Artifact(id=r["id"], session_id=r["session_id"], data_type=r["data_type"], rel_path=r["rel_path"], label=r["label"])
            for r in rows
        ]

    def delete(self, artifact_id: int) -> None:
        self._con.execute("DELETE FROM artifact WHERE id=?", (artifact_id,))
        self._con.commit()

class ArtifactMetadataRepository:
    def __init__(self, con: sqlite3.Connection):
        self._con = con

    def upsert(
        self,
        artifact_id: int,
        section: Optional[str],
        key: str,
        value: Optional[str],
        value_type: Optional[str] = None
    ) -> ArtifactMeta:
        self._con.execute(
            """
            INSERT INTO artifact_metadata(artifact_id, section, key, value, value_type)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(artifact_id, section, key) DO UPDATE SET
              value = excluded.value,
              value_type = excluded.value_type
            """,
            (artifact_id, section, key, value, value_type),
        )
        self._con.commit()
        return self.get(artifact_id, section, key)

    def get(self, artifact_id: int, section: Optional[str], key: str) -> ArtifactMeta:
        row = self._con.execute(
            """
            SELECT id, artifact_id, section, key, value, value_type
            FROM artifact_metadata
            WHERE artifact_id=? AND section IS ? AND key=?
            """,
            (artifact_id, section, key),
        ).fetchone()
        if row is None:
            raise KeyError(f"Metadata not found: artifact_id={artifact_id}, section={section}, key={key}")
        return ArtifactMeta(
            id=row["id"],
            artifact_id=row["artifact_id"],
            section=row["section"],
            key=row["key"],
            value=row["value"],
            value_type=row["value_type"],
        )

    def list_by_artifact(self, artifact_id: int) -> list[ArtifactMeta]:
        rows = self._con.execute(
            """
            SELECT id, artifact_id, section, key, value, value_type
            FROM artifact_metadata
            WHERE artifact_id=?
            ORDER BY section, key
            """,
            (artifact_id,),
        ).fetchall()
        return [ArtifactMeta(
            id=r["id"], artifact_id=r["artifact_id"], section=r["section"],
            key=r["key"], value=r["value"], value_type=r["value_type"]
        ) for r in rows]


# ============================================================
# High-level service: compute paths + convenience operations
# ============================================================

class DataIndexService:
    """
    Service layer to:
    - manage CRUD via repositories
    - reconstruct filesystem paths deterministically
    """
    def __init__(self, con: sqlite3.Connection):
        self._con = con
        self.storage_roots = StorageRootRepository(con)
        self.patients = PatientRepository(con)
        self.sessions = SessionRepository(con)
        self.artifacts = ArtifactRepository(con)
        self.artifact_meta = ArtifactMetadataRepository(con)

    def get_session_path(self, ipp: str, session_index: int) -> Path:
        row = self._con.execute(
            """
            SELECT r.root_path AS root_path, p.folder_name AS patient_folder, s.folder_name AS session_folder
            FROM session s
            JOIN patient p ON p.ipp = s.ipp
            JOIN storage_root r ON r.id = p.storage_root_id
            WHERE s.ipp=? AND s.session_index=?
            """,
            (ipp, session_index),
        ).fetchone()
        if row is None:
            raise KeyError(f"Cannot build session path: ipp={ipp}, session_index={session_index}")

        return PathAdapter.from_db(row["root_path"]) / row["patient_folder"] / row["session_folder"]

    def get_artifact_path(self, artifact_id: int) -> Path:
        row = self._con.execute(
            """
            SELECT a.rel_path AS rel_path, s.ipp AS ipp, s.session_index AS session_index
            FROM artifact a
            JOIN session s ON s.id = a.session_id
            WHERE a.id=?
            """,
            (artifact_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Artifact not found: id={artifact_id}")

        base = self.get_session_path(row["ipp"], row["session_index"])
        return base / row["rel_path"]


# ============================================================
# Example usage
# ============================================================

def main() -> None:
    db_path = Path("Z:\\Donnees_Nexus\\lamDB.sqlite")

    factory = SQLiteConnectionFactory(db_path)
    con = factory.connect()


    SchemaManager(con).ensure_schema()
    svc = DataIndexService(con)

    # 1) Define your Z: root (or any root)
    # 

    root = svc.storage_roots.upsert(name="Z_MAIN", root_path=r"Z:\Donnees_Nexus\AQM Adultes")  # adapt if your root is Z:\Data etc.
    # root = svc.storage_roots.get_by_name("Z_MAIN")

    svc.storage_roots.delete("AQM Adultes")
    rename = svc.storage_roots.update_name_by_id(root.id, new_name="AQM Adultes")


    # 2) Register a patient
    # If folder name == ipp, keep it identical. Otherwise store the actual folder name.
    patient = svc.patients.upsert(ipp="666", folder_name="ABARNOU Martin", storage_root_id=root.id)

    # 3) Register sessions
    s1 = svc.sessions.create_or_update(ipp=patient.ipp, session_index=1, folder_name="Session 1", session_date="2025-02-03")
    s2 = svc.sessions.create_or_update(ipp=patient.ipp, session_index=2, folder_name="Session 2c")

    # 4) Compute paths (deterministic)
    print("Session 1 path:", svc.get_session_path(patient.ipp, 1))
    print("Session 2 path:", svc.get_session_path(patient.ipp, 2))

    # 5) (Optional) Register artifacts inside a session folder (relative paths)
    a1 = svc.artifacts.upsert(session_id=s1.id, data_type="c3d", rel_path="trial01.c3d", label="Trial 01")
    a2 = svc.artifacts.upsert(session_id=s1.id, data_type="report_pdf", rel_path=r"Reports\summary.pdf")

    print("Artifact a1 full path:", svc.get_artifact_path(a1.id))

    # 6) List sessions
    for s in svc.sessions.list_by_patient(patient.ipp):
        print(f"Session {s.session_index}:", svc.get_session_path(patient.ipp, s.session_index))

    con.close()


