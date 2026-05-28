import sys
import tkinter as tk
from tkinter import filedialog

def uiGetDir(title: str = "Select a Data Folder",
             start_dir: str = "") -> str | None:

    root = tk.Tk()
    root.withdraw()  # cache la fenêtre principale

    directory = filedialog.askdirectory(
        title=title,
        initialdir=start_dir
    )

    root.destroy()

    if not directory:
        return None

    return directory.replace("/", "\\") + "\\"

def uiGetFiles(title: str = "Select Files",
               start_dir: str = "",
               filetypes: list[tuple[str, str]] | None = None) -> list[str] | None:
    """
    Ouvre un dialogue pour sélectionner un ou plusieurs fichiers.
    
    Args:
        title: Titre du dialogue
        start_dir: Répertoire initial
        filetypes: Liste de tuples (description, extension) ex: [("CSV files", "*.csv"), ("All files", "*.*")]
    
    Returns:
        Liste des chemins de fichiers sélectionnés (avec backslashes) ou None si annulé

        files = uiTools.uiGetFiles(
        title="Sélectionner le fichier Bilan clinique",
        start_dir=sessionPath+oldReportFolder,
        filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
    )
    filenameBilanClinique = files[0] if files else None
    """
    root = tk.Tk()
    root.withdraw()  # cache la fenêtre principale

    files = filedialog.askopenfilenames(
        title=title,
        initialdir=start_dir,
        filetypes=filetypes or [("All files", "*.*")]
    )

    root.destroy()

    if not files:
        return None

    return [file.replace("/", "\\") for file in files]