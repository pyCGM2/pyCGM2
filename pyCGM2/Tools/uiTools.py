"""User-interface tools (tkinter file and folder dialogs)."""
import sys
import tkinter as tk
from tkinter import filedialog

def uiGetDir(title: str = "Select a Data Folder",
             start_dir: str = "") -> str | None:
    """Open a dialog to select a folder.

    Args:
        title (str, optional): Dialog title. Defaults to ``"Select a Data Folder"``.
        start_dir (str, optional): Initial folder.

    Returns:
        Optional[str]: Selected folder (with backslashes and a trailing backslash), or None if cancelled.
    """

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
    """Open a dialog to select one or several files.

    Args:
        title (str, optional): Dialog title. Defaults to ``"Select Files"``.
        start_dir (str, optional): Initial folder.
        filetypes (list, optional): List of ``(description, pattern)`` tuples,
            e.g. ``[("CSV files", "*.csv"), ("All files", "*.*")]``.

    Returns:
        Optional[List[str]]: Selected file paths (with backslashes), or None if cancelled.

    Example:

    .. code-block:: python

        files = uiTools.uiGetFiles(
            title="Select the clinical examination file",
            start_dir=sessionPath + oldReportFolder,
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")])
        filename = files[0] if files else None
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