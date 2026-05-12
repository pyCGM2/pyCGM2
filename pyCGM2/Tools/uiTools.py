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