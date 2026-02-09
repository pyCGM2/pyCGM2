import sys
from PySide6.QtWidgets import QApplication, QFileDialog


def uiGetDir(title: str = "Select a Data Folder",
                  start_dir: str = "") -> str | None:
    """
    Open a native dialog to select a directory.

    Returns
    -------
    str | None
        Selected directory path, or None if cancelled.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    directory = QFileDialog.getExistingDirectory(
        None,
        title,
        start_dir,
        QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
    )

    return directory if directory else None