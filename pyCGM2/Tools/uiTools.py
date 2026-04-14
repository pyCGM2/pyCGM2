import sys
from PySide6.QtWidgets import QApplication, QFileDialog


def uiGetDir(title: str = "Select a Data Folder",
             start_dir: str = "") -> str | None:

    existing_app = QApplication.instance()
    app = existing_app if existing_app is not None else QApplication(sys.argv)
    created_locally = existing_app is None

    directory = QFileDialog.getExistingDirectory(
        None,
        title,
        start_dir,
        QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
    )

    if created_locally:
        app.quit()
        del app  # détruit le singleton pour libérer la place

    directory = directory.replace("/", "\\") + "\\"
    return directory if directory else None