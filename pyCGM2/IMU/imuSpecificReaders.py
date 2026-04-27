import pandas as pd

import pyCGM2
from pyCGM2.Utils import readers


def read_blueTridentCsv(filepath: str) -> pd.DataFrame:
    imuTranslators  = readers.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER +"IMU\\","viconBlueTrident.translators")
    translators = imuTranslators["Translators"]

    df = pd.read_csv(filepath)
    
    # inverser le dict : {ancien_nom: nouveau_nom}
    rename_map = {v: k for k, v in translators.items()}

    df.rename(columns=rename_map, inplace=True)

    return df


def read_imu_kinventCsv(filepath: str) -> pd.DataFrame:
    """Lit un CSV sans header et renomme les colonnes selon translators.

    Args:
        filepath:    Chemin vers le fichier CSV.

    Returns:
        DataFrame avec uniquement les colonnes spécifiées, renommées.
    """
    imuTranslators  = readers.openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER +"IMU\\","kiventKForceIMU.translators")
    translators = imuTranslators["Translators"]

    # Facteurs de conversion : (offset, scale) → (raw - offset) * scale
    conversions = {
        "Quaternion.X":      (32768, 1 / 16384),       # sans unité [-2, 2]
        "Quaternion.Y":      (32768, 1 / 16384),
        "Quaternion.Z":      (32768, 1 / 16384),
        "Quaternion.R":      (32768, 1 / 16384),
        "Accel.X":           (32768, 8 / 32768),        # g
        "Accel.Y":           (32768, 8 / 32768),
        "Accel.Z":           (32768, 8 / 32768),
        "AngularVelocity.X": (32768, 2000.0 / 32768),   # deg/s
        "AngularVelocity.Y": (32768, 2000.0 / 32768),
        "AngularVelocity.Z": (32768, 2000.0 / 32768),
        "Magneto.X":         (32768, 2500.0 / 32768),   # µT
        "Magneto.Y":         (32768, 2500.0 / 32768),
        "Magneto.Z":         (32768, 2500.0 / 32768),
    }


    df_raw = pd.read_csv(filepath, header=None)
    df = df_raw.iloc[:, list(translators.values())].copy()
    df.columns = pd.Index(list(translators.keys()))

    for label, (offset, scale) in conversions.items():
        if label in df.columns:
            df[label] = (df[label] - offset) * scale

    return df
    
    
    return df