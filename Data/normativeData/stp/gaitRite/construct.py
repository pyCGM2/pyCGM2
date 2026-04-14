
import pytest
import numpy as np
import pyCGM2
import pandas as pd


filepath = pyCGM2.NORMATIVE_DATABASE_PATH+"stp\\normaldata.csv"
        
raw = pd.read_csv(filepath, sep=";", header=None, encoding="utf-8")

# les 2 premières lignes sont les headers
# ligne 0 : noms de colonnes (avec doublons min/max)
# ligne 1 : suffixes min/max

header0 = raw.iloc[0].fillna("").tolist()
header1 = raw.iloc[1].fillna("").tolist()

# reconstruction des noms de colonnes uniques
# avec propagation du label h0 quand il est vide
col_names = []
last_h0 = ""
for h0, h1 in zip(header0, header1):
    h0 = str(h0).strip()
    h1 = str(h1).strip()
    
    if h0:
        last_h0 = h0  # mémorise le dernier label non vide
    
    if h1 in ("min", "max"):
        col_names.append(f"{last_h0}_{h1}")
    elif h0:
        col_names.append(h0)
    else:
        col_names.append("_empty")

# données = à partir de la ligne 2
df = raw.iloc[2:].copy()
df.columns = col_names

# colonnes à convertir cm -> m
CM_TO_M_COLS = [
    col for col in df.columns
    if any(unit in col for unit in ["cm", "cm."])
]

for col in CM_TO_M_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce") / 100
    # renommer pour refléter l'unité
    df.rename(columns={col: col.replace("cm.", "m.").replace("cm", "m")}, inplace=True)

import re

def clean_units(col: str) -> str:
    col = re.sub(r'm\./', 'm/', col)      # m./  -> m/
    col = re.sub(r'm\.',  'm',  col)      # m.   -> m
    col = re.sub(r'sec\.','sec', col)     # sec. -> sec
    col = re.sub(r'/',    '.',   col)     # /    -> .
    col = re.sub(r'min\.','min', col)     # min. -> min
    return col

df.columns = [clean_units(col) for col in df.columns]

df.to_excel(pyCGM2.NORMATIVE_DATABASE_PATH+"stp\\pyCGM2_normaldata.xlsx", index=False)

import ipdb; ipdb.set_trace()