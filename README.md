# pyCGM2

| Type                   | Status |
|------------------------|--------|
| License                | [![License: CC BY-SA 4.0](https://licensebuttons.net/l/by-sa/4.0/80x15.png)](https://creativecommons.org/licenses/by-sa/4.0/) |
| Continuous integration | [![Build status-python3.9](https://github.com/pyCGM2/pyCGM2/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/pyCGM2/pyCGM2/actions/) |

`pyCGM2` is an open-source Python library for gait analysis based on the **Conventional Gait Model (CGM)**.  
It is designed for both research and clinical use, with a particular focus on compatibility with **Vicon Nexus**, **OpenSim**, and **IMU-based** workflows.

---

## 🔗 Project Links

- 🌐 Main website: [pycgm2.netlify.app](https://pycgm2.netlify.app)
- 📚 API Documentation: [pyCGM2 GitHub Pages](https://pycgm2.github.io/pyCGM2/)
- 📥 Installation guide: [Installation instructions](https://pycgm2.netlify.app/installation/)

---

## 📦 Installation

Clone or download the repository and open an **Anaconda Prompt**.  
Navigate to the local folder where `pyCGM2` was extracted:

```bash
cd path/to/my/pycgm2/folder
```

Create the conda environment and install the package:

```bash
conda env create -f environment.yml
conda activate pycgm310
pip install -e .
```

---

## ⚙️ Integration with Vicon Nexus

If you are using **Vicon Nexus** and want to run `pyCGM2` directly from Nexus pipelines, run the following two commands:

```bash
pyCGM2-setup_NexusPackages.exe
pyCGM2-generate_pyCGM2_Nexus.exe
```

- The first command installs the Vicon-distributed Python packages into Nexus’s internal SDK folder.
- The second command creates a ready-to-use **CGM2.3 pipeline** that can be imported into Nexus.  
  It is configured to activate your `pyCGM2` environment using a script located in `C:/Users/YourName/AppData/pycgm2`.

---

## 📣 Contributions & Issues

We welcome contributions, bug reports, and feature suggestions!  
Please use the [GitHub Issue Tracker](https://github.com/pyCGM2/pyCGM2/issues) for any feedback or questions.

---

## 📜 License

This project is licensed under the [Creative Commons BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.

## 📜 Release
[See full release notes](./release.md)
---
