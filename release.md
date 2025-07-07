# pyCGM2 — Release Note  
**Version:** 4.4 (Release Candidate 1)  
**Date:** July 2025  

---

## Overview

This release introduces major improvements to the pipeline structure, setup process, and analysis modules, while preserving all features from version 4.3-rc3. The release enhances performance, documentation, and adds new features related to IMU processing, event detection, and musculoskeletal modeling.

---

## 🔧 NEW FEATURES

- **Setup Script Overhaul**  
  `setup.py` was fully rewritten for better environment handling and package deployment.

- **Extended Python Compatibility**  
  pyCGM2 now supports Python versions from **3.9 to 3.11**.


- **Kalman Filter Kinematic Fitting for CGM2.2**  
  Kalman-based fitting is now integrated for CGM2.2 pipelines.

- **Musculoskeletal Modeling Support for CGM2.2**  
  Enables static scaling and inverse kinematics using OpenSim models.

- **intellevent Integration**  
  integration of intelevent ( deep learning based algorithm) to identify gait events.

- **Opensense Integration**  
  New support for Opensense IMU Placer and IMU Kinematics Fitter.

- **IMU Module Rewritten**  
  Refactored for clarity, flexibility, and robustness.


- **Gloersen et al. Gap Filling Method**  
  Now available as a CLI command from `pycgm2.exe`.


- **New Tutorial Files and Test Suites**  
  Added several educational examples and robustness checks to guide usage.

---

## ✅ IMPROVEMENTS

- **Ground Reaction Force Plotting Tools**  
  New CLI tools for generating temporal, time-normalized, and comparison plots of GRFs.

- **Joint Moment Comparison Tool**  
  Added functionality to compare joint moments between trials (CLI callable).

- **Frame-specific Calibration**  
  All CGMi calibration functions now accept a `frames` keyword, enabling model calibration over a selected frame range.

- **Nexus Parser Updated**  
  Now supports both `Start` and `End` general events and integrates rigid gap filling strategies.

- **Optimized CGM Pipeline**  
  Internal refactoring of `cgm.py` and `cgm2.py` for improved maintainability.

- **Visualization Tools**  
  New functions to visualize coordinate systems and body segments.

- **Y-axis Auto-scaling**  
  Graphs now adjust limits dynamically if values exceed defaults.

- **Simplified Scheme and Filtering Options**  
  Enhanced flexibility for defining and processing custom analysis flows.

---

## Work In progress

- **New "mek" module**  
  Introduces a simplified interface for data structuring and flow control using customizable templates.

- **New "flow" module** 
   Introduces  flow control using customizable templates. 

## 🐞 FIXES

- Fixed crash when launching `pycgm2.exe` without Vicon Nexus installed.
- CGM2.4 now tolerates missing markers without interrupting processing.
- Fixed misdetection of progression axis when starting from seated postures (removed reliance on heel markers).
- `smartGetEvents` no longer returns duplicate events.
- Corrected handling of lateral shank + KAD configuration.
- Fixed OpenSim segment referential definitions in CGM2.2 and CGM2.3.
- Fixed bug in `comAcceleration` for short gait cycles (< 15 frames).
- Added missing `parser.parse_args()` to Kalman and Gloersen scripts for CLI use.
- Improved force plate filtering for duplicated analog channels (Type 3 plates).

---

## 🛠 DEV & PACKAGE STRUCTURE

- Meta.yaml updated with required Conda channels (`conda-forge`, `opensim-org`).
- Pipeline file structure reorganized and renamed.
- Onnxruntime support added.
- New utilities for upsampling, marker management, and local/global transformations.
- Documentation rebuilt using **PyData Sphinx Theme** — full API coverage.

---

## 💬 Feedback and Issues

Please report bugs or suggest improvements via the [GitHub issue tracker](https://github.com/pyCGM2/pyCGM2/issues).  
You can also contact: **fabien.leboeuf@gmail.com**

---
