# Eye-Tracking Biometrics with GazeBase

This project investigates whether motion- and saccade-specific feature engineering can improve cross-session eye-movement biometric identification.

## Research Question

Can motion- and saccade-specific feature engineering improve cross-session eye-movement biometric identification compared with a classical feature representation adapted from Harezlak et al.?

## Final Project Summary

The project follows four main steps:

1. **Replication** — adapt the classical Harezlak et al. eye-movement biometric approach to GazeBase.
2. **Exploration / problem discovery** — identify a strong HSS + RAN protocol and measure the loss in accuracy when the recording session changes.
3. **Proposed method** — combine Yin-inspired full-response motion features with Rigas-inspired principal-saccade dynamics.
4. **External validation** — freeze the method after Round-1 development and evaluate it on held-out Round 2.

The final task is **closed-set biometric identification** among 50 participants.

## Key Results

### Replication

The three-point Random Forest replication achieved:

- **72.19% ± 0.93%** vector-level accuracy
- **96.67% ± 1.76%** subject-level accuracy

### Exploration

Using the classical HSS + RAN system:

- development accuracy: **95.8% ± 0.63%**
- separate-session validation accuracy: **76.2% ± 1.14%**

This drop motivated the final cross-session robustness study.

### Round-1 Development Comparison

| Representation | R1 S1→S2 | R1 S2→S1 | Bidirectional Mean |
|---|---:|---:|---:|
| Classical baseline, 230 features | 86.8% | 82.4% | **84.6%** |
| Yin-inspired motion, 123 features | 96.4% | 91.4% | **93.9%** |
| Final proposed representation, 144 features | 99.8% | 94.0% | **96.9%** |

### Round-2 External Validation

All Round-1 S1 + S2 data are used for fitting. Round 2 is then evaluated without post-validation tuning.

| Representation | R2-S1 | R2-S2 | Two-Session Mean |
|---|---:|---:|---:|
| Classical baseline, 230 features | 44.2% | 49.0% | **46.6%** |
| Yin-inspired motion, 123 features | 60.8% | 58.8% | **59.8%** |
| Final proposed representation, 144 features | 73.2% | 82.4% | **77.8%** |

The final proposed representation improves the Round-2 mean by **31.2 percentage points** over the same-setup classical baseline and by **18.0 percentage points** over the Yin-only representation.

## Proposed Method

The final representation contains **144 features**:

- **123 Yin-inspired motion-information features**
- **19 Rigas-inspired principal-saccade dynamic features**
- **1 training-derived saccadic-vigor feature**
- **1 missing-saccade indicator**

The motion representation captures displacement magnitude, direction, temporal structure, turning behavior, path length, and speed across the full one-second response.

The saccade representation focuses on the largest valid saccade in the response and describes properties including onset, duration, amplitude, path length, peak speed, acceleration, deceleration, efficiency, trajectory shape, and saccadic vigor.

A Random Forest classifier is retained so that the comparison primarily evaluates the feature representation rather than a substantially different model architecture.

## Final Evaluation Protocol

- 50 fixed participants
- HSS and RAN tasks
- 1-second response windows
- Random Forest with 500 trees
- `class_weight="balanced_subsample"`
- seeds 0 through 9
- HSS and RAN modeled separately
- subject evidence = mean predicted class probabilities across windows
- HSS/RAN evidence fusion = 50/50

Round 1 is used for session-separated development in both directions:

- R1-S1 → R1-S2
- R1-S2 → R1-S1

After development is frozen, all Round-1 S1 + S2 data are used for fitting and Round-2 S1/S2 are used for external validation.

## Repository Structure

```text
.
├── replication/
│   ├── README.md
│   ├── src/
│   └── data/          # created after extracting reproduction_data.zip
├── exploration/
│   ├── README.md
│   ├── results/
│   ├── src/
│   └── data/          # created after extracting reproduction_data.zip
├── proposed_method/
│   ├── README.md
│   ├── src/
│   └── data/          # created after extracting reproduction_data.zip
├── data/
│   └── gazebase_manifest.csv
├── requirements.txt
└── README.md
```

## Reproducing the Results on a New Machine

### 1. Download the repository

Use GitHub's **Code → Download ZIP** option or clone the repository.

### 2. Extracting the reproduction data

The repository does not initially contain `data/` folders inside `replication/`, `exploration/`, or `proposed_method/`. These folders and their processed CSV files are created when the reproduction-data ZIP is extracted.

Download `reproduction_data.zip` from the `v1.0-reproduction` GitHub Release.

The ZIP should be extracted directly into the root of the downloaded repository, which is the same folder that contains `README.md`, `requirements.txt`, `replication/`, `exploration/`, and `proposed_method/`.

#### Windows

Place `reproduction_data.zip` in the repository root and run:

```powershell
python -c "import zipfile; zipfile.ZipFile('reproduction_data.zip').extractall('.')"
```

#### macOS / Linux

Place `reproduction_data.zip` in the repository root and run:

```bash
python3 -c "import zipfile; zipfile.ZipFile('reproduction_data.zip').extractall('.')"
```

After extraction, the following folders should exist:

```text
replication/data/
exploration/data/
proposed_method/data/
```

For example, the following files should now be present:

```text
replication/data/replication_features_100point.csv

exploration/data/development.csv
exploration/data/test.csv
exploration/data/validation.csv

proposed_method/data/cohort_ids.csv
proposed_method/data/r1_motion_features.csv
proposed_method/data/r1_s1_classical_features.csv
proposed_method/data/r1_s2_classical_features.csv
proposed_method/data/r1_saccade_features.csv
proposed_method/data/r2_motion_features.csv
proposed_method/data/r2_s1_classical_features.csv
proposed_method/data/r2_s2_classical_features.csv
proposed_method/data/r2_saccade_features.csv
```

These processed datasets are sufficient to reproduce the reported model-evaluation results. The full raw GazeBase dataset is not required unless the user wants to rebuild the feature datasets from the original eye-tracking recordings.

### 3. Create a clean Python environment

The final clean-machine test was performed with **Python 3.13.7**.

#### Windows PowerShell

From the repository root:

```powershell
python -m venv .venv
```

No PowerShell activation is required. Install dependencies directly through the environment's Python executable:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

#### macOS / Linux

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

## Primary Reproduction Command

The main final-project experiment is the Round-2 external comparison.

### Windows

```powershell
.\.venv\Scripts\python.exe proposed_method\src\evaluate_comparison.py
```

### macOS / Linux

```bash
./.venv/bin/python proposed_method/src/evaluate_comparison.py
```

This command compares:

- `BASELINE_FULL230`
- `YIN_ONLY123`
- `FINAL_YIN_SACCADE144`

Expected primary output:

```text
BASELINE_FULL230       R2-S1 44.2%   R2-S2 49.0%   mean 46.6%
YIN_ONLY123            R2-S1 60.8%   R2-S2 58.8%   mean 59.8%
FINAL_YIN_SACCADE144   R2-S1 73.2%   R2-S2 82.4%   mean 77.8%
```

The script also writes:

```text
proposed_method/results/external_validation_results.csv
proposed_method/results/external_validation_diagnostics.csv
```

## Optional: Reproduce Earlier Project Phases

These commands are not required to run the final comparison, but they reproduce the earlier experiments documented in the project.

### Replication

```powershell
.\.venv\Scripts\python.exe replication\src\evaluate_replication.py
```

Expected Random Forest results:

- vector-level: **72.19% ± 0.93%**
- subject-level: **96.67% ± 1.76%**

### Exploration Development

```powershell
.\.venv\Scripts\python.exe exploration\src\evaluate_development.py
```

Expected HSS + RAN subject-level result:

- **95.8% ± 0.63%**

### Exploration Cross-Session Validation

```powershell
.\.venv\Scripts\python.exe exploration\src\evaluate_validation.py
```

Expected HSS + RAN result:

- **76.2% ± 1.14%**

### Round-1 Final-Method Development Comparison

```powershell
.\.venv\Scripts\python.exe proposed_method\src\evaluate_development_comparison.py
```

Expected bidirectional means:

- classical baseline: **84.6%**
- Yin-only: **93.9%**
- final 144-feature representation: **96.9%**

## Raw GazeBase Data

This project uses the GazeBase dataset introduced by Griffith et al. (2021).

The processed reproduction package above is sufficient to reproduce the reported model-evaluation results.

The original raw GazeBase dataset is **not included** in this repository. Raw recordings are only needed when rebuilding feature datasets from the original eye-tracking recordings. The code assumes raw recordings are placed under:

```text
data/raw/
```

## Primary References

**Harezlak, K., Blasiak, P., & Kasprowski, P. (2021).**  
*Biometric Identification Based on Eye Movement Dynamic Features.*

**Griffith, H. et al. (2021).**  
*GazeBase, a large-scale, multi-stimulus, longitudinal eye movement dataset.*

**Yin, J., Sun, J., Li, J., & Liu, K. (2022).**  
*An Effective Gaze-Based Authentication Method with the Spatiotemporal Feature of Eye Movement.*

**Rigas, I., Komogortsev, O. V., & Shadmehr, R. (2016).**  
*Biometric Recognition via Eye Movements: Saccadic Vigor and Acceleration Cues.*

## Notes

The experimental task in this project is biometric **identification**, not verification/authentication. The system selects one identity from the enrolled participant set.

The project emphasizes session-separated evaluation, held-out temporal validation, reproducibility, and CPU-friendly classical machine learning.
