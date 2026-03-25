# 🌲 Tree Carbon ML — U.S. Aboveground Tree Carbon Response to Nitrogen Deposition

> Machine learning pipeline for analyzing and predicting aboveground tree carbon responses to nitrogen deposition across U.S. FIA forest plots.

**Source dataset:** [EPA/data.gov — dC/dN paper dataset](https://catalog.data.gov/dataset/data-from-dc-dn-paper-wide-variation-in-u-s-aboveground-tree-carbon-responses-to-nitrogen-)  
**DOI:** https://doi.org/10.23719/1528045

---

## 📁 Project Structure

```
tree_carbon_ml/
├── data/
│   ├── raw/                    # Original downloaded CSVs (never modified)
│   ├── interim/                # Partially processed data
│   └── processed/              # Final cleaned datasets for modeling
│
├── notebooks/
│   ├── 01_data_ingestion.ipynb         # Download & validate raw data
│   ├── 02_eda_exploration.ipynb        # Exploratory Data Analysis
│   ├── 03_preprocessing.ipynb          # Cleaning, imputation, encoding
│   ├── 04_feature_extraction.ipynb     # Feature engineering & selection
│   └── 05_model_ready_dataset.ipynb    # Final dataset prep + sanity checks
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py         # Data download utilities
│   │   └── validate.py         # Schema validation helpers
│   ├── features/
│   │   ├── __init__.py
│   │   ├── engineer.py         # Feature engineering functions
│   │   └── selector.py         # Feature selection utilities
│   ├── models/                 # (for future model notebooks)
│   │   └── __init__.py
│   └── visualization/
│       ├── __init__.py
│       └── plots.py            # Reusable plotting functions
│
├── outputs/
│   ├── figures/                # Saved EDA and feature plots
│   └── reports/                # HTML/PDF reports
│
├── tests/
│   └── test_preprocessing.py   # Unit tests for src modules
│
├── requirements.txt            # Python dependencies
├── setup.py                    # Package install (for Google Colab)
└── README.md
```

---

## 🚀 Quick Start (Google Colab)

Open any notebook in `notebooks/` and run the first cell — it will auto-install dependencies and clone/mount everything needed.

```python
# Paste this into a Colab cell to get started
!git clone https://github.com/YOUR_USERNAME/tree_carbon_ml.git
%cd tree_carbon_ml
!pip install -r requirements.txt -q
```

Then open notebooks in order: `01 → 02 → 03 → 04 → 05`.

---

## 📊 Dataset Overview

| File | Description |
|------|-------------|
| `SN_gs_dCdN01_...csv` | Plot-level dC/dN values for all FIA plots |
| `SN_gs_dCdN01_..._column key.csv` | Column definitions and metadata |

### Key Variables

| Column | Description |
|--------|-------------|
| `PLT_CN` | FIA Plot ID |
| `TPH.gs.dC.dN0.01` | **Target**: Net dC/dN (growth + survival) × trees/ha |
| `LAT`, `LON` | Plot coordinates |
| `STATECD`, `COUNTYCD` | Geographic identifiers |
| `US_L4CODE`, `NA_L3CODE`, `NA_L1CODE` | Ecoregion codes |
| `EXPN.ha` | FIA expansion factor (hectares) |

---

## 🔬 Pipeline Overview

```
Raw Data → Ingestion → EDA → Preprocessing → Feature Engineering → Model-Ready Dataset → ML Models
  01           02         03         04                05
```

---

## 📦 Requirements

- Python ≥ 3.8
- pandas, numpy, scipy
- scikit-learn
- matplotlib, seaborn, plotly
- geopandas (optional, for spatial plots)
- jupyter

---

## 📄 Citation

Clark, C. et al. "Wide variation in U.S. aboveground tree carbon responses to nitrogen deposition and a weakening response since the 1980s-90s." U.S. EPA Office of Research and Development. https://doi.org/10.23719/1528045
