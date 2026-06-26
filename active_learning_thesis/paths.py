from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
PREDICTIVE_CODE_DIR = REPO_ROOT / "SA_ML_predictive" / "code"
PREDICTIVE_DATA_DIR = REPO_ROOT / "SA_ML_predictive" / "data"
PREDICTIVE_MODEL_DIR = REPO_ROOT / "SA_ML_predictive" / "models"
GENERATIVE_DIR = REPO_ROOT / "SA_ML_generative"
MD_DIR = REPO_ROOT / "MD"
MD_BURA_TEMPLATE_DIR = MD_DIR / "CG_sims_BURA" / "CG_sims_BURA"
DATASET_PATH = PREDICTIVE_DATA_DIR / "data_SA.csv"
