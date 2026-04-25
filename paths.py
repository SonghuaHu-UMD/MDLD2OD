from pathlib import Path

# Root locations — change these when moving disks
RESEARCH_ROOT = Path(r"F:\Research\MDLD_OD")
VERASET_DIR = Path(r"F:\Data\Dewey\Veraset")
VISITS_DIR = VERASET_DIR / "visits"
HOME_VISITS_DIR = VERASET_DIR / "home-visits"
WORK_VISITS_DIR = VERASET_DIR / "work-visits"

# Project subdirectories
MDLDOD_DIR = RESEARCH_ROOT / "MDLDod"
SHP_DIR = MDLDOD_DIR / "shp"
OD_DIR = MDLDOD_DIR / "od"
RAW_DATA_DIR = MDLDOD_DIR / "raw_data"
SIMULATION_DIR = MDLDOD_DIR / "simulation"
RESULTS_DIR = MDLDOD_DIR / "results"
RESULTS_ALL_DIR = MDLDOD_DIR / "results_all"

# External datasets and binaries
VOLUME_DIR = RESEARCH_ROOT / "Volume" / "HPMS_2020"
DTALITE_EXE = RESEARCH_ROOT / "DTALite_230915.exe"
