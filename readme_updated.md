# National-Scale Traffic Demand Modeling with Mobile Phone Data and DTALite

This repository contains a multi-stage pipeline to construct, simulate, and validate large-scale dynamic traffic demand using mobile phone data, DTALite simulation, and FHWA AADT ground truth for the top 100 U.S. metro areas.

**Main Highlights:**
- Provides **public OD datasets and simulation-ready networks** for the **top 100 CBSAs** in the United States.
- OD data is extracted from mobile phone traces and **calibrated using link-level traffic volumes (AADT)**.
- All files will be public and are directly compatible with **Dynamic traffic assignment**.

---

## 📁 Pipeline Structure

### Step 0: OD Extraction from Mobile Data

- `0-OD_from_Individual.py`: Processes Veraset mobile phone data to generate daily OD flows at the Census Block Group (CBG) level. Includes home location detection, trip chaining, and hourly trip estimation.

### Step 1: DTALite Simulation Preparation and Execution

- `1-main_DTALite_230915.py`: For each of the top 100 CBSAs:
  - Downloads road networks (OSM) and converts them to DTALite format using `osm2gmns`.
  - Matches nodes to CBG zones and assigns OD flows.
  - Merges in AADT data from HPMS for ground truth.
  - Prepares input files for DTALite simulation (nodes, links, demand, sensor data).
  - Runs DTALite to simulate 7–8am peak demand with both raw and weighted OD matrices.

### Step 2: Results Analysis and Visualization

- `2-results_analysis.py`: For each CBSA:
  - Compares raw, weighted, and ODME-adjusted OD flows.
  - Visualizes road network speed, OD flow maps, and link-level volumes.
  - Compares simulated volumes with AADT data.
  - Correlates changes in OD with population density and other socio-demographic attributes.

---

## 🗃️ Inputs and External Data

- **Mobile OD data**: Veraset Visits data (Snappy Parquet format).
- **Geospatial shapefiles**: Census CBGs, tracts, CBSAs, and places.
- **Socio-demographics**: NHGIS, SmartLocation Database.
- **Road networks**: Downloaded from OpenStreetMap via `osmnx`.
- **Simulation tool**: [DTALite](https://github.com/DrKeHan/DTALite).
- **AADT**: HPMS 2024 shapefiles for ground-truth traffic volumes.

---

## 🧩 Key Features

- Real OD extraction from individual mobile device traces.
- Automatic imputation of missing home-origin and home-destination trips.
- Hourly and monthly OD demand generation.
- National-scale simulation for 100 U.S. metro areas.
- ODME calibration using AADT sensors.
- High-resolution visualization of traffic volumes and OD comparisons.
- Publicly available calibrated OD and simulation network files for use in DTA.

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/traffic-demand-dtalite.git
cd traffic-demand-dtalite
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

You also need:
- `osmnx`, `osm2gmns`, `geopandas`, `contextily`, `DTALite`
- `matplotlib`, `seaborn`, `tqdm`, `mapclassify`

### 3. Run scripts step-by-step

- Extract OD:  
  ```bash
  python 0-OD_from_Individual.py
  ```

- Run DTALite pipeline (each CBSA):
  ```bash
  python 1-main_DTALite_230915.py
  ```

- Analyze results:
  ```bash
  python 2-results_analysis.py
  ```

---

## 📂 Output Files

Each metro area will generate:

- `Raw_OD/`, `Weighted_OD/`, `ODME/`, `Final/`: Simulation results
- `link.csv`, `zone_id.csv`: Network and zone assignments
- `link_performance_*.csv`: Simulated link-level volumes
- `valid_linkss.csv`: AADT-matched links
- `*.pdf`, `*.png`: Maps of volume, OD flow, and error comparisons
- `link_flows.pkl`: Combined simulation + ground truth for evaluation


---

## 🙏 Acknowledgments

- Veraset and SafeGraph for mobile data
- FHWA for traffic datasets
- DTALite and osm2gmns developers
- OpenStreetMap for road network

