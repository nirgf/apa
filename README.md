# 🛣️ APA – Advanced Pavement Analytics

## 📌 Project Overview

**APA (Advanced Pavement Analytics)** is a geospatial AI pipeline that uses **satellite imagery** to predict the **Pavement Condition Index (PCI)** of urban roads.

This project helps **municipalities**:
- Evaluate the condition of roads efficiently
- Generate AI-based insights for decision-makers
- Optimize maintenance strategies and urban planning

---

## 🧠 How It Works – Pipeline Overview

### 1. `data_runner.py` – Entry Point
Runs the full pipeline, loading settings from the config file and executing all steps end-to-end.

---

### 2. `config.yaml` – Configuration File
Defines:
- Input/output paths
- Data source (e.g., satellite type)
- ROI coordinates
- Filtering and caching settings

Automatically loaded by `data_runner.py`.

---

### 3. Data Import – `apa_utils.data_importer`
Handles imagery import based on the selected enum.

#### 📦 Supported Data Sources
Enums defined in `enum/`:
- `VENUS_ISRAEL_KIRYAT_ATA`
- `VENUS_DETROIT`
- `AIRBUS_DETROIT_MULTISPECTRAL`
- `AIRBUS_DETROIT_PANCHROMATIC`

More info in `data/` and `enum/` directories.

---

### 4. ROI Cropping
The region of interest (ROI) is read from the config file and used to crop the image to the relevant urban area.

---

### 5. Road Extraction – OpenStreetMap API
- Generates a road mask from geographic coordinates using OpenStreetMap.
- Produces a binary mask that highlights road pixels.
- Pixel-level details are not used yet—only GPS coordinates.

---

### 6. PCI Ground Truth – Segmentation
- Loads real PCI scores from **Detroit municipality CSV** files.
- Splits roads into segments using **Dijkstra’s algorithm** on the road mask.
- Each pixel within a segment receives the same PCI score.

> ⏳ This is a time-intensive step.

---

### 7. Caching Intermediate Results – `.npz`
- Intermediate data is saved as `.npz` files to avoid recomputation.
- File names include the data source enum.
- Caching happens automatically after heavy processing steps.

---

### 8. Image Filtering – `process_labeled_image()`
Post-processing step to clean and isolate relevant labeled data:
- Applies a **dilation kernel** to connect close pixels.
- Filters out:
  - Irrelevant colors
  - Disconnected or noisy segments
- All parameters are configurable via `config.yaml`.

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/apa.git
cd apa
