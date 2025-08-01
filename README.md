# Logistics Route Optimization

This project demonstrates route optimization algorithms (Ant Colony Optimization and Nearest Neighbor) for urban logistics using real geographical coordinates and travel time data from the OpenRouteService API. It supports animated map visualizations and performance analysis.


## Features

- Real route travel time matrix generated from OpenRouteService
- Ant Colony Optimization (ACO) with map animation and convergence plots
- Performance comparison plots (ACO: travel time vs. ants and iterations)
- Nearest Neighbor algorithm with dynamic map animation
- Visualizations on OpenStreetMap basemap
- CSV exports of optimal routes and travel time summaries
- Unified launcher (`main.py`) to run all components step-by-step

## Project Structure

<img width="495" height="332" alt="image" src="https://github.com/user-attachments/assets/6d10d81c-f5f8-46b4-8383-21c960b16582" />

## Tech Stack

- **Python 3.9+**
- **Libraries**: `networkx`, `matplotlib`, `contextily`, `pandas`, `numpy`, `openrouteservice`, `tqdm`
- **Map provider**: OpenStreetMap via `contextily`
- **API integration**: OpenRouteService

---

## Local Setup Instructions

### 1a. Unpack and Open the Project
```
Download and unzip the project archive. Open the extracted folder in your preferred IDE (e.g., Visual Studio Code).
```
#### OR

### 1b. Clone the repository
```
git clone https://github.com/d-kudin/Logistics_Route_Optimization.git
cd Logistics_Route_Optimization
```

### 2. Create and Activate a Virtual Environment
```
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux / macOS
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```

### 4. Configure Environment Variables
```
Write your API keys to the file .env instead of:
your_ORS_api_key

```
How to Get an OpenRouteService API Key

1. Visit: https://openrouteservice.org/dev/#/signup
2. Create a free account.
3. Go to Dashboard > API Keys and copy your personal key.
4. Copy the generated key and store it in your `.env` file
---
## Input Format

---

The file data/data.csv must include the following columns:

---
| ID | ADDRESS                 | X (lon) | Y (lat) |
|----|-------------------------|---------|---------|
| 0  | 123 Main St, CityName  | -95.12  | 29.76   |
| 1  | 456 Elm St, CityName   | -95.43  | 29.78   |
---

The generate_distance_matrix.py script will query the OpenRouteService Matrix API to generate output/distance_matrix.csv, containing the travel time matrix.

---

### Run Test first(All unit tests should pass)

```
pytest test_generate_distance_matrix.py
```

### How to Run algorithms
---

Run everything (distance matrix + both algorithms + benchmarking)
```
python main.py
```
---

Or run individual components:

---
Generate distance matrix from coordinates:
```
python generate_distance_matrix.py
```
---
Run Ant Colony Optimization with animated map:
```
python algorithms/ant_colony_with_map.py
```
---
Run Nearest Neighbor with animated map:
```
python algorithms/nearest_neighbor_with_map.py
```
---
Run ACO performance benchmark:
```
python algorithms/ant_colony_with_plot.py
```
---

All outputs are stored in output/

---

## Additional Notes:

---

If distance_matrix.csv has unusually large values (thousands of seconds), the system will automatically convert them to minutes.
All visualizations use OpenStreetMap tiles via contextily.
Real driving times ensure realistic logistics approximations.
You must be online for distance matrix generation via OpenRouteService.

---
