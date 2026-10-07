# California Housing Price Prediction

A machine learning regression project that predicts median house values in California districts using the classic **California Housing** dataset and **scikit-learn**.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

---

## Overview

This project builds an end-to-end regression workflow that:

1. Loads and explores the California Housing dataset
2. Visualizes house prices geographically
3. Preprocesses data (train/test split and feature scaling)
4. Trains a **Linear Regression** model
5. Evaluates performance with MSE and R²
6. Packages preprocessing and modeling into a scikit-learn **Pipeline**
7. Exposes a simple function that estimates a house price from 8 input features

## Dataset

The data comes from `sklearn.datasets.fetch_california_housing` (based on the 1990 U.S. Census): **20,640 samples** and **8 features**.

| Feature | Description |
|---|---|
| `MedInc` | Median income in the block group |
| `HouseAge` | Median house age in the block group |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average number of household members |
| `Latitude` | Block group latitude |
| `Longitude` | Block group longitude |

**Target:** `MedHouseVal`, the median house value in units of **$100,000**.

## Project Workflow

1. **Data loading:** fetch the dataset into a pandas DataFrame.
2. **Exploratory analysis:** scatter plot of longitude vs. latitude, colored by median house value.
3. **Preprocessing:** 80/20 train-test split (`random_state=42`) and `StandardScaler` normalization.
4. **Modeling:** `LinearRegression` trained on the scaled features.
5. **Evaluation:** Mean Squared Error, R² score, and an Actual vs. Predicted plot.
6. **Pipeline:** `StandardScaler` + `LinearRegression` combined so raw inputs can be passed straight to the model.
7. **Prediction function:** `predict_house_price(...)` returns an estimated price in dollars.

## Results

| Metric | Value |
|---|---|
| Mean Squared Error (MSE) | 0.56 |
| R² Score | 0.576 |

The linear model explains roughly 58% of the variance in house prices. It provides a solid baseline, and more complex models should improve on it.

### Example Prediction

```python
predict_house_price(
    MedInc=8.3, HouseAge=41.0, AveRooms=6.0, AveBedrms=1.0,
    Population=980, AveOccup=2.5, Latitude=37.5, Longitude=-122.3
)
# 'Estimated House Price: $443,209.67'
```

## Installation

**Prerequisites:** Python 3.9+ and Jupyter.

```bash
# Clone the repository
git clone https://github.com/<your-username>/california-housing-prediction.git
cd california-housing-prediction

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install numpy pandas matplotlib scikit-learn jupyter
```

## Usage

1. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
2. Open `California_Housing_Price_Prediction.ipynb`.
3. Run all cells (**Kernel → Restart & Run All**).

The dataset is downloaded automatically by scikit-learn on first run, so an internet connection is required once.

## Project Structure

```
.
├── California_Housing_Price_Prediction.ipynb   # Main notebook
└── README.md
```

