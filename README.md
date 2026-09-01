# Predicting Public Transport Delays Using Weather & Events

End-to-end regression pipeline predicting arrival delay (in minutes) for public transit trips using weather conditions, city events, and traffic data — with all features restricted to information available **before departure**.

## Dataset

2,000 trips across multiple transport types (bus, metro, tram, rail), routes, and seasons. Features include temperature, precipitation, humidity, wind speed, traffic congestion index, city event type, and scheduled duration.

## Pipeline

| Step | What |
|------|------|
| Cleaning | Parse datetimes, handle overnight trips, fill missing event types |
| Leakage guard | Drop `actual_departure_delay_min` — available post-departure, not pre-departure |
| Feature engineering | `has_event`, `severe_weather`, `temp_extreme`, `humid_precip` interaction |
| Encoding | Label-encode categoricals (route, station, weather condition, season) |
| Models | Linear Regression, Random Forest (200 trees), XGBoost (300 rounds) |

## Results

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | 7.78 min | 9.30 min | -0.025 |
| Random Forest | 7.83 min | 9.35 min | -0.038 |
| XGBoost | 7.96 min | 9.46 min | -0.062 |

All three models perform near mean-predictor level (R² ≈ 0), which is itself a finding: weather, events, and traffic features as recorded here do not explain delay variance well. The top Random Forest features — `temperature_C`, `traffic_congestion_index`, `destination_station`, `wind_speed_kmh` — have low individual importances (< 0.10), consistent with a high-noise target.

## Outputs

| File | Description |
|------|-------------|
| `01_eda.png` | 9-panel EDA: delay by weather, transport type, event, season, peak hour, precipitation |
| `02_model_results.png` | Actual vs predicted, MAE/RMSE comparison, top 15 feature importances |
| `03_residuals.png` | Residual distribution and R² comparison across models |

## Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
python transport_model.py
```
