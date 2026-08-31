import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# 1 — load & inspect data
print("STEP 1: LOADING DATA")
df = pd.read_csv('public_transport_delays.csv')
print(f"Rows: {len(df):,}  |  Columns: {df.shape[1]}")
print(df.dtypes)

# 2 — data cleaning
print("STEP 2: DATA CLEANING")
print()
# Fill missing event_type with 'None'
df['event_type'] = df['event_type'].fillna('None')
print("Missing values after fill:", df.isnull().sum().sum())

# Parse datetime columns
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
df['date'] = pd.to_datetime(df['date'])

# Scheduled duration (how long the trip is supposed to take)
df['scheduled_dep'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['scheduled_departure'])
df['scheduled_arr'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['scheduled_arrival'])
df['scheduled_duration_min'] = (df['scheduled_arr'] - df['scheduled_dep']).dt.total_seconds() / 60
# Handle overnight trips
df.loc[df['scheduled_duration_min'] < 0, 'scheduled_duration_min'] += 1440

# Drop raw columns we've already processed or don't need
df.drop(columns=['trip_id','scheduled_departure','scheduled_arrival',
                  'scheduled_dep','scheduled_arr','date','time','datetime',
                  'actual_departure_delay_min'],  # avoid leakage from departure delay
        inplace=True)

print("Columns after cleaning:", df.columns.tolist())

# 3 — feature engineering
print("STEP 3: FEATURE ENGINEERING")

# Binary: is there a city event happening?
df['has_event'] = (df['event_type'] != 'None').astype(int)

# Severe weather flag   
severe_weather = ['Storm', 'Snow', 'Fog']
df['severe_weather'] = df['weather_condition'].isin(severe_weather).astype(int)

# Temperature extremes (below 5°C or above 35°C)
df['temp_extreme'] = ((df['temperature_C'] < 5) | (df['temperature_C'] > 35)).astype(int)

# Humidity × precipitation interaction
df['humid_precip'] = df['humidity_percent'] * df['precipitation_mm']

# Label-encode all remaining categoricals
cat_cols = ['transport_type', 'route_id', 'origin_station',
            'destination_station', 'weather_condition', 'event_type', 'season']
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

print("New engineered features: has_event, severe_weather, temp_extreme, humid_precip")
print("Categorical columns encoded:", cat_cols)
print(f"Final feature count: {df.shape[1] - 2}")  # minus target & delayed flag

# 4 — exploratory data analysis
print("STEP 4: EXPLORATORY ANALYSIS (saved to plots)")

df_raw = pd.read_csv('public_transport_delays.csv')
df_raw['event_type'] = df_raw['event_type'].fillna('None')

fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('#0f1117')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

COLORS = ['#4FC3F7','#81C784','#FFB74D','#F06292','#CE93D8','#80DEEA']
TARGET = 'actual_arrival_delay_min'

def style_ax(ax, title):
    ax.set_facecolor('#1a1d27')
    ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=10)
    ax.tick_params(colors='#aaaaaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333344')
    ax.xaxis.label.set_color('#aaaaaa')
    ax.yaxis.label.set_color('#aaaaaa')

# Delay by Weather
ax1 = fig.add_subplot(gs[0, 0])
order = df_raw.groupby('weather_condition')[TARGET].mean().sort_values(ascending=False).index
means = df_raw.groupby('weather_condition')[TARGET].mean().loc[order]
bars = ax1.bar(order, means, color=COLORS[:len(order)])
ax1.set_ylabel('Avg Delay (min)')
style_ax(ax1, '☁ Avg Delay by Weather Condition')
ax1.set_xticklabels(order, rotation=30, ha='right', fontsize=9, color='#aaaaaa')
for bar, val in zip(bars, means):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
             f'{val:.1f}', ha='center', va='bottom', color='white', fontsize=8)

# --- Plot 2: Delay by Transport Type ---
ax2 = fig.add_subplot(gs[0, 1])
order2 = df_raw.groupby('transport_type')[TARGET].mean().sort_values(ascending=False).index
means2 = df_raw.groupby('transport_type')[TARGET].mean().loc[order2]
bars2 = ax2.bar(order2, means2, color=COLORS[:len(order2)])
style_ax(ax2, '🚌 Avg Delay by Transport Type')
ax2.set_ylabel('Avg Delay (min)')
for bar, val in zip(bars2, means2):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
             f'{val:.1f}', ha='center', va='bottom', color='white', fontsize=8)

# --- Plot 3: Delay by Event Type ---
ax3 = fig.add_subplot(gs[0, 2])
order3 = df_raw.groupby('event_type')[TARGET].mean().sort_values(ascending=False).index
means3 = df_raw.groupby('event_type')[TARGET].mean().loc[order3]
bars3 = ax3.bar(order3, means3, color=COLORS[:len(order3)])
style_ax(ax3, '🎪 Avg Delay by City Event')
ax3.set_ylabel('Avg Delay (min)')
ax3.set_xticklabels(order3, rotation=30, ha='right', fontsize=9, color='#aaaaaa')
for bar, val in zip(bars3, means3):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
             f'{val:.1f}', ha='center', va='bottom', color='white', fontsize=8)

# --- Plot 4: Temp vs Delay scatter ---
ax4 = fig.add_subplot(gs[1, 0])
sc = ax4.scatter(df_raw['temperature_C'], df_raw[TARGET],
                 alpha=0.3, s=10, c=df_raw[TARGET], cmap='plasma')
ax4.set_xlabel('Temperature (°C)')
ax4.set_ylabel('Delay (min)')
style_ax(ax4, '🌡 Temperature vs Delay')
plt.colorbar(sc, ax=ax4).ax.yaxis.set_tick_params(color='white')

# --- Plot 5: Traffic congestion vs delay ---
ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(df_raw['traffic_congestion_index'], df_raw[TARGET],
            alpha=0.3, s=10, color='#4FC3F7')
ax5.set_xlabel('Traffic Congestion Index')
ax5.set_ylabel('Delay (min)')
style_ax(ax5, '🚦 Traffic Congestion vs Delay')

# --- Plot 6: Delay distribution ---
ax6 = fig.add_subplot(gs[1, 2])
ax6.hist(df_raw[TARGET], bins=30, color='#81C784', edgecolor='#0f1117')
ax6.axvline(df_raw[TARGET].mean(), color='#FFB74D', linestyle='--', linewidth=2, label=f"Mean: {df_raw[TARGET].mean():.1f}")
ax6.set_xlabel('Arrival Delay (min)')
ax6.set_ylabel('Count')
ax6.legend(facecolor='#1a1d27', labelcolor='white', fontsize=9)
style_ax(ax6, '📊 Delay Distribution')

# --- Plot 7: Season vs delay ---
ax7 = fig.add_subplot(gs[2, 0])
order4 = df_raw.groupby('season')[TARGET].mean().sort_values(ascending=False).index
bp = df_raw.groupby('season')[TARGET].apply(list)
bdata = [bp[s] for s in order4]
bp_plot = ax7.boxplot(bdata, labels=order4, patch_artist=True, medianprops=dict(color='white', linewidth=2))
for patch, color in zip(bp_plot['boxes'], COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax7.set_ylabel('Delay (min)')
ax7.tick_params(axis='x', colors='#aaaaaa')
style_ax(ax7, '🍂 Delay Distribution by Season')

# --- Plot 8: Peak hour vs non-peak ---
ax8 = fig.add_subplot(gs[2, 1])
peak_means = df_raw.groupby('peak_hour')[TARGET].mean()
ax8.bar(['Off-Peak', 'Peak Hour'], peak_means.values, color=['#4FC3F7','#F06292'])
for i, val in enumerate(peak_means.values):
    ax8.text(i, val+0.2, f'{val:.1f}', ha='center', color='white', fontsize=10)
ax8.set_ylabel('Avg Delay (min)')
style_ax(ax8, '⏰ Peak vs Off-Peak Delays')

# --- Plot 9: Precipitation vs delay ---
ax9 = fig.add_subplot(gs[2, 2])
prec_bins = pd.cut(df_raw['precipitation_mm'], bins=[0,5,10,15,20,50], labels=['0-5','5-10','10-15','15-20','20+'])
df_raw['prec_bin'] = prec_bins
means9 = df_raw.groupby('prec_bin', observed=True)[TARGET].mean()
ax9.bar(means9.index.astype(str), means9.values, color=COLORS[:len(means9)])
ax9.set_xlabel('Precipitation (mm)')
ax9.set_ylabel('Avg Delay (min)')
style_ax(ax9, '🌧 Precipitation vs Delay')

plt.suptitle('Public Transport Delay — Exploratory Analysis',
             color='white', fontsize=15, fontweight='bold', y=1.01)
plt.savefig('01_eda.png', bbox_inches='tight',
            facecolor='#0f1117', dpi=150)
plt.close()
print("EDA plot saved.")

# 5 — train / test split
print("STEP 5: TRAIN / TEST SPLIT")

TARGET_COL = 'actual_arrival_delay_min'
DROP_COLS  = ['delayed']          # binary label derived from target — leakage

X = df.drop(columns=[TARGET_COL] + DROP_COLS)
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
print(f"Features: {X_train.shape[1]}")

# 6 — train three models
print("STEP 6: TRAINING MODELS")

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest':     RandomForestRegressor(n_estimators=200, max_depth=12,
                                               min_samples_leaf=5, random_state=42, n_jobs=-1),
    'XGBoost':           XGBRegressor(n_estimators=300, max_depth=6,
                                      learning_rate=0.05, subsample=0.8,
                                      colsample_bytree=0.8, random_state=42,
                                      verbosity=0)
}

results = {}
for name, model in models.items():
    print(f"  Training {name}...", end='')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    results[name] = {'model': model, 'preds': preds, 'MAE': mae, 'RMSE': rmse, 'R2': r2}
    print(f"  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.3f}")

# 7 — model comparison + feature importance plot
print("STEP 7: MODEL COMPARISON PLOTS")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.patch.set_facecolor('#0f1117')

# Row 1: actual vs predicted for each model
for idx, (name, res) in enumerate(results.items()):
    ax = axes[0, idx]
    ax.set_facecolor('#1a1d27')
    ax.scatter(y_test, res['preds'], alpha=0.35, s=12, color=COLORS[idx])
    lo, hi = y_test.min(), y_test.max()
    ax.plot([lo, hi], [lo, hi], 'w--', linewidth=1.5, label='Perfect fit')
    ax.set_xlabel('Actual Delay (min)', color='#aaaaaa')
    ax.set_ylabel('Predicted Delay (min)', color='#aaaaaa')
    ax.set_title(f'{name}\nMAE={res["MAE"]:.2f}  R²={res["R2"]:.3f}',
                 color='white', fontsize=11, fontweight='bold')
    ax.tick_params(colors='#aaaaaa')
    ax.legend(facecolor='#1a1d27', labelcolor='white', fontsize=8)
    for spine in ax.spines.values(): spine.set_edgecolor('#333344')

# Row 2, col 0-1: metric comparison bar charts
metric_names = ['MAE', 'RMSE']
for midx, metric in enumerate(metric_names):
    ax = axes[1, midx]
    ax.set_facecolor('#1a1d27')
    vals = [results[n][metric] for n in results]
    bars = ax.bar(list(results.keys()), vals, color=COLORS[:3])
    ax.set_title(f'Model Comparison — {metric}', color='white', fontsize=11, fontweight='bold')
    ax.set_ylabel(metric, color='#aaaaaa')
    ax.tick_params(colors='#aaaaaa')
    for spine in ax.spines.values(): spine.set_edgecolor('#333344')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                f'{val:.2f}', ha='center', color='white', fontsize=10)

# Row 2, col 2: Feature importances (Random Forest)
ax = axes[1, 2]
ax.set_facecolor('#1a1d27')
rf_model  = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
top15 = importances.nlargest(15).sort_values()
colors_imp = plt.cm.plasma(np.linspace(0.3, 0.95, len(top15)))
ax.barh(top15.index, top15.values, color=colors_imp)
ax.set_title('Top 15 Feature Importances\n(Random Forest)', color='white',
             fontsize=11, fontweight='bold')
ax.tick_params(colors='#aaaaaa')
for spine in ax.spines.values(): spine.set_edgecolor('#333344')

plt.suptitle('Model Training Results & Evaluation',
             color='white', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('02_model_results.png', bbox_inches='tight',
            facecolor='#0f1117', dpi=150)
plt.close()
print("Model results plot saved.")

# 8 — residual analysis (best model)
print("STEP 8: RESIDUAL ANALYSIS")

best_name = min(results, key=lambda n: results[n]['MAE'])
best_res  = results[best_name]
print(f"Best model: {best_name}  (MAE = {best_res['MAE']:.2f})")

residuals = y_test.values - best_res['preds']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor('#0f1117')

# Residuals vs predicted
ax = axes[0]
ax.set_facecolor('#1a1d27')
ax.scatter(best_res['preds'], residuals, alpha=0.4, s=12, color='#4FC3F7')
ax.axhline(0, color='white', linestyle='--', linewidth=1.5)
ax.set_xlabel('Predicted Delay (min)', color='#aaaaaa')
ax.set_ylabel('Residual', color='#aaaaaa')
ax.set_title(f'Residuals vs Predicted\n({best_name})', color='white', fontweight='bold')
ax.tick_params(colors='#aaaaaa')
for sp in ax.spines.values(): sp.set_edgecolor('#333344')

# Residual distribution
ax = axes[1]
ax.set_facecolor('#1a1d27')
ax.hist(residuals, bins=30, color='#81C784', edgecolor='#0f1117')
ax.axvline(0, color='white', linestyle='--', linewidth=1.5)
ax.set_xlabel('Residual (min)', color='#aaaaaa')
ax.set_ylabel('Count', color='#aaaaaa')
ax.set_title('Residual Distribution', color='white', fontweight='bold')
ax.tick_params(colors='#aaaaaa')
for sp in ax.spines.values(): sp.set_edgecolor('#333344')

# R² comparison
ax = axes[2]
ax.set_facecolor('#1a1d27')
r2s = {n: results[n]['R2'] for n in results}
bars = ax.bar(list(r2s.keys()), list(r2s.values()), color=COLORS[:3])
ax.set_ylim(0, 1)
ax.set_ylabel('R² Score', color='#aaaaaa')
ax.set_title('R² Score Comparison', color='white', fontweight='bold')
ax.tick_params(colors='#aaaaaa')
for sp in ax.spines.values(): sp.set_edgecolor('#333344')
for bar, val in zip(bars, r2s.values()):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f'{val:.3f}', ha='center', color='white', fontsize=10)

plt.suptitle(f'Residual & Error Analysis — {best_name}',
             color='white', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('03_residuals.png', bbox_inches='tight',
            facecolor='#0f1117', dpi=150)
plt.close()
print("Residual plot saved.")

# Final summary
print("FINAL RESULTS SUMMARY")
for name, res in results.items():
    print(f"{name:20s}  MAE={res['MAE']:.2f} min  RMSE={res['RMSE']:.2f} min  R²={res['R2']:.3f}")
print(f"\nBest model: {best_name}")
print("Output files: 01_eda.png | 02_model_results.png | 03_residuals.png")
