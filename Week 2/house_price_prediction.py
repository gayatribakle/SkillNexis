"""
Mini Project 2: House Price Prediction Model
Using Linear Regression on Kaggle House Prices Dataset
Tasks:
- Data Preprocessing and Feature Engineering
- Train Linear Regression model
- Predict house prices
- Evaluate R² score, MAE, MSE, RMSE
- Plot predicted vs actual values
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Set plot styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def run_house_price_prediction(data_path='housing.csv', output_plot='house_price_prediction_plots.png'):
    print("=" * 70)
    print("MINI PROJECT 2: HOUSE PRICE PREDICTION MODEL (LINEAR REGRESSION)")
    print("=" * 70)

    # 1. Load Data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"\n[1] Dataset loaded successfully.")
    print(f"    - Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"    - Columns: {list(df.columns)}")
    print(f"    - Missing Values: {df.isnull().sum().sum()}")

    # 2. Data Preprocessing & Categorical Encoding
    print("\n[2] Preprocessing Data & Feature Encoding...")
    df_processed = df.copy()

    # Binary columns (yes/no -> 1/0)
    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    for col in binary_cols:
        df_processed[col] = df_processed[col].map({'yes': 1, 'no': 0})
    
    # Categorical column with 3 levels: furnishingstatus
    # Using drop_first=True to avoid dummy variable trap (multicollinearity)
    df_processed = pd.get_dummies(df_processed, columns=['furnishingstatus'], drop_first=True, dtype=int)

    print(f"    - Encoded binary columns: {binary_cols}")
    print(f"    - One-hot encoded 'furnishingstatus' -> {list(df_processed.columns[-2:])}")
    print(f"    - Final Feature Count: {df_processed.shape[1] - 1} features")

    # 3. Train-Test Split
    X = df_processed.drop('price', axis=1)
    y = df_processed['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"\n[3] Train-Test Split (80% Train, 20% Test):")
    print(f"    - Training samples: {X_train.shape[0]}")
    print(f"    - Testing samples:  {X_test.shape[0]}")

    # 4. Train Linear Regression Model
    print("\n[4] Training Linear Regression Model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("    - Model training completed.")
    print(f"    - Intercept (b0): ₹{model.intercept_:,.2f}")

    # Display Model Coefficients
    coeff_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', ascending=False)
    print("\n    Model Coefficients:")
    for _, row in coeff_df.iterrows():
        print(f"      • {row['Feature']:<28}: {row['Coefficient']:>14,.2f}")

    # 5. Predict House Prices
    print("\n[5] Predicting House Prices on Test Set...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 6. Evaluate Model Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    
    # Adjusted R2
    n = len(y_test)
    p = X_test.shape[1]
    adj_r2 = 1 - (1 - test_r2) * (n - 1) / (n - p - 1)

    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS (TEST SET):")
    print("=" * 50)
    print(f"  ● Train R² Score            : {train_r2:.4f} ({train_r2*100:.2f}%)")
    print(f"  ● Test R² Score             : {test_r2:.4f} ({test_r2*100:.2f}%)")
    print(f"  ● Adjusted R² Score         : {adj_r2:.4f} ({adj_r2*100:.2f}%)")
    print(f"  ● Mean Absolute Error (MAE) : ₹{mae:,.2f}")
    print(f"  ● Mean Squared Error (MSE)  : {mse:,.2f}")
    print(f"  ● Root Mean Sq Error (RMSE) : ₹{rmse:,.2f}")
    print("=" * 50)

    # Display Sample Predictions vs Actual
    sample_comparison = pd.DataFrame({
        'Actual Price': y_test.values[:10],
        'Predicted Price': y_test_pred[:10],
        'Difference (Residual)': y_test.values[:10] - y_test_pred[:10],
        'Percentage Error (%)': np.abs((y_test.values[:10] - y_test_pred[:10]) / y_test.values[:10]) * 100
    })
    print("\nSample 10 Predictions vs Actual Prices:")
    print(sample_comparison.to_string(index=False, formatters={
        'Actual Price': '₹{:,.0f}'.format,
        'Predicted Price': '₹{:,.0f}'.format,
        'Difference (Residual)': '₹{:,.0f}'.format,
        'Percentage Error (%)': '{:.2f}%'.format
    }))

    # 7. Visualizations
    print(f"\n[7] Generating Evaluation Plots -> {output_plot}...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.patch.set_facecolor('#f8f9fa')

    # Plot 1: Actual vs Predicted Prices Scatter
    ax1 = axes[0, 0]
    ax1.set_facecolor('white')
    ax1.scatter(y_test / 1e6, y_test_pred / 1e6, color='#2563eb', alpha=0.7, edgecolors='k', s=60, label='Test Houses')
    min_val = min(y_test.min(), y_test_pred.min()) / 1e6
    max_val = max(y_test.max(), y_test_pred.max()) / 1e6
    ax1.plot([min_val, max_val], [min_val, max_val], color='#dc2626', linestyle='--', linewidth=2, label='Perfect Fit (y = x)')
    ax1.set_title(f'Actual vs. Predicted House Prices\n(R² = {test_r2:.4f})', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel('Actual Price (in Millions ₹)', fontsize=11, fontweight='semibold')
    ax1.set_ylabel('Predicted Price (in Millions ₹)', fontsize=11, fontweight='semibold')
    ax1.legend(loc='upper left', frameon=True)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Plot 2: Residuals Distribution (Errors)
    ax2 = axes[0, 1]
    ax2.set_facecolor('white')
    residuals = (y_test - y_test_pred) / 1e6
    sns.histplot(residuals, kde=True, ax=ax2, color='#0d9488', bins=20, edgecolor='black', alpha=0.6)
    ax2.axvline(0, color='#dc2626', linestyle='--', linewidth=2, label='Zero Error')
    ax2.set_title('Residuals (Error) Distribution', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel('Residual: Actual - Predicted (in Millions ₹)', fontsize=11, fontweight='semibold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='semibold')
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Plot 3: Residuals vs Predicted Values (Homoscedasticity Check)
    ax3 = axes[1, 0]
    ax3.set_facecolor('white')
    ax3.scatter(y_test_pred / 1e6, residuals, color='#7c3aed', alpha=0.7, edgecolors='k', s=60)
    ax3.axhline(0, color='#dc2626', linestyle='--', linewidth=2, label='Zero Residual Line')
    ax3.set_title('Residuals vs. Predicted Values', fontsize=13, fontweight='bold', pad=10)
    ax3.set_xlabel('Predicted Price (in Millions ₹)', fontsize=11, fontweight='semibold')
    ax3.set_ylabel('Residual (in Millions ₹)', fontsize=11, fontweight='semibold')
    ax3.legend(loc='upper right', frameon=True)
    ax3.grid(True, linestyle=':', alpha=0.6)

    # Plot 4: Feature Importance (Coefficients)
    ax4 = axes[1, 1]
    ax4.set_facecolor('white')
    colors = ['#16a34a' if c > 0 else '#dc2626' for c in coeff_df['Coefficient']]
    ax4.barh(coeff_df['Feature'], coeff_df['Coefficient'] / 1e5, color=colors, edgecolor='black', alpha=0.85)
    ax4.axvline(0, color='black', linestyle='-', linewidth=0.8)
    ax4.set_title('Linear Regression Feature Coefficients', fontsize=13, fontweight='bold', pad=10)
    ax4.set_xlabel('Coefficient Impact (in ₹ Lakhs)', fontsize=11, fontweight='semibold')
    ax4.set_ylabel('Features', fontsize=11, fontweight='semibold')
    ax4.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout(pad=3.0)
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    - Plot saved successfully as '{output_plot}'.")
    print("\nExecution of Mini Project 2 completed successfully!")

    return {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'adj_r2': adj_r2,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'sample_comparison': sample_comparison
    }

if __name__ == '__main__':
    run_house_price_prediction()
