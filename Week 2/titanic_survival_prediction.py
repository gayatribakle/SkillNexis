"""
Task 2: Titanic Survival Prediction Model
Using Logistic Regression on the Titanic Dataset
Tasks:
- Exploratory Data Analysis & Handling Missing Values
- Feature Engineering & Encoding
- Feature Scaling using StandardScaler
- Train Logistic Regression model
- Predict survival and evaluate Accuracy, Confusion Matrix, Classification Report, ROC-AUC
- Plot Confusion Matrix, ROC Curve, Feature Importances, and Probability Distributions
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def run_titanic_survival_prediction(data_path='titanic.csv', output_plot='titanic_survival_plots.png'):
    print("=" * 70)
    print("TASK 2: TITANIC SURVIVAL PREDICTION MODEL (LOGISTIC REGRESSION)")
    print("=" * 70)

    # 1. Load Data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"\n[1] Titanic Dataset loaded successfully.")
    print(f"    - Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"    - Target distribution:\n{df['survived'].value_counts(normalize=True).rename({0: 'Did Not Survive (0)', 1: 'Survived (1)'}).map('{:.2%}'.format)}")

    # 2. Preprocessing & Feature Engineering
    print("\n[2] Preprocessing Data & Handling Missing Values...")
    df_clean = df.copy()

    # Retain core predictive features
    features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    target = 'survived'
    df_clean = df_clean[features + [target]].copy()

    # Missing value imputation:
    # Impute Age with median age grouped by pclass and sex (realistic and accurate)
    df_clean['age'] = df_clean.groupby(['pclass', 'sex'])['age'].transform(lambda x: x.fillna(x.median()))
    # Impute Embarked with mode
    mode_embarked = df_clean['embarked'].mode()[0]
    df_clean['embarked'] = df_clean['embarked'].fillna(mode_embarked)
    # Impute Fare with median if any missing
    df_clean['fare'] = df_clean['fare'].fillna(df_clean['fare'].median())

    # Feature Engineering
    df_clean['family_size'] = df_clean['sibsp'] + df_clean['parch'] + 1
    df_clean['is_alone'] = (df_clean['family_size'] == 1).astype(int)

    # Encoding Categorical Variables
    # Binary encoding: sex (female=1, male=0)
    df_clean['sex'] = df_clean['sex'].map({'female': 1, 'male': 0})
    
    # One-hot encoding embarked (drop_first=True to avoid dummy trap)
    df_clean = pd.get_dummies(df_clean, columns=['embarked'], drop_first=True, dtype=int)

    print(f"    - Missing values after imputation: {df_clean.isnull().sum().sum()}")
    print(f"    - Engineered features: 'family_size', 'is_alone'")
    print(f"    - Encoded 'sex' (female=1, male=0) and 'embarked' dummy variables")
    print(f"    - Total predictor features: {df_clean.shape[1] - 1}")

    # 3. Features & Target Split
    X = df_clean.drop('survived', axis=1)
    y = df_clean['survived']

    # Train-test split (80% train, 20% test with stratification)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[3] Train-Test Split (80% Train, 20% Test, Stratified):")
    print(f"    - Training set size: {X_train.shape[0]} samples")
    print(f"    - Test set size:     {X_test.shape[0]} samples")

    # Feature Scaling using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train Logistic Regression Model
    print("\n[4] Training Logistic Regression Model...")
    model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
    model.fit(X_train_scaled, y_train)
    print("    - Logistic Regression training completed successfully.")
    print(f"    - Model Intercept: {model.intercept_[0]:.4f}")

    # Display Coefficients and Odds Ratios
    coeff_summary = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient (Log-Odds)': model.coef_[0],
        'Odds Ratio (e^coef)': np.exp(model.coef_[0])
    }).sort_values(by='Coefficient (Log-Odds)', ascending=False)

    print("\n    Model Coefficients & Odds Ratios (Impact on Survival Odds):")
    for _, row in coeff_summary.iterrows():
        direction = "Positive (Increases Survival)" if row['Coefficient (Log-Odds)'] > 0 else "Negative (Decreases Survival)"
        print(f"      • {row['Feature']:<16}: Coef={row['Coefficient (Log-Odds)']:>7.4f} | Odds Ratio={row['Odds Ratio (e^coef)']:>6.3f} ({direction})")

    # 5. Predict Survival
    print("\n[5] Predicting Survival on Test Set...")
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    y_test_prob = model.predict_proba(X_test_scaled)[:, 1]

    # 6. Evaluation Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_prob)
    cm = confusion_matrix(y_test, y_test_pred)

    print("\n" + "=" * 55)
    print("LOGISTIC REGRESSION EVALUATION METRICS (TEST SET):")
    print("=" * 55)
    print(f"  ● Training Accuracy         : {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  ● Testing Accuracy          : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  ● Precision (Class 1)       : {precision:.4f} ({precision*100:.2f}%)")
    print(f"  ● Recall (Class 1)          : {recall:.4f} ({recall*100:.2f}%)")
    print(f"  ● F1-Score (Class 1)        : {f1:.4f}")
    print(f"  ● ROC-AUC Score             : {roc_auc:.4f}")
    print("=" * 55)

    print("\nConfusion Matrix:")
    print(f"  True Negatives (Correctly predicted Non-survivors) : {cm[0, 0]}")
    print(f"  False Positives (Incorrectly predicted Survivors)  : {cm[0, 1]}")
    print(f"  False Negatives (Incorrectly predicted Non-surv)   : {cm[1, 0]}")
    print(f"  True Positives (Correctly predicted Survivors)     : {cm[1, 1]}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Perished (0)', 'Survived (1)']))

    # Sample Predictions
    sample_res = pd.DataFrame({
        'Pclass': X_test['pclass'].values[:10],
        'Sex': ['Female' if s == 1 else 'Male' for s in X_test['sex'].values[:10]],
        'Age': X_test['age'].values[:10],
        'Actual': ['Survived' if a == 1 else 'Perished' for a in y_test.values[:10]],
        'Predicted': ['Survived' if p == 1 else 'Perished' for p in y_test_pred[:10]],
        'Survival Prob': [f"{p:.1%}" for p in y_test_prob[:10]],
        'Match': ['✓' if a == p else '✗' for a, p in zip(y_test.values[:10], y_test_pred[:10])]
    })
    print("Sample 10 Test Predictions:")
    print(sample_res.to_string(index=False))

    # 7. Visualizations
    print(f"\n[7] Generating Titanic Visualizations -> {output_plot}...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.patch.set_facecolor('#f8f9fa')

    # Subplot 1: Confusion Matrix Heatmap
    ax1 = axes[0, 0]
    ax1.set_facecolor('white')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1,
                annot_kws={'size': 14, 'weight': 'bold'},
                xticklabels=['Predicted Perished (0)', 'Predicted Survived (1)'],
                yticklabels=['Actual Perished (0)', 'Actual Survived (1)'])
    ax1.set_title(f'Confusion Matrix\n(Accuracy: {test_acc:.2%})', fontsize=13, fontweight='bold', pad=10)
    ax1.set_ylabel('Actual Label', fontsize=11, fontweight='semibold')
    ax1.set_xlabel('Predicted Label', fontsize=11, fontweight='semibold')

    # Subplot 2: ROC Curve
    ax2 = axes[0, 1]
    ax2.set_facecolor('white')
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    ax2.plot(fpr, tpr, color='#2563eb', linewidth=2.5, label=f'Logistic Regression (AUC = {roc_auc:.4f})')
    ax2.plot([0, 1], [0, 1], color='#dc2626', linestyle='--', linewidth=1.5, label='Random Guess (AUC = 0.50)')
    ax2.set_title(f'Receiver Operating Characteristic (ROC) Curve\n(AUC = {roc_auc:.4f})', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='semibold')
    ax2.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11, fontweight='semibold')
    ax2.legend(loc='lower right', frameon=True)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Subplot 3: Feature Coefficients (Log-Odds Impact)
    ax3 = axes[1, 0]
    ax3.set_facecolor('white')
    colors = ['#16a34a' if c > 0 else '#dc2626' for c in coeff_summary['Coefficient (Log-Odds)']]
    ax3.barh(coeff_summary['Feature'], coeff_summary['Coefficient (Log-Odds)'], color=colors, edgecolor='black', alpha=0.85)
    ax3.axvline(0, color='black', linestyle='-', linewidth=0.8)
    ax3.set_title('Feature Importance (Standardized Coefficients)', fontsize=13, fontweight='bold', pad=10)
    ax3.set_xlabel('Log-Odds Coefficient Impact', fontsize=11, fontweight='semibold')
    ax3.set_ylabel('Features', fontsize=11, fontweight='semibold')
    ax3.grid(True, linestyle=':', alpha=0.6)

    # Subplot 4: Predicted Probability Distribution by Actual Class
    ax4 = axes[1, 1]
    ax4.set_facecolor('white')
    sns.kdeplot(y_test_prob[y_test == 0], ax=ax4, label='Actual Perished (0)', color='#dc2626', fill=True, alpha=0.4)
    sns.kdeplot(y_test_prob[y_test == 1], ax=ax4, label='Actual Survived (1)', color='#16a34a', fill=True, alpha=0.4)
    ax4.axvline(0.5, color='black', linestyle='--', linewidth=1.5, label='Decision Threshold (0.5)')
    ax4.set_title('Predicted Probability Distribution by Actual Class', fontsize=13, fontweight='bold', pad=10)
    ax4.set_xlabel('Predicted Probability of Survival P(Y=1)', fontsize=11, fontweight='semibold')
    ax4.set_ylabel('Density', fontsize=11, fontweight='semibold')
    ax4.legend(loc='upper center', frameon=True)
    ax4.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout(pad=3.0)
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    - Plot saved successfully as '{output_plot}'.")
    print("\nExecution of Titanic Survival Prediction completed successfully!")

    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }

if __name__ == '__main__':
    run_titanic_survival_prediction()
