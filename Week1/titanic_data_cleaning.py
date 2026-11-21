"""
Titanic Survival Prediction - Data Cleaning & Preprocessing
Dataset: Titanic Dataset (Kaggle)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Set plotting aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['figure.dpi'] = 300

DATA_PATH = r"d:\Week1\titanic.csv"
OUTPUT_CSV = r"d:\Week1\titanic_cleaned.csv"
PLOT_PATH = r"d:\Week1\age_distribution.png"

def main():
    print("=" * 80)
    print("ASSIGNMENT 1: LOAD DATASET & SUMMARIZE BASIC STATS")
    print("=" * 80)
    
    # 1. Load the dataset
    df = pd.read_csv(DATA_PATH)
    print(f"\n[+] Successfully loaded dataset from: {DATA_PATH}")
    print(f"[+] Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    
    print("\n--- First 5 Rows (df.head()) ---")
    print(df.head())

    print("\n--- Dataset Summary (df.info()) ---")
    df.info()

    print("\n--- Numerical Summary Statistics (df.describe()) ---")
    print(df.describe().T)

    print("\n--- Categorical Summary Statistics (df.describe(include=['O'])) ---")
    print(df.describe(include=['O']).T)

    print("\n--- Missing Value Audit ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Missing_Count': missing, 'Percentage (%)': missing_pct})
    print(missing_df[missing_df['Missing_Count'] > 0])

    print("\n" + "=" * 80)
    print("ASSIGNMENT 2: HANDLE MISSING DATA USING MEAN/MEDIAN IMPUTATION")
    print("=" * 80)
    
    # Analyze Age distribution before imputation
    age_mean = df['Age'].mean()
    age_median = df['Age'].median()
    age_std = df['Age'].std()
    age_skew = df['Age'].skew()
    print(f"Age Statistics (Before Imputation):")
    print(f"  Mean   : {age_mean:.2f}")
    print(f"  Median : {age_median:.2f}")
    print(f"  Std Dev: {age_std:.2f}")
    print(f"  Skew   : {age_skew:.2f} (Mild positive right skew)")

    # Demonstration: Mean vs Median Imputation comparison
    age_mean_imputed = df['Age'].fillna(age_mean)
    age_median_imputed = df['Age'].fillna(age_median)
    print(f"\nComparison of Imputation Strategies on Age:")
    print(f"  Original Age Std Dev       : {df['Age'].std():.4f}")
    print(f"  Mean Imputed Age Std Dev   : {age_mean_imputed.std():.4f} (variance reduced)")
    print(f"  Median Imputed Age Std Dev : {age_median_imputed.std():.4f} (robust against outlier influence)")

    # We also show scikit-learn SimpleImputer usage
    imputer_mean = SimpleImputer(strategy='mean')
    imputer_median = SimpleImputer(strategy='median')
    
    # Create working copy for data cleaning
    df_clean = df.copy()

    # 1. Age Imputation: Median is chosen as the standard for Titanic because age is right-skewed
    df_clean['Age_Median_Imputed'] = imputer_median.fit_transform(df_clean[['Age']])
    # In the final clean df, we update 'Age' directly with the median
    df_clean['Age'] = df_clean['Age_Median_Imputed']
    df_clean.drop(columns=['Age_Median_Imputed'], inplace=True)
    print(f"[+] 'Age' missing values imputed using Median ({age_median:.1f} years). Remaining nulls: {df_clean['Age'].isnull().sum()}")

    # 2. Embarked Imputation: 2 missing values imputed using Mode ('S')
    embarked_mode = df_clean['Embarked'].mode()[0]
    df_clean['Embarked'].fillna(embarked_mode, inplace=True)
    print(f"[+] 'Embarked' missing values imputed using Mode ('{embarked_mode}'). Remaining nulls: {df_clean['Embarked'].isnull().sum()}")

    # 3. Cabin: 687 missing values (77.1%). Imputing ~77% values introduces heavy noise.
    # We drop 'Cabin' for the standard tabular model, or record whether a passenger had a Cabin.
    df_clean['Has_Cabin'] = df_clean['Cabin'].notnull().astype(int)
    df_clean.drop(columns=['Cabin'], inplace=True)
    print(f"[+] 'Cabin' dropped due to {missing_pct['Cabin']:.1f}% missingness (engineered 'Has_Cabin' flag).")

    print("\n--- Missing Values in Cleaned DataFrame ---")
    print(df_clean.isnull().sum())

    print("\n" + "=" * 80)
    print("ASSIGNMENT 3: ENCODE CATEGORICAL VARIABLES (LabelEncoder & OneHotEncoder)")
    print("=" * 80)

    # A. LabelEncoder Demonstration
    print("\n--- Part A: Scikit-Learn LabelEncoder ---")
    # LabelEncoder is suited for binary categorical variables or target variables
    le_sex = LabelEncoder()
    df_clean['Sex_LabelEncoded'] = le_sex.fit_transform(df_clean['Sex'])
    print("Sex Label Encoding Mapping:")
    for idx, class_name in enumerate(le_sex.classes_):
        print(f"  '{class_name}' -> {idx}")

    le_embarked = LabelEncoder()
    embarked_le_demo = le_embarked.fit_transform(df_clean['Embarked'])
    print("\nEmbarked Label Encoding Mapping (Demonstration):")
    for idx, class_name in enumerate(le_embarked.classes_):
        print(f"  '{class_name}' -> {idx}")
    print("  Note: LabelEncoder assigns integer ranks (0, 1, 2) which may mislead models into assuming order (Q > C > S).")

    # B. OneHotEncoder Demonstration
    print("\n--- Part B: Scikit-Learn OneHotEncoder ---")
    # OneHotEncoder is optimal for nominal categories without natural ordering (e.g. Embarked)
    ohe = OneHotEncoder(sparse_output=False, drop=None)
    embarked_ohe = ohe.fit_transform(df_clean[['Embarked']])
    embarked_feature_names = [f"Embarked_{cat}" for cat in ohe.categories_[0]]
    embarked_ohe_df = pd.DataFrame(embarked_ohe, columns=embarked_feature_names, index=df_clean.index)
    print("OneHotEncoder Embarked Columns Preview:")
    print(pd.concat([df_clean[['Embarked']], embarked_ohe_df], axis=1).head(5))

    # Incorporate encoding into df_clean:
    # 1. Sex: Replace with binary 0/1 (female=0, male=1)
    df_clean['Sex'] = df_clean['Sex_LabelEncoded']
    df_clean.drop(columns=['Sex_LabelEncoded'], inplace=True)

    # 2. Embarked: One-hot encode using pd.get_dummies / Scikit-Learn OHE
    df_clean = pd.concat([df_clean, embarked_ohe_df], axis=1)
    # Drop original 'Embarked' column now that one-hot dummy columns exist
    df_clean.drop(columns=['Embarked'], inplace=True)

    print("\nCleaned & Encoded Columns:")
    print(df_clean.columns.tolist())

    print("\n" + "=" * 80)
    print("MINI PROJECT 1: VISUALIZE AGE DISTRIBUTION (Matplotlib & Seaborn)")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Titanic Survival Analysis: Age Distribution & Cleaning Visualizations', fontsize=18, fontweight='bold', y=0.98)

    # 1. Subplot 1: Age Distribution Before vs After Imputation
    ax1 = axes[0, 0]
    sns.histplot(df['Age'].dropna(), color='#1f77b4', kde=True, stat="density", label='Original Age (Before Imputation)', ax=ax1, bins=30, alpha=0.5)
    sns.histplot(df_clean['Age'], color='#ff7f0e', kde=True, stat="density", label='Imputed Age (Median = 28.0)', ax=ax1, bins=30, alpha=0.3)
    ax1.axvline(age_median, color='red', linestyle='--', linewidth=2, label=f'Median: {age_median:.1f}')
    ax1.axvline(age_mean, color='green', linestyle=':', linewidth=2, label=f'Mean: {age_mean:.1f}')
    ax1.set_title('1. Age Distribution: Raw vs Median Imputed', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Age (Years)', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.legend(loc='upper right', frameon=True)

    # 2. Subplot 2: Age Distribution by Survival Status
    ax2 = axes[0, 1]
    sns.kdeplot(data=df_clean[df_clean['Survived'] == 0]['Age'], label='Did Not Survive (0)', color='#d62728', fill=True, alpha=0.4, linewidth=2, ax=ax2)
    sns.kdeplot(data=df_clean[df_clean['Survived'] == 1]['Age'], label='Survived (1)', color='#2ca02c', fill=True, alpha=0.4, linewidth=2, ax=ax2)
    ax2.set_title('2. Age Density by Survival Status (KDE)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Age (Years)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.legend(loc='upper right', frameon=True)

    # 3. Subplot 3: Age Distribution Across Passenger Classes (Pclass)
    ax3 = axes[1, 0]
    sns.boxplot(x='Pclass', y='Age', data=df_clean, palette='Blues_r', ax=ax3, width=0.5, boxprops=dict(alpha=0.8), hue='Pclass', legend=False)
    sns.stripplot(x='Pclass', y='Age', data=df_clean, color='black', alpha=0.2, jitter=0.2, size=3, ax=ax3)
    ax3.set_title('3. Age Distribution by Passenger Class (Pclass)', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)', fontsize=11)
    ax3.set_ylabel('Age (Years)', fontsize=11)

    # 4. Subplot 4: Age Distribution by Gender and Survival Status
    ax4 = axes[1, 1]
    # In df_clean: Sex 0 = female, 1 = male
    df_plot = df_clean.copy()
    df_plot['Gender'] = df_plot['Sex'].map({0: 'Female', 1: 'Male'})
    df_plot['Survival_Status'] = df_plot['Survived'].map({0: 'Perished', 1: 'Survived'})
    sns.violinplot(x='Gender', y='Age', hue='Survival_Status', data=df_plot, split=True, palette={'Perished': '#e74c3c', 'Survived': '#2ecc71'}, inner='quartile', ax=ax4)
    ax4.set_title('4. Age Distribution by Gender and Survival (Split Violin)', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Gender', fontsize=11)
    ax4.set_ylabel('Age (Years)', fontsize=11)
    ax4.legend(title='Outcome', loc='upper right', frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(PLOT_PATH, bbox_inches='tight')
    plt.close()
    print(f"[+] Visualization saved successfully to: {PLOT_PATH}")

    print("\n" + "=" * 80)
    print("OUTPUT CLEANED DATASET TO NEW CSV")
    print("=" * 80)
    
    # Drop raw string identifier columns unsuited for numerical models
    columns_to_drop_for_ml = ['PassengerId', 'Name', 'Ticket']
    df_final_ml = df_clean.drop(columns=columns_to_drop_for_ml)
    
    # Save the cleaned CSV
    df_final_ml.to_csv(OUTPUT_CSV, index=False)
    print(f"[+] Cleaned dataset exported to: {OUTPUT_CSV}")
    print(f"[+] Final Cleaned Dataset Shape: {df_final_ml.shape}")
    print("\nFirst 5 rows of cleaned dataset:")
    print(df_final_ml.head())
    
    print("\nMissing values check in cleaned dataset:")
    print(df_final_ml.isnull().sum())
    print("\n[✓] ALL TASKS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
