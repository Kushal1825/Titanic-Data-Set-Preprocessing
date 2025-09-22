import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Show dataset basic info
def show_basic_info(df, name="Dataset"):
    print(f"\n == {name} shape: {df.shape} ==")
    print(df.head(3).to_string(index=False))
    print("\nInfo:")
    print(df.info())
    print("\nMissing values (counts):")
    print(df.isnull().sum())
    print("\nMissing values (%):")
    print((df.isnull().mean() * 100).round(2))


# Outlier removal using IQR
def remove_outliers_iqr(df, cols):
    df = df.copy()
    initial_len = len(df)
    for col in cols:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        before = len(df)
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        after = len(df)
        print(f"Outlier removal on '{col}': removed {before - after} rows")
    print(f"Total rows removed: {initial_len - len(df)}")
    return df


# Compare boxplots before/after outlier removal
def plot_boxplots_before_after(df_before, df_after, cols, suptitle=None):
    n = len(cols)
    plt.figure(figsize=(6 * n, 5))
    for i, col in enumerate(cols, start=1):
        plt.subplot(2, n, i)
        plt.boxplot(df_before[col].dropna(), vert=False)
        plt.title(f"Before: {col}")
        plt.subplot(2, n, i + n)
        plt.boxplot(df_after[col].dropna(), vert=False)
        plt.title(f"After: {col}")
    if suptitle:
        plt.suptitle(suptitle)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# Standardize numerical columns
def standardize_columns(df, cols):
    for col in cols:
        mean = df[col].mean()
        std = df[col].std()
        if std == 0 or np.isnan(std):
            raise ValueError(f"Standard deviation for {col} is zero or NaN.")
        df[col] = (df[col] - mean) / std
    return df


# Full preprocessing pipeline
def preprocess_titanic(df, remove_outliers=True, drop_cols=("Cabin",)):
    df = df.copy()

    # 1) Basic info

    show_basic_info(df, name="Original dataset")

    # 2) Drop unwanted columns

    for c in drop_cols:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # 3) Handle missing values

    if "Age" in df.columns:
        df["Age"].fillna(df["Age"].median(), inplace=True)

    if "Fare" in df.columns:
        df["Fare"].fillna(df["Fare"].median(), inplace=True)

    if "Embarked" in df.columns:
        df["Embarked"].fillna(df["Embarked"].mode().iloc[0], inplace=True)

    print("\nAfter imputation missing counts:")
    print(df.isnull().sum())

    # 4) Visualize outliers before removal

    cols_to_check = [c for c in ["Age", "Fare"] if c in df.columns]

    # 5) Remove outliers

    if remove_outliers and cols_to_check:
        df_after = remove_outliers_iqr(df, cols_to_check)
        print("\nPlotting boxplots BEFORE vs AFTER outlier removal...")
        plot_boxplots_before_after(df, df_after, cols_to_check,suptitle="Before vs After IQR Outlier Removal")
        df = df_after

    # 6) Encode categorical features

    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    if "Embarked" in df.columns:
        df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

    # 7) Standardize numerical columns

    num_cols = [c for c in ["Age", "Fare"] if c in df.columns]

    if num_cols:
        df = standardize_columns(df, num_cols)

    print("\nAfter preprocessing preview:")
    show_basic_info(df, name="Preprocessed dataset")

    return df


if __name__ == "__main__":
    

    df_raw = pd.read_csv("TitanicDataset.csv")

    # Run preprocessing

    processed_df = preprocess_titanic(df_raw, remove_outliers=True)

    # Save cleaned dataset

    processed_df.to_csv("./titanic_preprocessed.csv", index=False)
    print("\nPreprocessing complete. Cleaned dataset saved as 'titanic_preprocessed.csv'")
