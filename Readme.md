# 🛳️ Titanic Dataset Preprocessing

This project performs **data cleaning and preprocessing** on the Titanic dataset using **Python, Pandas, NumPy, and Matplotlib**.  
The goal is to prepare the data for machine learning tasks by handling missing values, encoding categorical variables, standardizing numerical features, and removing outliers.

---

## 📌 Features

- Load Titanic dataset from CSV
- Display dataset shape, missing values, and info
- Handle missing values:
  - Fill `Age` and `Fare` with **median**
  - Fill `Embarked` with **mode**
- Drop unnecessary columns (`Cabin`)
- Encode categorical features:
  - `Sex` → numeric (male = 0, female = 1)
  - `Embarked` → one-hot encoding (Q, S)
- Detect and remove outliers using **IQR method**
- Visualize outliers using **boxplots (before & after)**
- Standardize numerical features (`Age`, `Fare`) using **z-score normalization**
- Save the **cleaned dataset** as `titanic_preprocessed.csv`

---

## 📂 Project Structure

├── index.py # Main preprocessing script

├── TitanicDataset.csv # Input dataset (provide your own path)

├── titanic_preprocessed.csv # Cleaned dataset (output)

├── README.md # Project documentation


---

## ⚙️ Requirements

Install dependencies using pip:

```bash
pip install pandas numpy matplotlib
```

## ▶️ Usage

1.Place your Titanic dataset in the project folder (example: TitanicDataset.csv).

2.Run the script:
``` bash
python index.py
```
3.The cleaned dataset will be saved as:

```
titanic_preprocessed.csv
```