---

# Penguin Body Mass Prediction using Linear Regression

This project demonstrates a data preprocessing, modeling, and evaluation pipeline for predicting penguin body mass using linear regression. The dataset is loaded from seaborn's built-in penguins dataset.

## Requirements

- Python 3.x
- NumPy
- Pandas
- Seaborn
- Matplotlib
- scikit-learn

## Overview

The script performs the following steps:
1. **Data Loading and Cleaning:**  
   - Loads the penguins dataset.
   - Removes rows with missing values in categorical columns.
   - Fills missing values in numerical columns using the median.

2. **Standardization of Variables:**  
   - Standardizes numerical variables (`bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, and `body_mass_g`).

3. **Encoding Categorical Variables:**  
   - Converts categorical columns (species, island, and sex) into dummy variables.

4. **Data Preparation:**  
   - Filters columns to include only standardized and encoded variables for the modeling.

5. **Model Training and Evaluation:**  
   - Splits the data into training and testing sets.
   - Trains a linear regression model.
   - Evaluates the model using the RMSE metric.

6. **Model Testing:**  
   - Applies the model to a new test sample with standardized features.

## Usage

To run the script, execute the following command in your terminal:

```bash
python script.py
```

Ensure you have installed all required libraries before running the script.

## Code

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the penguins dataset from seaborn
penguin = sns.load_dataset('penguins')

# Handling missing values

# For categorical columns, drop rows with null values
penguin = penguin.dropna(subset=['species', 'island', 'sex']).copy()    
    # 'subset' analyzes only the specified columns
    # '.copy()' avoids a SettingWithCopyWarning

# For numerical columns, replace missing values with the median
numeric_columns = penguin.select_dtypes(include=['float64', 'int64']).columns

for col in numeric_columns:
    penguin.loc[:, col] = penguin[col].fillna(penguin[col].median())

# Standardizing numerical variables
cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

for col in cols:
    penguin.loc[:, f'{col}_std'] = (penguin[col] - penguin[col].mean()) / penguin[col].std()

# Encoding categorical variables

penguin.loc[:, 'species_Adelie_nom'] = penguin['species'].apply(lambda species: 1 if species == 'Adelie' else 0)
penguin.loc[:, 'species_Chinstrap_nom'] = penguin['species'].apply(lambda species: 1 if species == 'Chinstrap' else 0)
penguin.loc[:, 'species_Gentoo_nom'] = penguin['species'].apply(lambda species: 1 if species == 'Gentoo' else 0)
penguin.loc[:, 'island_Torgersen_nom'] = penguin['island'].apply(lambda island: 1 if island == 'Torgersen' else 0)
penguin.loc[:, 'island_Dream_nom'] = penguin['island'].apply(lambda island: 1 if island == 'Dream' else 0)
penguin.loc[:, 'island_Biscoe_nom'] = penguin['island'].apply(lambda island: 1 if island == 'Biscoe' else 0)

penguin.loc[:, 'sex_nom'] = penguin['sex'].apply(lambda sex: 1 if sex == 'M' else 0)

# Drop the original categorical columns
penguin = penguin.drop(['species', 'island', 'sex'], axis=1)

# Cleaning data by selecting only standardized and encoded columns
final_data_penguin = [col for col in penguin.columns if col.endswith(('_std', '_nom', '_ord')) and col != 'body_mass_g_std'] + ['body_mass_g']
penguin = penguin[final_data_penguin]

print(penguin.head())

# Splitting the data into training and testing sets
predictors_train, predictors_test, target_train, target_test = train_test_split(
    penguin.drop(['body_mass_g'], axis=1),
    penguin['body_mass_g'],
    test_size=1/3,
    random_state=123  # arbitrary random seed
)

# Training the linear regression model
model = LinearRegression()
model = model.fit(predictors_train, target_train)

# Evaluating the model
target_predicted = model.predict(predictors_test)

rmse = np.sqrt(mean_squared_error(target_test, target_predicted))
print(rmse)  # RMSE is approximately 324.44g

# Testing the model with a new sample
penguin_test_df = sns.load_dataset('penguins')

bill_length_test = (38.2 - penguin_test_df['bill_length_mm'].mean()) / penguin_test_df['bill_length_mm'].std()
bill_depth_test = (18.1 - penguin_test_df['bill_depth_mm'].mean()) / penguin_test_df['bill_depth_mm'].std()
flipper_length_test = (185.0 - penguin_test_df['flipper_length_mm'].mean()) / penguin_test_df['flipper_length_mm'].std()

test_penguin = pd.DataFrame({
    'bill_length_mm_std': [bill_length_test],
    'bill_depth_mm_std': [bill_depth_test],
    'flipper_length_mm_std': [flipper_length_test],
    'species_Adelie_nom': [1],
    'species_Chinstrap_nom': [0],
    'species_Gentoo_nom': [0],
    'island_Torgersen_nom': [0],
    'island_Dream_nom': [0],
    'island_Biscoe_nom': [1],
    'sex_nom': [1]
})

predicted_body_mass = model.predict(test_penguin)
print(predicted_body_mass)  # predicted_body_mass is approximately 3556.53g
```

## License

This project is licensed under the MIT License.

## Acknowledgements

This project uses the penguins dataset provided by seaborn.  
Happy coding!

---