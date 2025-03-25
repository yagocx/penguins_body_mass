import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

penguin = sns.load_dataset('penguins')

# Tratando dados nulos

# para colunas categóricas, descarte-as
penguin = penguin.dropna(subset=['species', 'island', 'sex']).copy()    
    #subset analisa apenas as colunas especificadas
    #.copy evita o aparecimento de um SettingWithCopyWarning

# para colunas numéricas, substitua pela mediana
numeric_columns = penguin.select_dtypes(include=['float64', 'int64']).columns

for col in numeric_columns:
    penguin.loc[:, col] = penguin[col].fillna(penguin[col].median())

# Padronizando variáveis numéricas
cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

for col in cols:
    penguin.loc[:, f'{col}_std'] = (penguin[col] - penguin[col].mean()) / penguin[col].std()

# Padronizando variáveis categóricas

penguin.loc[:, 'species_Adelie_nom'] = penguin['species'].apply(lambda species: 1 if species == 'Adelie' else 0)
penguin.loc[:, 'species_Chinstrap_nom'] = penguin['species'].apply(lambda species: 1 if species == 'Chinstrap' else 0)
penguin.loc[:, 'species_Gentoo_nom'] = penguin['species'].apply(lambda species: 1 if species == 'Gentoo' else 0)
penguin.loc[:, 'island_Torgersen_nom'] = penguin['island'].apply(lambda island: 1 if island == 'Torgersen' else 0)
penguin.loc[:, 'island_Dream_nom'] = penguin['island'].apply(lambda island: 1 if island == 'Dream' else 0)
penguin.loc[:, 'island_Biscoe_nom'] = penguin['island'].apply(lambda island: 1 if island == 'Biscoe' else 0)

penguin.loc[:, 'sex_nom'] = penguin['sex'].apply(lambda sex: 1 if sex == 'M' else 0)

penguin = penguin.drop(['species', 'island', 'sex'], axis=1)

# Fazendo limpeza de dados
final_data_penguin = [col for col in penguin.columns if col.endswith(('_std', '_nom', '_ord')) and col != 'body_mass_g_std'] + ['body_mass_g']
penguin = penguin[final_data_penguin]

print(penguin.head())

# Dividindo os dados em treino e teste
predictors_train, predictors_test, target_train, target_test = train_test_split(
	penguin.drop(['body_mass_g'], axis=1),
	penguin['body_mass_g'],
	test_size=1/3,
	random_state=123 #número qualquer
 )

# Treinando o modelo
model = LinearRegression()
model = model.fit(predictors_train, target_train)

# Avaliando o modelo
target_predicted = model.predict(predictors_test)

rmse = np.sqrt(mean_squared_error(target_test, target_predicted))
print(rmse) #rmse ~= 324.44g

# Testando o modelo
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
print(predicted_body_mass) #predicted_body_mass =~ 3556,53g