import panda as pd

#Lecture du dataset

 Df = pd.read_csv('Credit_data.csv')
 Importer numpy en tant que np

Importer matplotlib.pyplot en tant que plt

Importer Seaborn en tant que SNS

À partir de sklearn.preprocessing import LabelEncoder, StandardScaler

À partir de sklearn.model_selection importer train_test_split

# Charger les données

Df = pd.read_csv('prêt_data.csv')

- Explorer les données

```python

# Vérifiez la forme et les colonnes des données

Imprimer(df.forme)

Imprimer(df.colonnes)

# Vérifiez les cinq premières lignes des données

Imprimer(df.head())

# Vérifiez les types de données et les valeurs manquantes des données

Imprimer(df.info())

# Vérifier les statistiques récapitulatives des variables numériques

Print(df.describe())

# Vérifiez les décomptes de fréquence des variables catégorielles

Print(df['loan_status'].value_counts())

Print(df['Purpose'].value_counts())

Print(df['employment_status'].value_counts())

# Visualiser les données


# Tracer la distribution de la variable cible

Sns.countplot(x='prêt_status', data=df)

Plt.show()

# Tracez la répartition du montant du prêt

Sns.histplot(x='montant_prêt', data=df)

Plt.show()

# Tracer la distribution du taux d'intérêt

Sns.histplot(x='interest_rate', data=df)

Plt.show()

# Tracer la distribution du pointage de crédit

Sns.histplot(x='credit_score', data=df)

Plt.show()

# Tracez les boxplots des variables numériques par la variable cible

Sns.boxplot(x='statut_prêt', y='montant_prêt', data=df)

Plt.show()

Sns.boxplot(x='statut_prêt', y='taux_intérêt', data=df)

Plt.show()

Sns.boxplot(x='loan_status', y='credit_score', data=df)

Plt.show()

# Tracez les comptes des variables catégorielles par la variable cible

Sns.countplot(x='but', hue='loan_status', data=df)

Plt.xticks(rotation=90)

Plt.show()

Sns.countplot(x='employment_status', hue='loan_status', data=df)

Plt.show()

# Tracer la carte thermique de la matrice de corrélation des variables numériques

Sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

Plt.show()

# Effectuer l'ingénierie et la sélection des fonctionnalités

```python

# Créer une nouvelle variable pour le ratio dette/revenu

Df['dti'] = df['mensuel_debt'] / df['mensuel_revenu']

# Transformer la variable de pointage de crédit en variable catégorielle

Df['credit_score_cat'] = pd.cut(df['credit_score'], bins=[0, 580, 670, 740, 800, 850], labels=['Très mauvais', 'Passable', 'Bon', 'Très bien', 'Exceptionnel'])

# Supprimez les variables non pertinentes ou redondantes

Df.drop(['loan_id', 'customer_id', 'credit_score', 'monthly_debt'], axis=1, inplace=True)

# Encodez les variables catégorielles à l'aide de l'encodage d'étiquettes

Le = LabelEncoder()

Df['loan_status'] = le.fit_transform(df['loan_status'])

Df['but'] = le.fit_transform(df['but'])

Df['employment_status'] = le.fit_transform(df['employment_status'])

Df['credit_score_cat'] = le.fit_transform(df['credit_score_cat'])

- Préparer les données pour la modélisation

```python

# Séparez les fonctionnalités et la cible

X = df.drop('loan_status', axis=1)

Y = df['statut_prêt']

# Mettre à l'échelle les caractéristiques numériques à l'aide de la mise à l'échelle standard

Scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Divisez les données en ensembles d'entraînement et de test

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# Charger les données de risque de crédit

Df = pd.read_csv("credit_risk_data.csv")

# Imputer les valeurs manquantes avec zéro

Df = df.fillna(0)
Explicateur = shap.TreeExplainer (modèle)

# Charger les données de risque de crédit

Df = pd.read_csv("credit_risk_data.csv")

# Effectuer un encodage unique sur le sexe et l'état civil

Ohe = OneHotEncoder(sparse=False)

Ohe_df = pd.DataFrame(ohe.fit_transform(df[['gender', 'marital_status']]), columns=ohe.get_feature_names())

# Effectuer un codage ordinal au niveau de l'éducation

Oe = OrdinalEncoder()

Oe_df = pd.DataFrame(oe.fit_transform(df[['education_level']]), colonnes=['education_level'])

# Concaténer les fonctionnalités codées avec les données d'origine

Df = pd.concat([df, ohe_df, oe_df], axis=1)

# Supprimez les colonnes catégorielles d'origine

Df = df.drop(['gender', 'marital_status', 'education_level'], axis=1)

