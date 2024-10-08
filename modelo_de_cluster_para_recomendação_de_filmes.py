# -*- coding: utf-8 -*-
"""Modelo de cluster para recomendação de filmes.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/16WJzkyfZHxKVY21Y8pftAI167jaCamxD

# CP 05 - Front End
Recomendação de filmes (**Clusterização**)

## Importação de bibliotecas e do dataset
"""

import pandas as pd
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
import nltk
from joblib import dump
import warnings

# Baixar stopwords em português da NLTK (caso não tenha baixado ainda)
nltk.download('stopwords')
stopwords_pt = stopwords.words('portuguese')

warnings.filterwarnings('ignore')

# Carregar o dataset
file_path = '/content/all_movies.csv'  # Substituir pelo caminho correto
df = pd.read_csv(file_path, delimiter=';')

"""## Análise exploratória (EDA)"""

# Visualizar as primeiras linhas do dataset
df.head()

# Estatísticas descritivas
df.describe()

"""### Distribuição de Ratings"""

plt.figure(figsize=(10, 6))
sns.histplot(df['rating'], kde=True, bins=20, color='blue')
plt.title('Distribuição das Avaliações (Ratings)')
plt.xlabel('Avaliação')
plt.ylabel('Frequência')
plt.show()

"""### Distribuição do Ano de Lançamento"""

plt.figure(figsize=(10, 6))
sns.histplot(df['year'], kde=True, bins=30, color='orange')
plt.title('Distribuição dos Anos de Lançamento')
plt.xlabel('Ano de Lançamento')
plt.ylabel('Frequência')
plt.show()

"""### Qtd Filmes por Gênero"""

plt.figure(figsize=(10, 6))
sns.countplot(y='genre', data=df, order=df['genre'].value_counts().index, palette='coolwarm')
plt.title('Qtd Filmes por Gênero')
plt.xlabel('Frequência')
plt.ylabel('Gênero')
plt.show()

"""### Qtd filmes por ano"""

# Criar uma nova coluna 'year_grouped' que agrupa os anos a cada 5 anos
df['year_grouped'] = (df['year'] // 5) * 5

plt.figure(figsize=(10, 6))
sns.countplot(x='year_grouped', data=df, palette='Blues')
plt.title('Contagem de Filmes por Ano de Lançamento (Agrupados a Cada 5 Anos)')
plt.xlabel('Ano (Agrupado a Cada 5 Anos)')
plt.ylabel('Quantidade de Filmes')
plt.xticks(rotation=45)
plt.show()

"""### Gráfico de Gênero e Avaliação"""

plt.figure(figsize=(12, 8))
sns.violinplot(x='genre', y='rating', data=df, palette='coolwarm')
plt.title('Distribuição de Avaliação por Gênero')
plt.xlabel('Gênero')
plt.ylabel('Avaliação')
plt.xticks(rotation=45)
plt.show()

"""### Relação entre Gênero e Ano de Lançamento"""

plt.figure(figsize=(12, 8))
sns.boxplot(x='genre', y='year_grouped', data=df, palette='Set2')
plt.title('Distribuição do Ano de Lançamento por Gênero')
plt.xlabel('Gênero')
plt.ylabel('Ano (Agrupado a Cada 5 Anos)')
plt.xticks(rotation=45)
plt.show()

"""### Heatmap para correlação entre as variáveis numéricas"""

plt.figure(figsize=(10, 6))

# Selecionar apenas as colunas numéricas, excluindo 'year_grouped'
numeric_columns = df.select_dtypes(include=['float64', 'int64']).drop(columns=['year_grouped'])

# Gerar o heatmap com as colunas numéricas restantes
sns.heatmap(numeric_columns.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlação entre Variáveis Numéricas (Sem year_grouped)')
plt.show()

"""## Tratamento dos dados"""

df = df.drop(columns=['year_grouped'])

"""### Normalização"""

scaler = StandardScaler()
df[['rating', 'year']] = scaler.fit_transform(df[['rating', 'year']])

"""### Embeddings"""

def get_bert_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output.detach().numpy()

# Carregar o modelo BERT e o tokenizer para bert-base-multilingual-uncased
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertModel.from_pretrained('bert-base-multilingual-uncased')

# Gerar embeddings para todas as sinopses
bert_embeddings = np.vstack([get_bert_embedding(sinopse, model, tokenizer) for sinopse in df['sinopse']])

"""## Modelo de clusterização

### Escolha da quantidade de cluster (Metodo do cotovelo)
"""

def plot_elbow_method(embeddings, max_clusters=10):
    sse = []
    for k in range(1, max_clusters+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(embeddings)
        sse.append(kmeans.inertia_)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters+1), sse, marker='o', linestyle='--', color='b')
    plt.title('Método do Cotovelo - Inércia por Quantidade de Clusters')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Inércia')
    plt.show()

plot_elbow_method(bert_embeddings)

"""### Modelo"""

# Escolher 5 clusters com base no gráfico do cotovelo
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(bert_embeddings)

"""## Avaliação do modelo

### Score
"""

silhouette_avg = silhouette_score(bert_embeddings, df['cluster'])
db_index = davies_bouldin_score(bert_embeddings, df['cluster'])

print(f'Silhouette Score: {silhouette_avg}')
print(f'Davies-Bouldin Index: {db_index}')

"""### Gráfico de Sombra/Silhueta"""

def plot_silhouette_analysis(embeddings, cluster_labels):
    silhouette_vals = silhouette_samples(embeddings, cluster_labels)
    y_lower, y_upper = 0, 0
    n_clusters = len(np.unique(cluster_labels))
    plt.figure(figsize=(10, 6))

    for i in range(n_clusters):
        cluster_silhouette_vals = silhouette_vals[cluster_labels == i]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals, alpha=0.7)
        plt.text(-0.05, y_lower + 0.5 * len(cluster_silhouette_vals), str(i))
        y_lower += len(cluster_silhouette_vals)

    plt.axvline(silhouette_avg, color="red", linestyle="--")
    plt.title('Análise Silhouette')
    plt.xlabel('Silhouette Coefficient')
    plt.ylabel('Cluster')
    plt.show()

plot_silhouette_analysis(bert_embeddings, df['cluster'])

"""### Qtd de Filmes por Gênero em cada Clusters"""

plt.figure(figsize=(12, 8))
sns.countplot(x='genre', hue='cluster', data=df, palette='coolwarm')
plt.title('Quantidade de Filmes por Gênero nos Clusters')
plt.xlabel('Gênero')
plt.ylabel('Frequência')
plt.xticks(rotation=45)
plt.show()

"""## Salvando o modelo, dataframe novo e embedding"""

# Salvar o modelo KMeans
dump(kmeans, 'kmeans_model.joblib')

# Salvando o Dataframe
# Buscando a coluna rating e ano originais
file_path = '/content/all_movies.csv'  # Substituir pelo caminho correto
df_original = pd.read_csv(file_path, delimiter=';')

# Adicionando colunas com os valores originais (antes da normalização)
df['rating_original'] = df_original['rating']
df['year_original'] = df_original['year']

# Salvar o DataFrame processado para usar no Streamlit
df.to_csv('all_movies_with_clusters.csv', index=False)

# Salvar os embeddings
np.save('bert_embeddings.npy', bert_embeddings)