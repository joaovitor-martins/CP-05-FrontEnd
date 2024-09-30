"""#CP 05 - Cluster

Inicio do CP de Front End
"""

from transformers                     import BertTokenizer, BertForSequenceClassification, AdamW, BertModel
from yellowbrick.cluster              import KElbowVisualizer, SilhouetteVisualizer
from sklearn.preprocessing            import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text  import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics                  import silhouette_score
from sklearn.cluster                  import KMeans
from pycaret.clustering               import *

import matplotlib.pyplot  as plt
import seaborn            as sns
import pandas             as pd
import numpy              as np
import torch
import joblib

from transformers                     import BertTokenizer, BertForSequenceClassification, AdamW
from yellowbrick.cluster              import KElbowVisualizer
from sklearn.feature_extraction.text  import ENGLISH_STOP_WORDS
from sklearn.cluster                  import KMeans
import matplotlib.pyplot  as plt
import seaborn            as sns
import pandas             as pd
import numpy              as np
import torch
import joblib

"""## Modelo 1 - PyCaret + TF-IDF

### Analise exploratoria
"""

df = pd.read_csv('all_movies.csv', sep=';')

print("Dimensões do dataset:", df.shape)
print("Colunas do dataset:", df.columns)

# Estatísticas descritivas
print(df.describe())

# Visualização da distribuição de gêneros
plt.figure(figsize=(10,6))
sns.countplot(y=df['genre'], order=df['genre'].value_counts().index)
plt.title("Distribuição de Gêneros")
plt.show()

"""### Tratamento dos dados"""

# Removendo valores nulos
df.dropna(inplace=True)

# Removendo Stopwords das sinopses
df['sinopse_clean'] = df['sinopse'].apply(lambda x: ' '.join([word for word in x.split() if word not in ENGLISH_STOP_WORDS]))

# Encoding para os títulos
encoder = LabelEncoder()
df['title_pt_encoded'] = encoder.fit_transform(df['title_pt'])
df['title_en_encoded'] = encoder.fit_transform(df['title_en'])

# TF-IDF para sinopse
tfidf = TfidfVectorizer(max_features=100)  # Limitando a 100 features para simplificação
sinopse_tfidf = tfidf.fit_transform(df['sinopse_clean']).toarray()

# Adicionando as features TF-IDF ao DataFrame
tfidf_columns = [f'tfidf_{i}' for i in range(sinopse_tfidf.shape[1])]
df_tfidf = pd.DataFrame(sinopse_tfidf, columns=tfidf_columns)
df = pd.concat([df, df_tfidf], axis=1)

# Normalizando os dados numéricos
df[['year', 'rating']] = MinMaxScaler().fit_transform(df[['year', 'rating']])

# Garantindo que todas as colunas numéricas sejam numéricas
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')  # Converte para numérico, tratando erros

"""### Modelo"""

# Configurando o PyCaret com as colunas necessárias, ignorando as colunas de texto originais
exp_clustering = setup(data=df, normalize=True, ignore_features=['title_pt', 'title_en', 'sinopse', 'sinopse_clean', 'genre'])

# Criando o modelo de KMeans
kmeans_model = create_model('kmeans')

# Ajuste de número de clusters com plot_model
plot_model(kmeans_model, plot='elbow')

"""### Avaliação"""

# Métricas de avaliação
evaluate_model(kmeans_model)

# Prevendo os clusters (garantindo que todas as colunas numéricas estejam corretamente convertidas)
df['Cluster'] = predict_model(kmeans_model, data=df[numeric_columns])['Cluster']

# Exibindo os clusters
df.head()

# Visualização dos clusters - Contagem de filmes por cluster
plt.figure(figsize=(10,6))
sns.countplot(x='Cluster', data=df)
plt.title("Distribuição de Filmes por Cluster")
plt.show()

# Agrupando os dados por Cluster e Gênero
cluster_genre_counts = df.groupby(['Cluster', 'genre']).size().reset_index(name='count')

# Criando o gráfico de barras
plt.figure(figsize=(12,8))
sns.barplot(x='Cluster', y='count', hue='genre', data=cluster_genre_counts)
plt.title("Quantidade de Filmes por Gênero em Cada Cluster")
plt.xlabel("Cluster")
plt.ylabel("Quantidade de Filmes")
plt.legend(title="Gênero", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

"""## Modelo 2 - SKLearn + TF-IDF

### Analise exploratoria
"""

df = pd.read_csv('all_movies.csv', sep=';')

print("Dimensões do dataset:", df.shape)
print("Colunas do dataset:", df.columns)

# Estatísticas descritivas
print(df.describe())

# Visualização da distribuição de gêneros
plt.figure(figsize=(10,6))
sns.countplot(y=df['genre'], order=df['genre'].value_counts().index)
plt.title("Distribuição de Gêneros")
plt.show()

"""### Tratamento dos dados"""

# Removendo valores nulos
df.dropna(inplace=True)

# Removendo Stopwords das sinopses
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
df['sinopse_clean'] = df['sinopse'].apply(lambda x: ' '.join([word for word in x.split() if word not in ENGLISH_STOP_WORDS]))

# Embedding Encoding para a coluna 'genre'
encoder = LabelEncoder()
df['genre_encoded'] = encoder.fit_transform(df['genre'])

# TF-IDF para vetorização das sinopses com parâmetros ajustados
tfidf = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=5, ngram_range=(1,2))
X = tfidf.fit_transform(df['sinopse_clean'])

# Normalizando os dados numéricos
scaler = MinMaxScaler()
df[['year', 'rating']] = scaler.fit_transform(df[['year', 'rating']])

"""### Modelo"""

kmeans = KMeans()

# KElbowVisualizer para encontrar o número ideal de clusters
visualizer = KElbowVisualizer(kmeans, k=(2,20))
visualizer.fit(X.toarray())  # Corrigido: converteu o sparse matrix para numpy array
visualizer.show()

# Ajustando o KMeans com o número ideal de clusters
kmeans = KMeans(n_clusters=visualizer.elbow_value_, init='k-means++', max_iter=500, n_init=50)
clusters = kmeans.fit_predict(X)

"""### Avaliação"""

#Silhouette Score
silhouette_avg = silhouette_score(X, clusters)
print(f"Silhouette Score: {silhouette_avg}")

# Visualização Silhouette
visualizer_sil = SilhouetteVisualizer(kmeans)
visualizer_sil.fit(X)
visualizer_sil.show()

# Atribuindo os clusters ao DataFrame
df['Cluster'] = clusters

# Exibindo os primeiros 5 resultados com seus clusters
df.head()

# Visualização dos clusters - Contagem de filmes por cluster
plt.figure(figsize=(10,6))
sns.countplot(x='Cluster', data=df)
plt.title("Distribuição de Filmes por Cluster")
plt.show()

# Agrupando os dados por Cluster e Gênero
cluster_genre_counts = df.groupby(['Cluster', 'genre']).size().reset_index(name='count')

# Criando o gráfico de barras
plt.figure(figsize=(12,8))
sns.barplot(x='Cluster', y='count', hue='genre', data=cluster_genre_counts)
plt.title("Quantidade de Filmes por Gênero em Cada Cluster")
plt.xlabel("Cluster")
plt.ylabel("Quantidade de Filmes")
plt.legend(title="Gênero", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

"""## Modelo 3 - SKLearn + BERT

### Analise exploratoria
"""

df = pd.read_csv('all_movies.csv', sep=';')

print("Dimensões do dataset:", df.shape)
print("Colunas do dataset:", df.columns)

# Estatísticas descritivas
print(df.describe())

# Visualização da distribuição de gêneros
plt.figure(figsize=(10,6))
sns.countplot(y=df['genre'], order=df['genre'].value_counts().index)
plt.title("Distribuição de Gêneros")
plt.show()

"""### Tratamento dos dados"""

# Removendo valores nulos
df.dropna(inplace=True)

# Removendo Stopwords das sinopses
df['sinopse_clean'] = df['sinopse'].apply(lambda x: ' '.join([word for word in x.split() if word not in set(stopwords.words('portuguese'))]))

# Encoding para os títulos
encoder = LabelEncoder()
df['title_pt_encoded'] = encoder.fit_transform(df['title_pt'])

"""### Modelo 3 - SKLearn + BERT"""

# Tokenização com BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def fine_tune_bert(texts, labels):
    model.train()
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Exemplo de treinamento sem o argumento 'labels'
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)  # Sem labels
        loss = outputs.last_hidden_state.mean()  # Usando uma métrica de exemplo para cálculo de perda
        loss.backward()
        optimizer.step()
    return model

# Fine-tuning com algumas sinopses como exemplo (processo pode ser mais longo)
texts = df['sinopse_clean'].values[:100]
labels = np.random.randint(0, 3, size=(100,))  # Apenas para fins de exemplo
fine_tune_bert(texts, labels)

# Vetorizando as sinopses com BERT após fine-tuning
def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()  # Tirando a média da última camada oculta para cada frase

embeddings = np.vstack([embed_text(text) for text in df['sinopse_clean']])

"""### Modelo"""

# Modelo de cluster usando KMeans do SKLearn
kmeans = KMeans()

# KElbowVisualizer para encontrar o número ideal de clusters
visualizer = KElbowVisualizer(kmeans, k=(2,20))
visualizer.fit(embeddings)
visualizer.show()

# Ajustando o KMeans com o número ideal de clusters
kmeans = KMeans(n_clusters=visualizer.elbow_value_, init='k-means++', max_iter=500, n_init=50)
clusters = kmeans.fit_predict(embeddings)

joblib.dump(kmeans, 'kmeans_model_2.0.joblib')

"""### Avaliação"""

# Atribuindo os clusters ao DataFrame
df['Cluster'] = clusters

# Exibindo os primeiros 5 resultados com seus clusters
df.head()

# Visualização dos clusters - Contagem de filmes por cluster
plt.figure(figsize=(10,6))
sns.countplot(x='Cluster', data=df)
plt.title("Distribuição de Filmes por Cluster")
plt.show()

# Agrupando os dados por Cluster e Gênero
cluster_genre_counts = df.groupby(['Cluster', 'genre']).size().reset_index(name='count')

# Criando o gráfico de barras
plt.figure(figsize=(12,8))
sns.barplot(x='Cluster', y='count', hue='genre', data=cluster_genre_counts)
plt.title("Quantidade de Filmes por Gênero em Cada Cluster")
plt.xlabel("Cluster")
plt.ylabel("Quantidade de Filmes")
plt.legend(title="Gênero", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


