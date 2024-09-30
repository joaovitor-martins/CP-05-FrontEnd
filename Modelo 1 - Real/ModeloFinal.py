nltk.download('stopwords')
from transformers                     import BertTokenizer, BertForSequenceClassification, AdamW, BertModel
from yellowbrick.cluster              import KElbowVisualizer, SilhouetteVisualizer
from sklearn.preprocessing            import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text  import TfidfVectorizer
from sklearn.metrics                  import silhouette_score
from sklearn.cluster                  import KMeans
from pycaret.clustering               import *
from nltk.corpus                      import stopwords

import matplotlib.pyplot  as plt
import seaborn            as sns
import pandas             as pd
import numpy              as np
import torch
import joblib
import nltk

### Analise exploratoria
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

# Salvando o DataFrame atualizado com a coluna 'Cluster'
df.to_csv('all_movies_with_clusters.csv', sep=';', index=False)

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