import streamlit as st
import pandas as pd
import joblib
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Carregar dataset atualizado com clusters
movies_df = pd.read_csv('all_movies_with_clusters.csv', delimiter=';')

# Carregar os modelos KMeans e PCA
kmeans_model = joblib.load('kmeans_model_2.0.joblib')
pca_model = joblib.load('pca_model.joblib')

# Carregar o modelo BERT e o tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Função para tokenizar e gerar embeddings com BERT
def embed_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Função para recomendar um filme com base na sinopse fornecida
def recommend_movie(new_synopsis, df, tokenizer, model, pca, kmeans):
    # Gerar embedding para a nova sinopse
    new_embedding = embed_text(new_synopsis, tokenizer, model)
    new_embedding_pca = pca.transform(new_embedding)  # Aplicar PCA na nova sinopse

    # Aplicar PCA nos embeddings dos filmes
    movie_embeddings_pca = np.vstack(df['embeddings'].apply(eval).values)  # Transformando a string de lista em array
    movie_embeddings_pca = pca.transform(movie_embeddings_pca)  # Aplicar PCA nos filmes

    # Identificar o cluster mais próximo para a nova sinopse
    cluster = kmeans.predict(new_embedding_pca)

    # Filtrar filmes no mesmo cluster
    same_cluster_movies = df[df['Cluster'] == cluster[0]]
    
    # Calcular similaridade por cosseno
    cosine_similarities = cosine_similarity(new_embedding_pca, movie_embeddings_pca[same_cluster_movies.index])
    most_similar_idx = np.argmax(cosine_similarities)
    
    # Retornar o filme mais similar
    recommended_movie = same_cluster_movies.iloc[most_similar_idx]['title_pt']
    return recommended_movie

# Interface do Streamlit
st.title("Recomendador de Filmes Baseado em Sinopse")

# Exibir as colunas disponíveis no dataset
st.write("Colunas disponíveis no dataset:", movies_df.columns)

# Input do usuário
new_synopsis = st.text_area("Digite a sinopse do filme:")

# Botão para gerar a recomendação
if st.button("Recomendar Filme"):
    if new_synopsis:
        recommended_movie = recommend_movie(new_synopsis, movies_df, tokenizer, bert_model, pca_model, kmeans_model)
        st.write(f"Filme recomendado: **{recommended_movie}**")
    else:
        st.write("Por favor, insira uma sinopse.")
