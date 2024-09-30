import streamlit as st
import pandas as pd
import random
import joblib
import torch
import numpy as np  # Adicionado aqui
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Carregar dataset atualizado com clusters
movies_df = pd.read_csv(r'all_movies_with_clusters.csv', delimiter=';')
model = joblib.load(r'kmeans_model_2.0.joblib')

# Carregar o modelo PCA
pca_model = joblib.load(r'pca_model.joblib')

# Carregar o modelo BERT e o tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Exibir as colunas disponíveis no dataset
st.write("Colunas disponíveis no dataset:", movies_df.columns)

# Ajustar os nomes das colunas para título e gênero
title_column = 'title_pt'
genre_column = 'genre'

# Função para gerar embeddings com o BERT
def embed_text_with_bert(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Função para aplicar PCA nos embeddings
def apply_pca(embedding):
    return pca_model.transform(embedding)

# Função para recomendar filmes com base na similaridade de cosseno
def recommend_movies_from_cluster_by_similarity(user_vector, cluster, num_recommendations=5):
    cluster_movies = movies_df[movies_df['Cluster'] == cluster]
    if len(cluster_movies) < num_recommendations:
        return cluster_movies
    # Gerar embeddings para as sinopses no cluster
    cluster_embeddings = np.vstack([embed_text_with_bert(text) for text in cluster_movies['sinopse_clean']])
    
    # Calcular a similaridade de cosseno
    similarities = cosine_similarity(user_vector, cluster_embeddings)
    
    # Obter os índices das sinopses mais similares
    similar_movie_indices = similarities.argsort()[0][-num_recommendations:][::-1]
    
    # Retornar os filmes mais similares
    return cluster_movies.iloc[similar_movie_indices]

# Função para processar a sinopse do usuário e identificar o cluster
def get_cluster_from_synopsis(user_synopsis):
    user_vector = embed_text_with_bert(user_synopsis)
    user_vector_pca = apply_pca(user_vector)
    predicted_cluster = model.predict(user_vector_pca)[0]
    return predicted_cluster

# Streamlit App
st.title("Recomendação de Filmes Baseada em Clusterização")

# Tela Inicial - Escolher o método
st.write("Escolha o método para receber recomendações de filmes:")

method = st.selectbox("Selecione o método:", ["Método 1: Escolher Sinopse", "Método 2: Escrever Sinopse"])

if method == "Método 1: Escolher Sinopse":
    st.write("Escolha uma sinopse de filme que lhe agrada para receber recomendações.")

    # Apresentar 3-5 sinopses aleatórias (sem título)
    random_synopses = random.sample(list(movies_df['sinopse']), 5)
    chosen_synopsis = st.radio("Escolha uma sinopse:", random_synopses)

    if st.button("Receber Recomendações"):
        # Identificar o cluster da sinopse escolhida
        chosen_vector = embed_text_with_bert(chosen_synopsis)
        chosen_vector_pca = apply_pca(chosen_vector)
        chosen_cluster = model.predict(chosen_vector_pca)[0]

        # Recomendando 5 filmes com base na similaridade
        recommendations = recommend_movies_from_cluster_by_similarity(chosen_vector, chosen_cluster)
        st.write("Aqui estão 5 filmes recomendados com base na sua escolha:")
        st.write(recommendations[[title_column, genre_column]])

elif method == "Método 2: Escrever Sinopse":
    st.write("Escreva uma sinopse de filme que você gostaria de assistir.")

    user_input = st.text_area("Escreva a sinopse do filme:")

    if st.button("Receber Recomendações"):
        # Identificar o cluster da sinopse escrita
        predicted_cluster = get_cluster_from_synopsis(user_input)
        
        # Recomendando 5 filmes com base na similaridade
        user_vector = embed_text_with_bert(user_input)
        recommendations = recommend_movies_from_cluster_by_similarity(user_vector, predicted_cluster)
        st.write("Aqui estão 5 filmes recomendados com base na sua sinopse:")
        st.write(recommendations[[title_column, genre_column]])
