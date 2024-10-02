import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Carregar o dataset com valores originais de rating e year
df = pd.read_csv('all_movies_with_clusters.csv')  # Dataset com os clusters
bert_embeddings = np.load('bert_embeddings.npy')  # Embeddings das sinopses
kmeans = load('kmeans_model.joblib')  # Modelo KMeans

# Carregar tokenizer e modelo BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertModel.from_pretrained('bert-base-multilingual-uncased')

# Função para gerar embeddings de sinopse
def get_bert_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output.detach().numpy()

# Função para recomendar filmes com base no cluster e similaridade de embeddings
def recommend_similar_movies(embedding, cluster_number, df, embeddings, num_recommendations=5):
    # Filtrar filmes do mesmo cluster
    cluster_movies = df[df['cluster'] == cluster_number]
    cluster_indices = cluster_movies.index.tolist()
    
    # Filtrar os embeddings dos filmes do mesmo cluster
    cluster_embeddings = embeddings[cluster_indices]
    
    # Calcular a similaridade de cosseno entre o embedding da sinopse do usuário e os embeddings dos filmes
    similarities = cosine_similarity(embedding, cluster_embeddings).flatten()
    
    # Encontrar os índices dos filmes mais similares
    similar_indices = np.argsort(similarities)[-num_recommendations:][::-1]  # Selecionar os mais similares
    
    # Selecionar os filmes correspondentes (usando os valores originais de rating e year)
    recommended_movies = cluster_movies.iloc[similar_indices]
    return recommended_movies[['title_pt', 'rating_original', 'year_original']]

# Função para recomendar filmes com base no cluster
def recommend_movies(cluster_number, df, num_recommendations=5):
    cluster_movies = df[df['cluster'] == cluster_number]
    # Usar os valores originais de rating e year
    recommended_movies = cluster_movies.sort_values(by='rating_original', ascending=False).head(num_recommendations)
    return recommended_movies[['title_pt', 'rating_original', 'year_original']]

# Função para recomendar filmes com base no gênero
def recommend_by_genre(genres, df, num_recommendations=5):
    genre_movies = df[df['genre'].isin(genres)]
    recommended_movies = genre_movies.sort_values(by='rating_original', ascending=False).head(num_recommendations)
    return recommended_movies[['title_pt', 'rating_original', 'year_original']]

# WebApp com Streamlit
st.title("Recomendação de Filmes por Clusterização")

# Opção de escolha de método de recomendação
st.subheader("Escolha um método de recomendação:")
method = st.selectbox("Método de Recomendação", ["Método 1: Escolha de Sinopse", "Método 2: Escreva sua Própria Sinopse (Cluster)", "Método 3: Escreva sua Própria Sinopse (Proximidade)", "Método 4: Escolha por Gênero"])

# Método 1: Escolha de Sinopse
if method == "Método 1: Escolha de Sinopse":
    st.write("Escolha uma sinopse abaixo que mais lhe agrada:")
    
    # Apresentar 5 sinopses aleatórias
    random_movies = df.sample(5)
    sinopse_choices = random_movies['sinopse'].tolist()
    
    sinopse_selecionada = st.radio("Escolha uma sinopse:", sinopse_choices)
    
    if st.button("Recomendar Filmes"):
        # Identificar o cluster da sinopse escolhida
        selected_movie = random_movies[random_movies['sinopse'] == sinopse_selecionada]
        selected_cluster = selected_movie['cluster'].values[0]
        
        # Recomendar filmes do mesmo cluster
        recommendations = recommend_movies(selected_cluster, df)
        
        st.write(f"Filmes recomendados para você (Cluster {selected_cluster}):")
        for index, row in recommendations.iterrows():
            st.write(f"{row['title_pt']} - Rating: {row['rating_original']} - Ano: {row['year_original']}")

# Método 2: Escreva sua Própria Sinopse (Cluster)
if method == "Método 2: Escreva sua Própria Sinopse (Cluster)":
    st.write("Escreva uma sinopse de um filme que você gostaria:")
    
    user_input = st.text_area("Digite sua sinopse aqui")
    
    if st.button("Recomendar Filmes"):
        # Gerar embedding da sinopse digitada
        user_embedding = get_bert_embedding(user_input, model, tokenizer)
        
        # Prever o cluster com base no embedding
        user_cluster = kmeans.predict(user_embedding)
        
        # Recomendar filmes do mesmo cluster
        recommendations = recommend_movies(user_cluster[0], df)
        
        st.write(f"Filmes recomendados para você (Cluster {user_cluster[0]}):")
        for index, row in recommendations.iterrows():
            st.write(f"{row['title_pt']} - Rating: {row['rating_original']} - Ano: {row['year_original']}")

# Método 3: Escreva sua Própria Sinopse (Proximidade de Embeddings)
if method == "Método 3: Escreva sua Própria Sinopse (Proximidade)":
    st.write("Escreva uma sinopse de um filme que você gostaria:")
    
    user_input = st.text_area("Digite sua sinopse aqui")
    
    if st.button("Recomendar Filmes"):
        # Gerar embedding da sinopse digitada
        user_embedding = get_bert_embedding(user_input, model, tokenizer)
        
        # Prever o cluster com base no embedding
        user_cluster = kmeans.predict(user_embedding)
        
        # Recomendar filmes com base na similaridade de embeddings
        recommendations = recommend_similar_movies(user_embedding, user_cluster[0], df, bert_embeddings)
        
        st.write(f"Filmes recomendados para você (Cluster {user_cluster[0]} - Proximidade de Embeddings):")
        for index, row in recommendations.iterrows():
            st.write(f"{row['title_pt']} - Rating: {row['rating_original']} - Ano: {row['year_original']}")

# Método 4: Recomendação por Gênero
if method == "Método 4: Escolha por Gênero":
    st.write("Selecione um ou mais gêneros de filmes:")
    
    # Listar todos os gêneros únicos
    unique_genres = df['genre'].unique().tolist()
    
    # Permitir que o usuário selecione um ou mais gêneros
    selected_genres = st.multiselect("Escolha os gêneros:", unique_genres)
    
    if st.button("Recomendar Filmes"):
        if selected_genres:
            # Recomendar filmes com base nos gêneros selecionados
            recommendations = recommend_by_genre(selected_genres, df)
            
            st.write(f"Filmes recomendados para você nos gêneros: {', '.join(selected_genres)}")
            for index, row in recommendations.iterrows():
                st.write(f"{row['title_pt']} - Rating: {row['rating_original']} - Ano: {row['year_original']}")
        else:
            st.write("Por favor, selecione pelo menos um gênero.")
