import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_blobs
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import time

# Configurazione pagina Streamlit
st.set_page_config(
    page_title="🎥 Clustering di Film con K-means",
    page_icon="🎬",
    layout="wide",
)

# Titolo e introduzione
st.title("🎬 Clustering di Film con l'Algoritmo K-means")
st.markdown("""
Questa applicazione visualizza passo-passo il funzionamento dell'algoritmo K-means applicato a diversi dataset,
tra cui uno basato su **film** (genere e valutazione IMDb). Puoi modificare vari parametri e vedere come cambia il comportamento dell'algoritmo.
""")

# Classe KMeansClustering (uguale)
class KMeansClustering:
    def __init__(self, k=3, max_iterations=100, random_state=42):
        self.k = k
        self.max_iterations = max_iterations
        self.random_state = random_state
        np.random.seed(random_state)
        
    def initialize_centroids(self, X):
        indices = np.random.choice(X.shape[0], self.k, replace=False)
        return X[indices]

    def assign_clusters(self, X, centroids):
        distances = np.zeros((X.shape[0], self.k))
        for i, centroid in enumerate(centroids):
            distances[:, i] = np.sqrt(np.sum((X - centroid)**2, axis=1))
        return np.argmin(distances, axis=1)

    def update_centroids(self, X, clusters):
        centroids = np.zeros((self.k, X.shape[1]))
        for i in range(self.k):
            if np.sum(clusters == i) > 0:
                centroids[i] = np.mean(X[clusters == i], axis=0)
        return centroids

    def compute_sse(self, X, clusters, centroids):
        sse = 0
        for i in range(self.k):
            cluster_points = X[clusters == i]
            if len(cluster_points) > 0:
                sse += np.sum((cluster_points - centroids[i])**2)
        return sse

    def fit(self, X, verbose=False):
        centroids = self.initialize_centroids(X)
        all_centroids = [centroids.copy()]
        all_clusters = []
        all_sse = []
        for iteration in range(self.max_iterations):
            clusters = self.assign_clusters(X, centroids)
            all_clusters.append(clusters.copy())
            new_centroids = self.update_centroids(X, clusters)
            sse = self.compute_sse(X, clusters, new_centroids)
            all_sse.append(sse)
            if verbose:
                print(f"Iterazione {iteration+1}, SSE: {sse:.4f}")
            if np.all(centroids == new_centroids):
                break
            centroids = new_centroids
            all_centroids.append(centroids.copy())
        self.centroids = centroids
        self.clusters = clusters
        self.all_centroids = all_centroids
        self.all_clusters = all_clusters
        self.all_sse = all_sse
        self.sse = all_sse[-1]
        self.n_iterations = len(all_centroids) - 1
        return self

# Funzione per preparare i dati Iris (per mantenere opzioni originali)
def prepare_iris_data(features):
    iris = load_iris()
    if features == "sepali":
        X = iris.data[:, :2]
        feature_names = [iris.feature_names[0], iris.feature_names[1]]
    elif features == "petali":
        X = iris.data[:, 2:4]
        feature_names = [iris.feature_names[2], iris.feature_names[3]]
    else:
        X = iris.data
        feature_names = iris.feature_names
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, iris.target, feature_names, iris.target_names

# Dati sintetici
def generate_synthetic_data(n_samples, n_clusters, random_state):
    X, y = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=2, random_state=random_state)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    feature_names = ["Feature 1", "Feature 2"]
    target_names = [f"Cluster {i+1}" for i in range(n_clusters)]
    return X_scaled, y, feature_names, target_names

# Nuova funzione per i dati dei film
def load_movie_data():
    df = pd.read_csv("film_genere_valutazione.csv")
    X = df[["Genere (codificato)", "Valutazione IMDb"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_true = df["Genere (codificato)"].values - 1
    feature_names = ["Genere", "Valutazione IMDb"]
    target_names = ["Action", "Drama", "Thriller", "Musical", "Romantic"]
    return X_scaled, y_true, feature_names, target_names

# Sidebar configurazione
st.sidebar.header("Configurazione")
dataset_option = st.sidebar.selectbox(
    "Seleziona dataset",
    ["Iris - Sepali", "Iris - Petali", "Dati Sintetici", "Film (Genere e IMDb)"]
)
k_value = st.sidebar.slider("Numero di cluster (k)", 2, 10, 3)
max_iterations = st.sidebar.slider("Numero massimo di iterazioni", 10, 100, 20)
random_state = st.sidebar.slider("Seed casuale", 0, 100, 42)
if dataset_option == "Dati Sintetici":
    n_samples = st.sidebar.slider("Numero di campioni", 50, 500, 150)
    n_clusters_true = st.sidebar.slider("Numero di cluster reali", 2, 10, 3)
    random_state_data = st.sidebar.slider("Seed per generazione dati", 0, 100, 42)

# Preparazione dati
dataset_name = ""
if dataset_option == "Iris - Sepali":
    X, y_true, feature_names, target_names = prepare_iris_data("sepali")
    dataset_name = "Iris (Sepali)"
elif dataset_option == "Iris - Petali":
    X, y_true, feature_names, target_names = prepare_iris_data("petali")
    dataset_name = "Iris (Petali)"
elif dataset_option == "Film (Genere e IMDb)":
    X, y_true, feature_names, target_names = load_movie_data()
    dataset_name = "Film (Genere e Valutazione)"
else:
    X, y_true, feature_names, target_names = generate_synthetic_data(n_samples, n_clusters_true, random_state_data)
    dataset_name = "Dati Sintetici"

# Layout principale
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader(f"Dataset: {dataset_name}")
    fig_original, ax = plt.subplots(figsize=(10, 6))
    for i in np.unique(y_true):
        ax.scatter(X[y_true == i, 0], X[y_true == i, 1], label=target_names[i])
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title('Dati originali con etichette vere')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig_original)

    if st.button("Esegui K-means"):
        kmeans = KMeansClustering(k=k_value, max_iterations=max_iterations, random_state=random_state)
        kmeans.fit(X)

        st.subheader("Evoluzione dell'algoritmo K-means")
        iteration_container = st.empty()
        n_iterations_to_show = min(10, kmeans.n_iterations + 1)
        for i in range(n_iterations_to_show):
            fig_iter, ax_iter = plt.subplots(figsize=(10, 6))
            if i > 0:
                for j in range(kmeans.k):
                    cluster_points = X[kmeans.all_clusters[i-1] == j]
                    ax_iter.scatter(cluster_points[:, 0], cluster_points[:, 1], alpha=0.7, label=f'Cluster {j+1}')
            else:
                ax_iter.scatter(X[:, 0], X[:, 1], alpha=0.3, color='gray')
            centroids = kmeans.all_centroids[i]
            ax_iter.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroidi')
            ax_iter.set_title(f"{'Inizializzazione' if i==0 else f'Iterazione {i} - SSE: {kmeans.all_sse[i-1]:.4f}' }")
            ax_iter.set_xlabel(feature_names[0])
            ax_iter.set_ylabel(feature_names[1])
            ax_iter.legend()
            ax_iter.grid(True, linestyle='--', alpha=0.7)
            iteration_container.pyplot(fig_iter)
            time.sleep(0.8)

        st.subheader("Evoluzione dell'errore (SSE)")
        fig_sse, ax_sse = plt.subplots(figsize=(10, 4))
        ax_sse.plot(range(1, len(kmeans.all_sse) + 1), kmeans.all_sse, marker='o')
        ax_sse.set_xlabel('Iterazione')
        ax_sse.set_ylabel('Somma degli errori quadratici (SSE)')
        ax_sse.set_title('Convergenza del K-means')
        ax_sse.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig_sse)

        st.subheader("Risultato finale del clustering")
        fig_final, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        for j in range(kmeans.k):
            cluster_points = X[kmeans.clusters == j]
            ax1.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {j+1}')
        ax1.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], s=200, c='red', marker='X', label='Centroidi')
        ax1.set_xlabel(feature_names[0])
        ax1.set_ylabel(feature_names[1])
        ax1.set_title('Risultato K-means')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        for i in np.unique(y_true):
            ax2.scatter(X[y_true == i, 0], X[y_true == i, 1], label=target_names[i])
        ax2.set_xlabel(feature_names[0])
        ax2.set_ylabel(feature_names[1])
        ax2.set_title('Classi reali')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig_final)

with col2:
    st.header("Come funziona K-means?")
    st.markdown("""
    K-means è un algoritmo di *clustering* non supervisionato che raggruppa i dati in **K gruppi** simili, minimizzando la distanza tra i punti e i loro centroidi.

    **Fasi principali:**
    1. Inizializza K centroidi casuali
    2. Assegna ogni punto al centroide più vicino
    3. Aggiorna i centroidi come media dei punti assegnati
    4. Ripeti fino alla convergenza

    **Applicazioni:**
    - Raccomandazioni di film
    - Segmentazione utenti
    - Analisi musicale e sociale
    """)
