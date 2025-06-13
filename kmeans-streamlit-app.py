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
    page_title="K-means Clustering Didattico",
    page_icon="🌸",
    layout="wide",
)

# Titolo e introduzione
st.title("🌸 Apprendimento didattico dell'algoritmo K-means")
st.markdown("""
Questa applicazione visualizza passo-passo il funzionamento dell'algoritmo K-means applicato al dataset Iris
o a dati sintetici generati casualmente. Puoi modificare vari parametri e vedere come cambia il comportamento dell'algoritmo.
""")

# Implementazione dell'algoritmo K-means
class KMeansClustering:
    def __init__(self, k=3, max_iterations=100, random_state=42):
        self.k = k
        self.max_iterations = max_iterations
        self.random_state = random_state
        np.random.seed(random_state)
        
    def initialize_centroids(self, X):
        """Inizializza i centroidi selezionando k punti casuali dal dataset"""
        indices = np.random.choice(X.shape[0], self.k, replace=False)
        return X[indices]
    
    def assign_clusters(self, X, centroids):
        """Assegna ogni punto al cluster più vicino"""
        distances = np.zeros((X.shape[0], self.k))
        
        for i, centroid in enumerate(centroids):
            # Calcolo la distanza euclidea tra ogni punto e il centroide
            distances[:, i] = np.sqrt(np.sum((X - centroid)**2, axis=1))
        
        # Assegno ogni punto al cluster del centroide più vicino
        return np.argmin(distances, axis=1)
    
    def update_centroids(self, X, clusters):
        """Aggiorna i centroidi calcolando la media dei punti in ogni cluster"""
        centroids = np.zeros((self.k, X.shape[1]))
        
        for i in range(self.k):
            # Se il cluster è vuoto, manteniamo il centroide precedente
            if np.sum(clusters == i) > 0:
                centroids[i] = np.mean(X[clusters == i], axis=0)
        
        return centroids
    
    def compute_sse(self, X, clusters, centroids):
        """Calcola la somma degli errori quadratici (SSE)"""
        sse = 0
        for i in range(self.k):
            cluster_points = X[clusters == i]
            if len(cluster_points) > 0:
                sse += np.sum((cluster_points - centroids[i])**2)
        return sse
    
    def fit(self, X, verbose=False):
        """Esegue l'algoritmo K-means e visualizza ogni iterazione se richiesto"""
        # Inizializzazione dei centroidi
        centroids = self.initialize_centroids(X)
        
        all_centroids = [centroids.copy()]  # Memorizzo tutti i centroidi per visualizzarli
        all_clusters = []  # Memorizzo tutti i cluster per visualizzarli
        all_sse = []  # Memorizzo tutti gli SSE per visualizzarli
        
        # Eseguo l'algoritmo K-means
        for iteration in range(self.max_iterations):
            # Assegno i cluster
            clusters = self.assign_clusters(X, centroids)
            all_clusters.append(clusters.copy())
            
            # Aggiorno i centroidi
            new_centroids = self.update_centroids(X, clusters)
            
            # Calcolo SSE
            sse = self.compute_sse(X, clusters, new_centroids)
            all_sse.append(sse)
            
            if verbose:
                print(f"Iterazione {iteration+1}, SSE: {sse:.4f}")
            
            # Controllo convergenza
            if np.all(centroids == new_centroids):
                if verbose:
                    print(f"Convergenza raggiunta dopo {iteration+1} iterazioni!")
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

# Funzione per preparare il dataset Iris
def prepare_iris_data(features):
    iris = load_iris()
    if features == "sepali":
        X = iris.data[:, :2]  # Lunghezza e larghezza del sepalo
        feature_names = [iris.feature_names[0], iris.feature_names[1]]
    elif features == "petali":
        X = iris.data[:, 2:4]  # Lunghezza e larghezza del petalo
        feature_names = [iris.feature_names[2], iris.feature_names[3]]
    else:  # Tutte le feature
        X = iris.data
        feature_names = iris.feature_names
    
    # Standardizziamo i dati
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, iris.target, feature_names, iris.target_names

# Funzione per generare dati sintetici
def generate_synthetic_data(n_samples, n_clusters, random_state):
    X, y = make_blobs(
        n_samples=n_samples,
        centers=n_clusters,
        n_features=2,
        random_state=random_state
    )
    
    # Standardizziamo i dati
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    feature_names = ["Feature 1", "Feature 2"]
    target_names = [f"Cluster {i+1}" for i in range(n_clusters)]
    
    return X_scaled, y, feature_names, target_names

# Sidebar per la configurazione
st.sidebar.header("Configurazione")

# Selezione del dataset
dataset_option = st.sidebar.selectbox(
    "Seleziona dataset",
    ["Iris - Sepali", "Iris - Petali", "Dati Sintetici"]
)

# Configurazione K-means
k_value = st.sidebar.slider("Numero di cluster (k)", 2, 10, 3)
max_iterations = st.sidebar.slider("Numero massimo di iterazioni", 10, 100, 20)
random_state = st.sidebar.slider("Seed casuale", 0, 100, 42)

# Parametri per dati sintetici
if dataset_option == "Dati Sintetici":
    n_samples = st.sidebar.slider("Numero di campioni", 50, 500, 150)
    n_clusters_true = st.sidebar.slider("Numero di cluster reali", 2, 10, 3)
    random_state_data = st.sidebar.slider("Seed per generazione dati", 0, 100, 42)

# Preparazione dei dati
if dataset_option == "Iris - Sepali":
    X, y_true, feature_names, target_names = prepare_iris_data("sepali")
    dataset_name = "Iris (Sepali)"
elif dataset_option == "Iris - Petali":
    X, y_true, feature_names, target_names = prepare_iris_data("petali")
    dataset_name = "Iris (Petali)"
else:  # Dati sintetici
    X, y_true, feature_names, target_names = generate_synthetic_data(
        n_samples, n_clusters_true, random_state_data
    )
    dataset_name = "Dati Sintetici"

# Layout principale
col1, col2 = st.columns([2, 1])

# Colonna principale per le visualizzazioni
with col1:
    # Visualizzazione dei dati originali
    st.subheader(f"Dataset: {dataset_name}")
    
    fig_original, ax = plt.subplots(figsize=(10, 6))
    
    if dataset_option.startswith("Iris"):
        for i, target_name in enumerate(target_names):
            ax.scatter(X[y_true == i, 0], X[y_true == i, 1], label=f'{target_name}')
    else:
        for i in range(n_clusters_true):
            ax.scatter(X[y_true == i, 0], X[y_true == i, 1], label=f'Cluster {i+1}')
    
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title('Dati originali con etichette vere')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    st.pyplot(fig_original)
    
    # Avvia il clustering
    if st.button("Esegui K-means"):
        # Istanziamo e addestriamo il modello K-means
        kmeans = KMeansClustering(k=k_value, max_iterations=max_iterations, random_state=random_state)
        kmeans.fit(X, verbose=False)
        
        # Visualizziamo le iterazioni una per una
        st.subheader("Evoluzione dell'algoritmo K-means")
        
        # Container per la visualizzazione animata
        iteration_container = st.empty()
        
        # Mostriamo le prime 10 iterazioni o tutte se sono meno di 10
        n_iterations_to_show = min(10, kmeans.n_iterations + 1)
        
        for i in range(n_iterations_to_show):
            fig_iter, ax_iter = plt.subplots(figsize=(10, 6))
            
            # Plottiamo i punti dati
            if i > 0:  # Dopo l'inizializzazione
                for j in range(kmeans.k):
                    cluster_points = X[kmeans.all_clusters[i-1] == j]
                    ax_iter.scatter(cluster_points[:, 0], cluster_points[:, 1], alpha=0.7, 
                                label=f'Cluster {j+1}')
            else:  # Solo per l'inizializzazione
                ax_iter.scatter(X[:, 0], X[:, 1], alpha=0.3, color='gray')
            
            # Plottiamo i centroidi
            centroids = kmeans.all_centroids[i]
            ax_iter.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', 
                          label='Centroidi')
            
            if i == 0:
                ax_iter.set_title('Inizializzazione dei centroidi')
            else:
                sse = kmeans.all_sse[i-1]
                ax_iter.set_title(f'Iterazione {i} - SSE: {sse:.4f}')
            
            ax_iter.set_xlabel(feature_names[0])
            ax_iter.set_ylabel(feature_names[1])
            ax_iter.legend()
            ax_iter.grid(True, linestyle='--', alpha=0.7)
            
            # Aggiorniamo la visualizzazione
            iteration_container.pyplot(fig_iter)
            time.sleep(0.8)  # Pausa per rendere visibile l'animazione
        
        # Visualizziamo l'evoluzione del SSE
        st.subheader("Evoluzione dell'errore (SSE)")
        
        fig_sse, ax_sse = plt.subplots(figsize=(10, 4))
        ax_sse.plot(range(1, len(kmeans.all_sse) + 1), kmeans.all_sse, marker='o')
        ax_sse.set_xlabel('Iterazione')
        ax_sse.set_ylabel('Somma degli errori quadratici (SSE)')
        ax_sse.set_title('Convergenza del K-means')
        ax_sse.grid(True, linestyle='--', alpha=0.7)
        
        st.pyplot(fig_sse)
        
        # Visualizziamo il risultato finale
        st.subheader("Risultato finale del clustering")
        
        fig_final, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # K-means clustering
        for j in range(kmeans.k):
            cluster_points = X[kmeans.clusters == j]
            ax1.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {j+1}')
        ax1.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], s=200, c='red', marker='X', label='Centroidi')
        ax1.set_xlabel(feature_names[0])
        ax1.set_ylabel(feature_names[1])
        ax1.set_title('Risultato K-means')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Classi reali
        if dataset_option.startswith("Iris"):
            for i, target_name in enumerate(target_names):
                ax2.scatter(X[y_true == i, 0], X[y_true == i, 1], label=f'{target_name}')
        else:
            for i in range(n_clusters_true):
                ax2.scatter(X[y_true == i, 0], X[y_true == i, 1], label=f'Cluster reale {i+1}')
        ax2.set_xlabel(feature_names[0])
        ax2.set_ylabel(feature_names[1])
        ax2.set_title('Classi reali')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        st.pyplot(fig_final)
        
        # Matrice di confusione per dataset Iris
        if dataset_option.startswith("Iris"):
            st.subheader("Valutazione del clustering")
            
            confusion_matrix = np.zeros((kmeans.k, len(target_names)), dtype=int)
            
            # Mappiamo i cluster alle classi reali
            for i in range(kmeans.k):
                for j in range(len(target_names)):
                    confusion_matrix[i, j] = np.sum((kmeans.clusters == i) & (y_true == j))
            
            # Visualizziamo la matrice di confusione
            fig_conf, ax_conf = plt.subplots(figsize=(10, 8))
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=target_names, yticklabels=[f'Cluster {i+1}' for i in range(kmeans.k)],
                        ax=ax_conf)
            ax_conf.set_xlabel('Classi reali')
            ax_conf.set_ylabel('Cluster K-means')
            ax_conf.set_title('Matrice di confusione tra cluster K-means e classi reali')
            
            st.pyplot(fig_conf)
            
            # Calcolo dell'accuratezza
            correct_classifications = 0
            for i in range(kmeans.k):
                correct_classifications += np.max(confusion_matrix[i])
            accuracy = correct_classifications / len(y_true)
            
            st.metric("Accuratezza del clustering", f"{accuracy:.2%}")

# Colonna laterale per informazioni didattiche
with col2:
    st.header("Apprendimento")
    
    st.subheader("Cos'è K-means?")
    st.markdown("""
    K-means è un algoritmo di clustering non supervisionato che raggruppa i dati in K cluster distinti 
    basandosi sulla loro similarità. L'algoritmo minimizza la somma delle distanze quadratiche tra i punti 
    e il centroide del loro cluster.
    """)
    
    st.subheader("Fasi dell'algoritmo")
    st.markdown("""
    1. **Inizializzazione**: Scegli casualmente K punti come centroidi iniziali
    2. **Assegnazione**: Assegna ogni punto al centroide più vicino
    3. **Aggiornamento**: Ricalcola i centroidi come media dei punti in ogni cluster
    4. **Ripetizione**: Ripeti i passi 2-3 fino alla convergenza
    """)
    
    st.subheader("Parametri chiave")
    st.markdown("""
    - **K**: Numero di cluster da identificare
    - **Inizializzazione**: Metodo per scegliere i centroidi iniziali
    - **Metrica di distanza**: Solitamente distanza euclidea
    - **Criteri di convergenza**: Quando fermare l'algoritmo
    """)
    
    st.subheader("Vantaggi e limiti")
    st.markdown("""
    **Vantaggi**:
    - Semplice da implementare
    - Scalabile ed efficiente
    - Funziona bene con cluster sferici
    
    **Limiti**:
    - Richiede di specificare K in anticipo
    - Sensibile all'inizializzazione dei centroidi
    - Funziona male con cluster di forma non sferica
    - Può convergere a minimi locali
    """)
    
    st.subheader("Applicazioni")
    st.markdown("""
    - Segmentazione clienti
    - Compressione di immagini
    - Analisi di documenti
    - Rilevamento di anomalie
    - Classificazione di dati biologici (come nel dataset Iris)
    """)

# Footer
st.markdown("""
---
### Come utilizzare questa applicazione

1. Scegli il dataset e configura i parametri nella barra laterale
2. Clicca su "Esegui K-means" per avviare l'algoritmo
3. Osserva le diverse iterazioni e come i centroidi si spostano
4. Analizza il risultato finale e confrontalo con le classi reali

L'applicazione mostra visivamente come l'algoritmo K-means converge verso una soluzione stabile, 
evidenziando i cluster formati ad ogni iterazione.
""")
