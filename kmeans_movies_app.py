import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import time

# Configurazione pagina Streamlit
st.set_page_config(
    page_title="K-means Clustering Film",
    page_icon="🎬",
    layout="wide",
)

# Titolo e introduzione
st.title("🎬 Apprendimento didattico dell'algoritmo K-means sui Film")
st.markdown("""
Questa applicazione visualizza passo-passo il funzionamento dell'algoritmo K-means applicato a un dataset di film
basato sui voti degli utenti e sui generi cinematografici. Puoi modificare vari parametri e vedere come cambia il comportamento dell'algoritmo.
""")

# Implementazione dell'algoritmo K-means (identica al codice originale)
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

# Funzione per creare il dataset sui film
def create_movies_dataset(n_movies=200, random_state=42):
    """Crea un dataset sintetico di film con voti e generi"""
    np.random.seed(random_state)
    
    # Definisco i generi cinematografici
    genres = ['Azione', 'Commedia', 'Dramma', 'Horror', 'Sci-Fi', 'Romance', 'Thriller', 'Animazione', 'Documentario', 'Fantasy']
    
    # Genero nomi di film casuali
    movie_titles = [f"Film_{i+1}" for i in range(n_movies)]
    
    # Genero dati per ogni film
    movies_data = []
    
    for i in range(n_movies):
        # Voto medio (scala 1-10) con distribuzione realistica
        if np.random.random() < 0.1:  # 10% film molto scarsi
            rating = np.random.uniform(1, 4)
        elif np.random.random() < 0.7:  # 70% film nella media
            rating = np.random.uniform(5, 7.5)
        else:  # 20% film eccellenti
            rating = np.random.uniform(7.5, 10)
        
        # Numero di recensioni (influenza la popolarità)
        if rating > 7.5:  # Film buoni tendono ad avere più recensioni
            num_reviews = np.random.randint(500, 5000)
        elif rating > 5:
            num_reviews = np.random.randint(100, 1000)
        else:
            num_reviews = np.random.randint(10, 200)
        
        # Genere principale (influenza il voto)
        genre = np.random.choice(genres)
        
        # Alcune correlazioni realistiche genere-voto
        if genre == 'Documentario':
            rating = min(rating + np.random.uniform(0, 1), 10)  # Documentari tendono ad avere voti più alti
        elif genre == 'Horror':
            rating = rating * np.random.uniform(0.8, 1.2)  # Voti più polarizzati
        elif genre == 'Animazione':
            rating = min(rating + np.random.uniform(0, 0.5), 10)  # Leggero bonus
        
        # Anno di uscita (influenza le recensioni)
        year = np.random.randint(1990, 2024)
        if year > 2015:  # Film recenti hanno più recensioni
            num_reviews = int(num_reviews * np.random.uniform(1.2, 2.0))
        
        # Durata in minuti
        if genre == 'Documentario':
            duration = np.random.randint(60, 180)
        elif genre in ['Azione', 'Sci-Fi', 'Fantasy']:
            duration = np.random.randint(90, 180)
        else:
            duration = np.random.randint(80, 150)
        
        movies_data.append({
            'Titolo': movie_titles[i],
            'Voto_Medio': round(rating, 1),
            'Num_Recensioni': num_reviews,
            'Genere': genre,
            'Anno': year,
            'Durata_Min': duration
        })
    
    return pd.DataFrame(movies_data)

# Funzione per preparare i dati per il clustering
def prepare_movie_data(df, features_selection, genre_encoding='numeric'):
    """Prepara i dati per il clustering K-means"""
    
    # Selezione delle feature numeriche base
    if features_selection == "Voto e Recensioni":
        numerical_features = ['Voto_Medio', 'Num_Recensioni']
    elif features_selection == "Voto e Durata":
        numerical_features = ['Voto_Medio', 'Durata_Min']
    elif features_selection == "Recensioni e Durata":
        numerical_features = ['Num_Recensioni', 'Durata_Min']
    elif features_selection == "Voto, Recensioni e Durata":
        numerical_features = ['Voto_Medio', 'Num_Recensioni', 'Durata_Min']
    else:  # Tutte le feature
        numerical_features = ['Voto_Medio', 'Num_Recensioni', 'Anno', 'Durata_Min']
    
    X = df[numerical_features].copy()
    
    # Encoding del genere se richiesto
    if genre_encoding == 'numeric':
        le = LabelEncoder()
        X['Genere_Encoded'] = le.fit_transform(df['Genere'])
        feature_names = numerical_features + ['Genere']
        genre_labels = le.classes_
    else:
        feature_names = numerical_features
        genre_labels = None
    
    # Standardizzazione
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Per la visualizzazione 2D, uso sempre le prime due componenti
    if X_scaled.shape[1] > 2:
        X_2d = X_scaled[:, :2]
        feature_names_2d = feature_names[:2]
    else:
        X_2d = X_scaled
        feature_names_2d = feature_names
    
    return X_2d, X_scaled, feature_names_2d, feature_names, df['Genere'].values, genre_labels

# Funzione per generare cluster artificiali sui film
def generate_movie_clusters(n_movies=200, n_clusters=4, random_state=42):
    """Genera cluster artificiali di film per testare l'algoritmo"""
    np.random.seed(random_state)
    
    # Definisco archetipi di film per ogni cluster
    cluster_archetypes = {
        0: {"name": "Blockbuster", "rating_range": (6.5, 8.5), "reviews_range": (1000, 5000)},
        1: {"name": "Film d'Arte", "rating_range": (7.0, 9.5), "reviews_range": (100, 800)},
        2: {"name": "Film Commerciali", "rating_range": (5.0, 7.0), "reviews_range": (500, 2000)},
        3: {"name": "Film di Nicchia", "rating_range": (6.0, 8.0), "reviews_range": (50, 300)}
    }
    
    movies_data = []
    true_clusters = []
    
    movies_per_cluster = n_movies // n_clusters
    
    for cluster_id in range(n_clusters):
        archetype = cluster_archetypes[cluster_id % len(cluster_archetypes)]
        
        for i in range(movies_per_cluster):
            # Genero caratteristiche basate sull'archetipo
            rating = np.random.uniform(*archetype["rating_range"])
            num_reviews = np.random.randint(*archetype["reviews_range"])
            
            # Aggiungo rumore per rendere più realistico
            rating += np.random.normal(0, 0.3)
            rating = np.clip(rating, 1, 10)
            
            num_reviews = int(num_reviews * np.random.uniform(0.7, 1.3))
            
            movies_data.append({
                'Titolo': f"Film_{cluster_id}_{i+1}",
                'Voto_Medio': round(rating, 1),
                'Num_Recensioni': num_reviews,
                'Genere': np.random.choice(['Azione', 'Commedia', 'Dramma', 'Horror', 'Sci-Fi']),
                'Anno': np.random.randint(1990, 2024),
                'Durata_Min': np.random.randint(80, 180)
            })
            true_clusters.append(cluster_id)
    
    # Aggiungo film rimanenti
    remaining = n_movies - len(movies_data)
    for i in range(remaining):
        cluster_id = np.random.randint(0, n_clusters)
        true_clusters.append(cluster_id)
        
        movies_data.append({
            'Titolo': f"Film_Extra_{i+1}",
            'Voto_Medio': round(np.random.uniform(1, 10), 1),
            'Num_Recensioni': np.random.randint(10, 5000),
            'Genere': np.random.choice(['Azione', 'Commedia', 'Dramma', 'Horror', 'Sci-Fi']),
            'Anno': np.random.randint(1990, 2024),
            'Durata_Min': np.random.randint(80, 180)
        })
    
    return pd.DataFrame(movies_data), np.array(true_clusters), cluster_archetypes

# Sidebar per la configurazione
st.sidebar.header("Configurazione")

# Selezione del tipo di dataset
dataset_type = st.sidebar.selectbox(
    "Tipo di dataset",
    ["Film Casuali", "Film con Cluster Definiti"]
)

# Configurazione del dataset
if dataset_type == "Film Casuali":
    n_movies = st.sidebar.slider("Numero di film", 100, 500, 200)
    random_state_data = st.sidebar.slider("Seed per generazione film", 0, 100, 42)
else:
    n_movies = st.sidebar.slider("Numero di film", 100, 500, 200)
    n_clusters_true = st.sidebar.slider("Numero di cluster reali", 2, 6, 4)
    random_state_data = st.sidebar.slider("Seed per generazione film", 0, 100, 42)

# Selezione delle feature
features_selection = st.sidebar.selectbox(
    "Feature da utilizzare",
    ["Voto e Recensioni", "Voto e Durata", "Recensioni e Durata", "Voto, Recensioni e Durata", "Tutte le feature"]
)

# Inclusione del genere
include_genre = st.sidebar.checkbox("Includi genere cinematografico", value=False)

# Configurazione K-means
k_value = st.sidebar.slider("Numero di cluster (k)", 2, 10, 4)
max_iterations = st.sidebar.slider("Numero massimo di iterazioni", 10, 100, 20)
random_state = st.sidebar.slider("Seed casuale", 0, 100, 42)

# Generazione del dataset
if dataset_type == "Film Casuali":
    movies_df = create_movies_dataset(n_movies, random_state_data)
    y_true = None
    cluster_info = None
else:
    movies_df, y_true, cluster_info = generate_movie_clusters(n_movies, n_clusters_true, random_state_data)

# Preparazione dei dati
X_2d, X_full, feature_names_2d, feature_names_full, genres, genre_labels = prepare_movie_data(
    movies_df, features_selection, 'numeric' if include_genre else 'none'
)

# Layout principale
col1, col2 = st.columns([2, 1])

# Colonna principale per le visualizzazioni
with col1:
    # Visualizzazione del dataset
    st.subheader(f"Dataset Film ({dataset_type})")
    
    # Mostra alcune statistiche del dataset
    st.write("**Statistiche del Dataset:**")
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        st.metric("Numero di Film", len(movies_df))
    with col_stat2:
        st.metric("Voto Medio", f"{movies_df['Voto_Medio'].mean():.1f}")
    with col_stat3:
        st.metric("Generi Unici", movies_df['Genere'].nunique())
    with col_stat4:
        st.metric("Anni Coperti", f"{movies_df['Anno'].max() - movies_df['Anno'].min()}")
    
    # Visualizzazione dei dati originali
    fig_original, ax = plt.subplots(figsize=(10, 6))
    
    if dataset_type == "Film con Cluster Definiti" and y_true is not None:
        for i in range(n_clusters_true):
            cluster_points = X_2d[y_true == i]
            cluster_name = cluster_info[i]["name"] if i in cluster_info else f"Cluster {i+1}"
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=cluster_name, alpha=0.7)
    else:
        # Colora per genere
        unique_genres = movies_df['Genere'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_genres)))
        
        for i, genre in enumerate(unique_genres):
            genre_mask = movies_df['Genere'] == genre
            ax.scatter(X_2d[genre_mask, 0], X_2d[genre_mask, 1], 
                      label=genre, alpha=0.7, color=colors[i])
    
    ax.set_xlabel(feature_names_2d[0])
    ax.set_ylabel(feature_names_2d[1])
    ax.set_title('Dataset Film - Distribuzione Originale')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    st.pyplot(fig_original)
    
    # Mostra un campione del dataset
    st.subheader("Campione del Dataset")
    st.dataframe(movies_df.head(10))
    
    # Avvia il clustering
    if st.button("Esegui K-means sui Film"):
        # Utilizzo X_2d per la visualizzazione, ma X_full per il clustering se disponibile
        X_clustering = X_full if include_genre else X_2d
        
        # Istanziamo e addestriamo il modello K-means
        kmeans = KMeansClustering(k=k_value, max_iterations=max_iterations, random_state=random_state)
        kmeans.fit(X_clustering, verbose=False)
        
        # Visualizziamo le iterazioni una per una
        st.subheader("Evoluzione dell'algoritmo K-means sui Film")
        
        # Container per la visualizzazione animata
        iteration_container = st.empty()
        
        # Mostriamo le prime 10 iterazioni o tutte se sono meno di 10
        n_iterations_to_show = min(10, kmeans.n_iterations + 1)
        
        for i in range(n_iterations_to_show):
            fig_iter, ax_iter = plt.subplots(figsize=(10, 6))
            
            # Plottiamo i punti dati (usando sempre X_2d per la visualizzazione)
            if i > 0:  # Dopo l'inizializzazione
                for j in range(kmeans.k):
                    cluster_points = X_2d[kmeans.all_clusters[i-1] == j]
                    ax_iter.scatter(cluster_points[:, 0], cluster_points[:, 1], alpha=0.7, 
                                label=f'Cluster Film {j+1}')
            else:  # Solo per l'inizializzazione
                ax_iter.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.3, color='gray')
            
            # Plottiamo i centroidi (proiettiamo su 2D se necessario)
            if X_clustering.shape[1] > 2:
                centroids_2d = kmeans.all_centroids[i][:, :2]  # Prime due dimensioni
            else:
                centroids_2d = kmeans.all_centroids[i]
                
            ax_iter.scatter(centroids_2d[:, 0], centroids_2d[:, 1], s=200, c='red', marker='X', 
                          label='Centroidi')
            
            if i == 0:
                ax_iter.set_title('Inizializzazione dei centroidi per Film')
            else:
                sse = kmeans.all_sse[i-1]
                ax_iter.set_title(f'Iterazione {i} - SSE: {sse:.4f}')
            
            ax_iter.set_xlabel(feature_names_2d[0])
            ax_iter.set_ylabel(feature_names_2d[1])
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
        ax_sse.set_title('Convergenza del K-means sui Film')
        ax_sse.grid(True, linestyle='--', alpha=0.7)
        
        st.pyplot(fig_sse)
        
        # Visualizziamo il risultato finale
        st.subheader("Risultato finale del clustering sui Film")
        
        if dataset_type == "Film con Cluster Definiti" and y_true is not None:
            fig_final, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # K-means clustering
            for j in range(kmeans.k):
                cluster_points = X_2d[kmeans.clusters == j]
                ax1.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {j+1}')
            
            if X_clustering.shape[1] > 2:
                centroids_2d = kmeans.centroids[:, :2]
            else:
                centroids_2d = kmeans.centroids
                
            ax1.scatter(centroids_2d[:, 0], centroids_2d[:, 1], s=200, c='red', marker='X', label='Centroidi')
            ax1.set_xlabel(feature_names_2d[0])
            ax1.set_ylabel(feature_names_2d[1])
            ax1.set_title('Risultato K-means sui Film')
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Cluster reali
            for i in range(n_clusters_true):
                cluster_points = X_2d[y_true == i]
                cluster_name = cluster_info[i]["name"] if i in cluster_info else f"Cluster reale {i+1}"
                ax2.scatter(cluster_points[:, 0], cluster_points[:, 1], label=cluster_name)
            ax2.set_xlabel(feature_names_2d[0])
            ax2.set_ylabel(feature_names_2d[1])
            ax2.set_title('Cluster Reali dei Film')
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            st.pyplot(fig_final)
            
            # Matrice di confusione
            st.subheader("Valutazione del clustering sui Film")
            
            confusion_matrix = np.zeros((kmeans.k, n_clusters_true), dtype=int)
            
            for i in range(kmeans.k):
                for j in range(n_clusters_true):
                    confusion_matrix[i, j] = np.sum((kmeans.clusters == i) & (y_true == j))
            
            fig_conf, ax_conf = plt.subplots(figsize=(10, 8))
            cluster_names = [cluster_info[i]["name"] if i in cluster_info else f"Tipo {i+1}" 
                           for i in range(n_clusters_true)]
            
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=cluster_names, 
                        yticklabels=[f'Cluster {i+1}' for i in range(kmeans.k)],
                        ax=ax_conf)
            ax_conf.set_xlabel('Tipi di Film Reali')
            ax_conf.set_ylabel('Cluster K-means')
            ax_conf.set_title('Matrice di confusione tra cluster K-means e tipi di film reali')
            
            st.pyplot(fig_conf)
            
            # Calcolo dell'accuratezza
            correct_classifications = 0
            for i in range(kmeans.k):
                correct_classifications += np.max(confusion_matrix[i])
            accuracy = correct_classifications / len(y_true)
            
            st.metric("Accuratezza del clustering", f"{accuracy:.2%}")
        
        else:
            # Solo risultato K-means
            fig_final, ax_final = plt.subplots(figsize=(12, 8))
            
            for j in range(kmeans.k):
                cluster_points = X_2d[kmeans.clusters == j]
                ax_final.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster Film {j+1}', s=50)
            
            if X_clustering.shape[1] > 2:
                centroids_2d = kmeans.centroids[:, :2]
            else:
                centroids_2d = kmeans.centroids
                
            ax_final.scatter(centroids_2d[:, 0], centroids_2d[:, 1], s=200, c='red', marker='X', label='Centroidi')
            ax_final.set_xlabel(feature_names_2d[0])
            ax_final.set_ylabel(feature_names_2d[1])
            ax_final.set_title('Clustering K-means sui Film')
            ax_final.legend()
            ax_final.grid(True, linestyle='--', alpha=0.7)
            
            st.pyplot(fig_final)
        
        # Analisi dei cluster trovati
        st.subheader("Analisi dei Cluster Trovati")
        
        # Aggiungo i cluster al dataframe per l'analisi
        movies_df_clustered = movies_df.copy()
        movies_df_clustered['Cluster'] = kmeans.clusters
        
        # Statistiche per cluster
        cluster_stats = movies_df_clustered.groupby('Cluster').agg({
            'Voto_Medio': ['mean', 'std', 'count'],
            'Num_Recensioni': ['mean', 'std'],
            'Durata_Min': ['mean', 'std'],
            'Genere': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'N/A'
        }).round(2)
        
        cluster_stats.columns = ['Voto_Medio', 'Voto_StdDev', 'Num_Film', 'Recensioni_Media', 'Recensioni_StdDev', 
                               'Durata_Media', 'Durata_StdDev', 'Genere_Principale']
        
        st.write("**Caratteristiche dei cluster identificati:**")
        st.dataframe(cluster_stats)
        
        # Interpretazione dei cluster
        st.write("**Interpretazione dei cluster:**")
        for i in range(kmeans.k):
            cluster_data = cluster_stats.loc[i]
            interpretation = f"**Cluster {i+1}**: "
            
            if cluster_data['Voto_Medio'] > 7.5:
                interpretation += "Film di alta qualità "
            elif cluster_data['Voto_Medio'] > 6:
                interpretation += "Film nella media "
            else:
                interpretation += "Film di bassa qualità "
                
            if cluster_data['Recensioni_Media'] > 1000:
                interpretation += "con alta popolarità "
            elif cluster_data['Recensioni_Media'] > 500:
                interpretation += "con media popolarità "
            else:
                interpretation += "di nicchia "
                
            interpretation += f"(genere principale: {cluster_data['Genere_Principale']})"
            
            st.write(interpretation)

# Colonna laterale per informazioni didattiche
with col2:
    st.header("Apprendimento")
    
    st.subheader("K-means sui Film")
    st.markdown("""
    In questo esempio, applichiamo K-means per identificare gruppi di film simili basandoci su:
    - **Voto medio degli utenti**: Qualità percepita del film
    - **Numero di recensioni**: Popolarità del film  
    - **Durata**: Lunghezza del film
    - **Genere**: Categoria cinematografica (opzionale)
    """)
    
    st.subheader("Applicazioni Pratiche")
    st.markdown("""
    Il clustering dei film può essere utile per:
    - **Sistemi di raccomandazione**: Suggerire film simili
    - **Analisi di mercato**: Identificare segmenti di pubblico
    - **Strategia di contenuti**: Pianificare produzioni
    - **Distribuzione**: Ottimizzare campagne marketing
    - **Catalogazione**: Organizzare librerie digitali
    """)
    
    st.subheader("Interpretazione dei Cluster")
    st.markdown("""
    I cluster tipici che si possono formare sono:
    - **Blockbuster**: Voti alti, molte recensioni
    - **Film d'Arte**: Voti molto alti, poche recensioni
    - **Film Commerciali**: Voti medi, recensioni medie
    - **Film di Nicchia**: Voti variabili, poche recensioni
    """)
    
    st.subheader("Fasi dell'algoritmo")
    st.markdown("""
    1. **Inizializzazione**: Scegli K centroidi casuali nello spazio delle feature
    2. **Assegnazione**: Ogni film viene assegnato al cluster più vicino
    3. **Aggiornamento**: I centroidi vengono ricalcolati come media dei film nel cluster
    4. **Ripetizione**: Continua fino alla convergenza
    """)
    
    st.subheader("Considerazioni sui Dati Film")
    st.markdown("""
    **Caratteristiche del dataset**:
    - **Voti**: Scala 1-10, riflette la qualità percepita
    - **Recensioni**: Indica popolarità e reach del film
    - **Genere**: Influenza il tipo di pubblico
    - **Durata**: Può indicare il tipo di produzione
    
    **Preprocessing importante**:
    - Standardizzazione delle scale diverse
    - Gestione dei valori estremi (outlier)
    - Encoding appropriato per variabili categoriche
    """)
    
    st.subheader("Vantaggi su Dataset Film")
    st.markdown("""
    **Vantaggi**:
    - Identifica pattern nascosti nelle preferenze
    - Scalabile per grandi cataloghi
    - Utile per personalizzazione automatica
    
    **Limitazioni**:
    - Richiede scelta del numero K
    - Sensibile alla scala delle feature
    - Non cattura relazioni non lineari complesse
    """)

# Footer con istruzioni specifiche per i film
st.markdown("""
---
### Come utilizzare questa applicazione per l'analisi dei film

1. **Scegli il tipo di dataset**: Film casuali o con cluster predefiniti
2. **Seleziona le feature**: Combina voti, recensioni, durata secondo il tuo interesse
3. **Configura K-means**: Imposta il numero di cluster e altri parametri
4. **Esegui l'algoritmo**: Osserva come i film vengono raggruppati
5. **Analizza i risultati**: Interpreta i cluster in base alle caratteristiche cinematografiche

### Esempi di insight che puoi ottenere:

- **Cluster "Blockbuster"**: Film con voti alti (7-8) e molte recensioni (>1000)
- **Cluster "Cult/Art House"**: Film con voti molto alti (8+) ma poche recensioni
- **Cluster "Mainstream"**: Film con voti medi (5-7) e recensioni moderate
- **Cluster "Nicchia"**: Film specializzati con caratteristiche uniche

### Suggerimenti per l'interpretazione:

- Osserva come le feature influenzano la formazione dei cluster
- Confronta i cluster trovati con i generi cinematografici
- Considera l'evoluzione temporale (anni di uscita) nei cluster
- Analizza la relazione tra qualità (voti) e popolarità (recensioni)

Questa visualizzazione ti aiuta a comprendere come l'algoritmo K-means può essere applicato 
nell'industria cinematografica per scoprire pattern nei dati e supportare decisioni di business.
""")

# Sezione aggiuntiva per dataset insight
if 'movies_df' in locals():
    st.sidebar.markdown("---")
    st.sidebar.subheader("Insights Dataset")
    
    # Distribuzione per genere
    genre_counts = movies_df['Genere'].value_counts()
    st.sidebar.write("**Top 3 Generi:**")
    for i, (genre, count) in enumerate(genre_counts.head(3).items()):
        st.sidebar.write(f"{i+1}. {genre}: {count} film")
    
    # Film con voto più alto
    best_movie = movies_df.loc[movies_df['Voto_Medio'].idxmax()]
    st.sidebar.write(f"**Miglior Film**: {best_movie['Titolo']} ({best_movie['Voto_Medio']}/10)")
    
    # Film più recensito
    most_reviewed = movies_df.loc[movies_df['Num_Recensioni'].idxmax()]
    st.sidebar.write(f"**Più Recensito**: {most_reviewed['Titolo']} ({most_reviewed['Num_Recensioni']} recensioni)")
    
    # Decade più rappresentata
    movies_df['Decade'] = (movies_df['Anno'] // 10) * 10
    decade_counts = movies_df['Decade'].value_counts()
    most_common_decade = decade_counts.index[0]
    st.sidebar.write(f"**Decade dominante**: {most_common_decade}s ({decade_counts.iloc[0]} film)")
    
    
   # Matrice di confusione per dataset Film
st.subheader("Valutazione del clustering")
confusion_matrix = np.zeros((kmeans.k, len(target_names)), dtype=int)
for i in range(kmeans.k):
    for j in range(len(target_names)):
        confusion_matrix[i, j] = np.sum((kmeans.clusters == i) & (y_true == j))

fig_conf, ax_conf = plt.subplots(figsize=(10, 8))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Azione', 'Dramma', 'Thriller', 'Musical', 'Romance'], yticklabels=[f'Cluster {i+1}' for i in range(kmeans.k)], ax=ax_conf)
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