import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import seaborn as sns
import time
from matplotlib.colors import LinearSegmentedColormap
import random

# Configurazione pagina Streamlit
st.set_page_config(
    page_title="K-means per Classificazione Film",
    page_icon="🎬",
    layout="wide",
)

# Titolo e introduzione
st.title("🎬 Classificazione di Film con K-means")
st.markdown("""
Questa applicazione dimostra come l'algoritmo K-means può essere utilizzato per classificare film
basandosi su caratteristiche come budget, durata, anno di uscita e incassi. Puoi modificare i parametri
dell'algoritmo e vedere come cambiano i cluster identificati.
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

# Funzione per calcolare le metriche di valutazione
def calculate_clustering_metrics(true_labels, predicted_labels):
    """Calcola diverse metriche per valutare la qualità del clustering"""
    # Adjusted Rand Index
    ari = adjusted_rand_score(true_labels, predicted_labels)
    
    # Normalized Mutual Information
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    
    return ari, nmi

# Funzione per creare una matrice di confusione dettagliata
def create_detailed_confusion_matrix(df, true_col, pred_col):
    """Crea una matrice di confusione con statistiche dettagliate"""
    confusion_matrix = pd.crosstab(
        df[pred_col], 
        df[true_col], 
        rownames=[pred_col], 
        colnames=[true_col],
        margins=True
    )
    
    # Calcola accuratezza per cluster
    cluster_accuracy = {}
    for cluster in range(df[pred_col].nunique()):
        cluster_data = confusion_matrix.loc[cluster, :-1]  # Escludi il totale
        if cluster_data.sum() > 0:
            max_genre_count = cluster_data.max()
            cluster_accuracy[cluster] = (max_genre_count / cluster_data.sum()) * 100
    
    return confusion_matrix, cluster_accuracy

# Funzione per generare il dataset sintetico di film
def generate_movies_dataset(n_samples=200, random_state=42):
    random.seed(random_state)
    np.random.seed(random_state)
    
    # Definizione dei generi e delle relative caratteristiche
    genres = {
        "Action": {
            "budget_range": (50, 250),  # milioni di dollari
            "duration_range": (90, 150),  # minuti
            "year_range": (1980, 2024),
            "revenue_multiplier_range": (2.0, 5.0)  # moltiplicatore rispetto al budget
        },
        "Drama": {
            "budget_range": (5, 80),
            "duration_range": (100, 180),
            "year_range": (1950, 2024),
            "revenue_multiplier_range": (1.0, 4.0)
        },
        "Comedy": {
            "budget_range": (20, 100),
            "duration_range": (85, 130),
            "year_range": (1970, 2024),
            "revenue_multiplier_range": (2.0, 6.0)
        },
        "Horror": {
            "budget_range": (3, 70),
            "duration_range": (80, 120),
            "year_range": (1970, 2024),
            "revenue_multiplier_range": (3.0, 10.0)
        },
        "Sci-Fi": {
            "budget_range": (40, 220),
            "duration_range": (100, 160),
            "year_range": (1970, 2024),
            "revenue_multiplier_range": (1.5, 5.5)
        }
    }
    
    # Lista di titoli di film per genere (esempi fittizi)
    movie_titles = {
        "Action": [
            "Explosive Impact", "Thunder Strike", "Velocity Limit", "Metal Warriors",
            "Last Stand", "Deadly Mission", "Power Surge", "Tactical Force",
            "Maximum Velocity", "Combat Zone", "Battle Ground", "Urban Warfare",
            "Critical Hit", "Strike Force", "Lethal Protocol", "Steel Justice",
            "Crisis Point", "Extreme Danger", "Terminal Velocity", "Rogue Element"
        ],
        "Drama": [
            "Silent Tears", "Forgotten Dreams", "The Weight of Words", "Lost Memories",
            "Endless Sky", "Broken Promises", "Distant Echoes", "Fading Light",
            "Hidden Truth", "The Road Ahead", "Final Chapter", "Emotional Landscape",
            "Whispers in Time", "Shattered Glass", "Redemption Path", "Uncertain Future",
            "Between Worlds", "Fractured Lives", "Beneath the Surface", "Inner Struggles"
        ],
        "Comedy": [
            "Laugh Factor", "Complete Chaos", "Family Disaster", "Office Hours",
            "Dating Troubles", "Weekend Warriors", "The Perfect Plan", "Unlikely Heroes",
            "Party Animals", "Vacation Disaster", "Awkward Moments", "Best Friends Forever",
            "Crazy Neighbors", "Double Trouble", "Wedding Crashers", "School Reunion",
            "Absolute Mayhem", "Happy Accidents", "Unexpected Detour", "Game Night"
        ],
        "Horror": [
            "Dark Shadows", "The Haunting", "Midnight Terror", "Silent Screams",
            "Abandoned", "Evil Presence", "Deadly Visions", "The Unknown Entity",
            "Cursed Land", "Paranormal Activity", "Blood Moon", "The Awakening",
            "Unspeakable Fear", "The Ritual", "Final Hour", "Possession",
            "Nightmare House", "Whispers of Death", "The Crypt", "Eternal Darkness"
        ],
        "Sci-Fi": [
            "Space Colony", "Time Paradox", "Quantum Field", "The Last Algorithm",
            "Beyond Horizons", "Digital Consciousness", "Alien Protocol", "Neural Interface",
            "Galactic Frontier", "Parallel Dimensions", "Future State", "Synthetic Humanity",
            "The Multiverse", "Planetary Exodus", "Deep Space", "Temporal Shift",
            "Cyber Evolution", "Solar Crisis", "Interstellar", "The Singularity"
        ]
    }
    
    # Creo liste per i dati
    titles = []
    budgets = []
    durations = []
    years = []
    revenues = []
    genre_labels = []
    
    # Numero di film per genere
    samples_per_genre = n_samples // len(genres)
    remainder = n_samples % len(genres)
    
    # Genera dati per ogni genere
    genre_index = 0
    for genre, properties in genres.items():
        # Aggiungi il resto ai primi generi se n_samples non è divisibile per il numero di generi
        samples = samples_per_genre + (1 if genre_index < remainder else 0)
        genre_index += 1
        
        for _ in range(samples):
            # Seleziona un titolo casuale per il genere
            title = random.choice(movie_titles[genre]) + " " + str(random.randint(1, 5))
            
            # Genera caratteristiche con un po' di rumore
            budget = random.uniform(*properties["budget_range"])
            duration = random.uniform(*properties["duration_range"])
            year = random.randint(*properties["year_range"])
            
            # Il revenue è una funzione del budget con un moltiplicatore variabile
            revenue_multiplier = random.uniform(*properties["revenue_multiplier_range"])
            revenue = budget * revenue_multiplier * (1 + random.uniform(-0.3, 0.3))  # Aggiunge variabilità
            
            # Aggiunge i dati alle liste
            titles.append(title)
            budgets.append(budget)
            durations.append(duration)
            years.append(year)
            revenues.append(revenue)
            genre_labels.append(genre)
    
    # Crea un DataFrame
    df = pd.DataFrame({
        'Title': titles,
        'Budget (mln $)': budgets,
        'Duration (min)': durations,
        'Year': years,
        'Revenue (mln $)': revenues,
        'Genre': genre_labels
    })
    
    # Converti i generi in numeri per l'analisi
    genre_mapping = {genre: i for i, genre in enumerate(genres.keys())}
    df['Genre_numeric'] = df['Genre'].map(genre_mapping)
    
    return df, list(genres.keys())

# Sidebar per la configurazione
st.sidebar.header("Configurazione")

# Numero di film nel dataset
n_movies = st.sidebar.slider("Numero di film nel dataset", 100, 500, 200)

# Seed per la generazione dei dati
data_seed = st.sidebar.slider("Seed per generazione dati", 0, 100, 42)

# Seleziona caratteristiche per clustering
st.sidebar.subheader("Caratteristiche per il clustering")
use_budget = st.sidebar.checkbox("Budget", value=True)
use_duration = st.sidebar.checkbox("Durata", value=True)
use_year = st.sidebar.checkbox("Anno di uscita", value=True)
use_revenue = st.sidebar.checkbox("Incassi", value=True)

# Configurazione K-means
st.sidebar.subheader("Parametri K-means")
k_value = st.sidebar.slider("Numero di cluster (k)", 2, 8, 5)
max_iterations = st.sidebar.slider("Numero massimo di iterazioni", 10, 100, 20)
kmeans_seed = st.sidebar.slider("Seed casuale per K-means", 0, 100, 42)

# Visualizzazione da mostrare
st.sidebar.subheader("Visualizzazione")
viz_features = st.sidebar.selectbox(
    "Caratteristiche da visualizzare",
    options=[
        "Budget vs Durata", 
        "Budget vs Incassi", 
        "Durata vs Anno", 
        "Anno vs Incassi"
    ]
)

# Mappa la scelta delle caratteristiche
feature_mapping = {
    "Budget vs Durata": ("Budget (mln $)", "Duration (min)"),
    "Budget vs Incassi": ("Budget (mln $)", "Revenue (mln $)"),
    "Durata vs Anno": ("Duration (min)", "Year"),
    "Anno vs Incassi": ("Year", "Revenue (mln $)")
}

# Genera il dataset di film
df_movies, genre_names = generate_movies_dataset(n_samples=n_movies, random_state=data_seed)

# Seleziona le caratteristiche in base alle caselle di controllo
selected_features = []
if use_budget:
    selected_features.append("Budget (mln $)")
if use_duration:
    selected_features.append("Duration (min)")
if use_year:
    selected_features.append("Year")
if use_revenue:
    selected_features.append("Revenue (mln $)")

# Verifica che almeno due caratteristiche siano selezionate
if len(selected_features) < 2:
    st.error("Seleziona almeno due caratteristiche per il clustering.")
else:
    # Estrai i dati selezionati
    X = df_movies[selected_features].values
    
    # Standardizzazione dei dati
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Mostra il dataset
    st.subheader("Dataset dei Film")
    st.write(df_movies[['Title', 'Genre', 'Budget (mln $)', 'Duration (min)', 'Year', 'Revenue (mln $)']].head(10))
    
    # Mostra statistiche
    st.subheader("Statistiche per Genere")
    
    # Crea una tabella di statistiche per genere
    genre_stats = df_movies.groupby('Genre').agg({
        'Budget (mln $)': ['mean', 'min', 'max'],
        'Duration (min)': ['mean', 'min', 'max'],
        'Year': ['mean', 'min', 'max'],
        'Revenue (mln $)': ['mean', 'min', 'max']
    }).round(2)
    
    st.write(genre_stats)
    
    # Layout principale per i grafici
    col1, col2 = st.columns([2, 1])
    
    # Seleziona le caratteristiche per la visualizzazione
    x_feature, y_feature = feature_mapping[viz_features]
    x_index = selected_features.index(x_feature) if x_feature in selected_features else 0
    y_index = selected_features.index(y_feature) if y_feature in selected_features else 1
    
    # Mostra la distribuzione dei generi reali
    with col1:
        st.subheader(f"Distribuzione dei Film: {viz_features}")
        
        # Crea una palette di colori personalizzata
        colors = plt.cm.tab10(np.linspace(0, 1, len(genre_names)))
        genre_colors = {genre: colors[i] for i, genre in enumerate(genre_names)}
        
        fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
        
        for genre in genre_names:
            genre_data = df_movies[df_movies['Genre'] == genre]
            ax_dist.scatter(
                genre_data[x_feature], 
                genre_data[y_feature],
                label=genre,
                alpha=0.7,
                edgecolors='w',
                linewidth=0.5
            )
        
        ax_dist.set_xlabel(x_feature)
        ax_dist.set_ylabel(y_feature)
        ax_dist.set_title(f'Distribuzione dei film per genere: {x_feature} vs {y_feature}')
        ax_dist.legend()
        ax_dist.grid(True, linestyle='--', alpha=0.7)
        
        st.pyplot(fig_dist)
    
    # Contenuto informativo
    with col2:
        st.subheader("Caratteristiche dei Generi Cinematografici")
        
        genre_descriptions = {
            "Action": "Film con elevati budget, durate moderate, effetti speciali costosi e incassi tipicamente alti.",
            "Drama": "Budget più contenuti, durate maggiori, storie emotive e incassi variabili.",
            "Comedy": "Budget medi, durate più brevi, umorismo e situazioni comiche con buoni incassi.",
            "Horror": "Budget bassi, durate contenute, tensione e paura con ottimi ritorni sull'investimento.",
            "Sci-Fi": "Budget elevati, durate lunghe, effetti speciali avanzati e incassi potenzialmente molto alti."
        }
        
        for genre, desc in genre_descriptions.items():
            st.markdown(f"**{genre}**: {desc}")
        
        st.markdown("---")
        
        st.subheader("Perché usare K-means per i film?")
        st.markdown("""
        Il clustering dei film può essere utile per:
        - Identificare nicchie di mercato
        - Ottimizzare budget e investimenti
        - Migliorare i sistemi di raccomandazione
        - Analizzare tendenze di mercato
        - Supportare decisioni di marketing
        """)
    
    # Avvia il clustering
    if st.button("Esegui K-means"):
        # Istanziamo e addestriamo il modello K-means
        kmeans = KMeansClustering(k=k_value, max_iterations=max_iterations, random_state=kmeans_seed)
        kmeans.fit(X_scaled, verbose=False)
        
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
                    cluster_points = X_scaled[kmeans.all_clusters[i-1] == j]
                    if len(cluster_points) > 0:
                        ax_iter.scatter(
                            cluster_points[:, x_index], 
                            cluster_points[:, y_index], 
                            alpha=0.7,
                            label=f'Cluster {j+1}'
                        )
            else:  # Solo per l'inizializzazione
                ax_iter.scatter(X_scaled[:, x_index], X_scaled[:, y_index], alpha=0.3, color='gray')
            
            # Plottiamo i centroidi
            centroids = kmeans.all_centroids[i]
            ax_iter.scatter(
                centroids[:, x_index], 
                centroids[:, y_index], 
                s=200, 
                c='red', 
                marker='X', 
                label='Centroidi'
            )
            
            if i == 0:
                ax_iter.set_title('Inizializzazione dei centroidi')
            else:
                sse = kmeans.all_sse[i-1]
                ax_iter.set_title(f'Iterazione {i} - SSE: {sse:.4f}')
            
            ax_iter.set_xlabel(x_feature)
            ax_iter.set_ylabel(y_feature)
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
        
        # Salviamo i cluster nel dataframe
        df_movies['Cluster'] = kmeans.clusters
        
        # Visualizziamo il risultato finale
        st.subheader("Risultato finale del clustering")
        
        fig_final, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # K-means clustering
        for j in range(kmeans.k):
            cluster_points = X_scaled[kmeans.clusters == j]
            cluster_indices = np.where(kmeans.clusters == j)[0]
            titles_in_cluster = df_movies.iloc[cluster_indices]['Title'].tolist()
            if len(cluster_points) > 0:
                ax1.scatter(
                    cluster_points[:, x_index], 
                    cluster_points[:, y_index], 
                    label=f'Cluster {j+1}',
                    alpha=0.7
                )
        
        ax1.scatter(
            kmeans.centroids[:, x_index], 
            kmeans.centroids[:, y_index], 
            s=200, 
            c='red', 
            marker='X', 
            label='Centroidi'
        )
        ax1.set_xlabel(x_feature)
        ax1.set_ylabel(y_feature)
        ax1.set_title('Risultato K-means')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Generi reali
        for i, genre in enumerate(genre_names):
            genre_points = X_scaled[df_movies['Genre'] == genre]
            if len(genre_points) > 0:
                ax2.scatter(
                    genre_points[:, x_index], 
                    genre_points[:, y_index], 
                    label=genre,
                    alpha=0.7
                )
        
        ax2.set_xlabel(x_feature)
        ax2.set_ylabel(y_feature)
        ax2.set_title('Generi reali')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        st.pyplot(fig_final)
        
        # Calcolo metriche di valutazione
        ari, nmi = calculate_clustering_metrics(df_movies['Genre_numeric'], df_movies['Cluster'])
        
        # Mostra metriche di valutazione
        st.subheader("Metriche di Valutazione del Clustering")
        
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        
        with col_metric1:
            st.metric(
                label="Adjusted Rand Index (ARI)",
                value=f"{ari:.3f}",
                help="Misura la similarità tra clustering predetto e vero. Valore da -1 a 1 (1 = perfetto)"
            )
        
        with col_metric2:
            st.metric(
                label="Normalized Mutual Information",
                value=f"{nmi:.3f}",
                help="Misura l'informazione condivisa tra clustering. Valore da 0 a 1 (1 = perfetto)"
            )
        
        with col_metric3:
            st.metric(
                label="SSE Finale",
                value=f"{kmeans.sse:.2f}",
                help="Somma degli errori quadratici - misura la compattezza dei cluster"
            )
        
        # Analisi dettagliata dei cluster
        st.subheader("Analisi Dettagliata dei Cluster")
        
        # Tabella dei film per cluster
        cluster_counts = df_movies['Cluster'].value_counts().sort_index()
        st.write(f"**Distribuzione dei film nei cluster:**")
        
        # Crea un dataframe per la distribuzione dei cluster
        cluster_dist_df = pd.DataFrame({
            'Cluster': [f"Cluster {i+1}" for i in range(k_value)],
            'Numero di Film': [cluster_counts[i] if i in cluster_counts.index else 0 for i in range(k_value)]
        })
        
        st.write(cluster_dist_df)
        
        # Matrice di confusione dettagliata
        st.subheader("📊 Matrice di Confusione Dettagliata")
        
        confusion_matrix, cluster_accuracy = create_detailed_confusion_matrix(
            df_movies, 'Genre', 'Cluster'
        )
        
        # Layout per matrice di confusione
        col_conf1, col_conf2 = st.columns([2, 1])
        
        with col_conf1:
            # Visualizzazione della matrice di confusione con heatmap
            fig_conf, ax_conf = plt.subplots(figsize=(12, 8))
            
            # Crea la heatmap senza i totali
            conf_matrix_no_totals = confusion_matrix.iloc[:-1, :-1]
            
            sns.heatmap(
                conf_matrix_no_totals, 
                annot=True, 
                fmt='d', 
                cmap='Blues', 
                ax=ax_conf,
                cbar_kws={'label': 'Numero di Film'}
            )
            
            ax_conf.set_title('Matrice di Confusione: Cluster vs Generi Reali', fontsize=14, fontweight='bold')
            ax_conf.set_xlabel('Genere Reale', fontsize=12)
            ax_conf.set_ylabel('Cluster Predetto', fontsize=12)
            
            # Aggiungi etichette personalizzate
            ax_conf.set_xticklabels(ax_conf.get_xticklabels(), rotation=45, ha='right')
            ax_conf.set_yticklabels([f'Cluster {i}' for i in range(k_value)], rotation=0)
            
            plt.tight_layout()
            st.pyplot(fig_conf)
        
        with col_conf2:
            st.write("**Accuratezza per Cluster:**")
            
            accuracy_df = pd.DataFrame([
                {'Cluster': f'Cluster {k}', 'Accuratezza (%)': f'{v:.1f}%'}
                for k, v in cluster_accuracy.items()
            ])
            
            st.write(accuracy_df)
            
            # Accuratezza media
            avg_accuracy = np.mean(list(cluster_accuracy.values()))
            st.metric(
                label="Accuratezza Media",
                value=f"{avg_accuracy:.1f}%",
                help="Percentuale media di film classificati correttamente per cluster"
            )
            
            st.markdown("---")
            
            st.write("**Interpretazione della Matrice:**")
            st.markdown("""
            - **Diagonale**: Film classificati correttamente
            - **Valori alti sulla diagonale**: Buona separazione
            - **Valori distribuiti**: Sovrapposizione tra generi
            - **Cluster puri**: Alta concentrazione su un genere
            """)
        
        # Tabella dettagliata della matrice di confusione
        st.write("**Matrice di Confusione Completa (con Totali):**")
        
        # Rinomina le righe per maggiore chiarezza
        confusion_display = confusion_matrix.copy()
        confusion_display.index = [f'Cluster {i}' if i != 'All' else 'Totale' for i in confusion_display.index]
        
        st.write(confusion_display)
        
     ax_dist.set_title(f'Distribuzione dei film per genere: {x_feature} vs {y_feature}')
ax_dist.legend()
st.pyplot(fig_dist)


    # Esecuzione del clustering K-means
    st.subheader("Clustering con K-means")
    kmeans = KMeansClustering(k=k_value, max_iterations=max_iterations, random_state=kmeans_seed)
    kmeans.fit(X_scaled)

    # Assegna i cluster al DataFrame
    df_movies['Cluster'] = kmeans.clusters

    # Visualizza i cluster sullo stesso grafico
    with col2:
        st.subheader("Cluster trovati")
        cluster_colors = plt.cm.Set1(np.linspace(0, 1, k_value))
        fig_cluster, ax_cluster = plt.subplots(figsize=(6, 6))
        for cluster in range(k_value):
            cluster_data = df_movies[df_movies['Cluster'] == cluster]
            ax_cluster.scatter(
                cluster_data[x_feature],
                cluster_data[y_feature],
                label=f'Cluster {cluster}',
                alpha=0.7,
                color=cluster_colors[cluster],
                edgecolors='k',
                linewidth=0.5
            )
        ax_cluster.set_xlabel(x_feature)
        ax_cluster.set_ylabel(y_feature)
        ax_cluster.set_title(f'Cluster K-means: {x_feature} vs {y_feature}')
        ax_cluster.legend()
        st.pyplot(fig_cluster)

    # Mostra la tabella con cluster e generi
    st.subheader("Tabella Cluster e Generi")
    st.write(df_movies[['Title', 'Genre', 'Cluster']].head(20))

    # Valutazione del clustering se disponibile la colonna 'Genre_numeric'
    if 'Genre_numeric' in df_movies.columns:
        ari, nmi = calculate_clustering_metrics(df_movies['Genre_numeric'], df_movies['Cluster'])
        st.markdown(f"**Adjusted Rand Index (ARI):** {ari:.3f}")
        st.markdown(f"**Normalized Mutual Information (NMI):** {nmi:.3f}")

        # Matrice di confusione dettagliata
        confusion_matrix, cluster_accuracy = create_detailed_confusion_matrix(df_movies, 'Genre', 'Cluster')
        st.subheader("Matrice di Confusione (Cluster vs Genere)")
        st.write(confusion_matrix)
        st.markdown("**Accuratezza per cluster:**")
        for cluster, acc in cluster_accuracy.items():
            st.markdown(f"- Cluster {cluster}: {acc:.1f}%")

    # Mostra l'andamento della SSE
    st.subheader("Andamento SSE (Somma degli Errori Quadratici)")
    fig_sse, ax_sse = plt.subplots()
    ax_sse.plot(range(1, len(kmeans.all_sse) + 1), kmeans.all_sse, marker='o')
    ax_sse.set_xlabel("Iterazione")
    ax_sse.set_ylabel("SSE")
    ax_sse.set_title("Andamento della SSE durante il K-means")
    st.pyplot(fig_sse)
