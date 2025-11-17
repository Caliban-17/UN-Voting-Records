import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import warnings
import logging
import os
import sys
import asyncio
import threading
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    UN_VOTES_CSV_PATH,
    VOTE_MAP,
    COLUMN_RENAME_MAP,
    ESSENTIAL_COLUMNS,
    OPTIONAL_COLUMNS,
    DEFAULT_N_CLUSTERS,
    DEFAULT_TRAIN_TEST_SPLIT_YEAR,
    DEFAULT_N_TOP_ISSUES,
    DEFAULT_FIG_SIZE,
    DEFAULT_MARKER_SIZE,
    DEFAULT_ALPHA,
    PLOT_COLORS
)
from src.data_processing import (
    load_and_preprocess_data,
    create_vote_matrix,
    calculate_country_entropy,
    get_issue_statistics
)
from src.model import train_vote_predictor, predict_votes
from src.visualization import (
    plot_cluster_vote_distribution,
    plot_pca_projection,
    plot_issue_salience,
    plot_resolution_polarity,
    plot_entropy_distribution
)

# Import functions from main.py
from src.main import (
    preprocess_for_similarity,
    calculate_similarity,
    perform_clustering,
    calculate_country_entropy,
    get_memory_usage,
    create_minimal_features,
    train_vote_predictor_async
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Global flag for shutdown management
shutdown_event = asyncio.Event()

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="UN Voting Analyzer (CSV Data)")
st.title("📊 UN Voting Analyzer & Predictor (CSV Data)")

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

# --- Session State Initialization ---
if 'un_data' not in st.session_state:
    st.session_state['un_data'] = pd.DataFrame()
if 'unique_issues' not in st.session_state:
    st.session_state['unique_issues'] = ["Load Data First"]
if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False
if 'shutdown_requested' not in st.session_state:
    st.session_state['shutdown_requested'] = False

def check_shutdown():
    """Check if shutdown has been requested."""
    if shutdown_event.is_set() or st.session_state['shutdown_requested']:
        st.warning("Shutdown requested. Please wait for cleanup...")
        st.stop()

def load_un_votes_data(filepath: str) -> tuple[pd.DataFrame | None, list | None]:
    """Loads UN voting data from the specified CSV, maps columns, preprocesses."""
    abs_filepath = os.path.abspath(filepath)
    if not os.path.exists(filepath):
        logger.error(f"File not found: '{filepath}' ({abs_filepath})")
        return None, None
    try:
        logger.info(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath, low_memory=False)
        logger.info(f"Loaded {len(df)} rows initially.")
        
        column_rename_map = {
            'undl_id': 'rcid',
            'ms_code': 'country_code',
            'ms_name': 'country_name',
            'ms_vote': 'vote',
            'date': 'date',
            'session': 'session',
            'title': 'descr',
            'subjects': 'issue',
            'resolution': 'resolution',
            'agenda_title': 'agenda',
            'undl_link': 'undl_link'
        }
        
        source_columns_needed = list(column_rename_map.keys())
        missing_source_cols = [col for col in source_columns_needed if col not in df.columns]
        if missing_source_cols:
            raise ValueError(f"CSV missing source columns: {', '.join(missing_source_cols)}")
            
        df.rename(columns=column_rename_map, inplace=True)
        
        essential_internal_cols = ['rcid', 'country_code', 'vote', 'date']
        missing_internal = [col for col in essential_internal_cols if col not in df.columns]
        if missing_internal:
            raise ValueError(f"Essential columns missing after renaming: {', '.join(missing_internal)}")
            
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        initial_rows = len(df)
        df.dropna(subset=['date'], inplace=True)
        if len(df) < initial_rows:
            logger.warning(f"Dropped {initial_rows - len(df)} invalid date rows.")
        if df.empty:
            raise ValueError("No valid date entries found.")
            
        df['year'] = df['date'].dt.year
        df['country_identifier'] = df['country_code'].astype(str)
        df.dropna(subset=['country_identifier'], inplace=True)
        
        df['vote'] = df['vote'].astype(str).str.strip()
        df.dropna(subset=['vote'], inplace=True)
        
        if 'issue' not in df.columns:
            if 'descr' in df.columns:
                df['issue'] = df['descr'].astype(str).str.split().str[:5].str.join(' ').fillna('Unknown/No Issue')
            else:
                df['issue'] = 'Unknown/No Issue'
        else:
            df['issue'] = df['issue'].fillna('Unknown/No Issue')
            
        # Handle potential multi-valued subjects (split by '; ' if present) take first?
        if df['issue'].dtype == 'object':  # Check if it's string data
            df['issue'] = df['issue'].astype(str).str.split(';').str[0].str.strip()
        
        unique_issues = sorted(df['issue'].astype(str).unique().tolist())
        
        final_cols_needed = ['rcid', 'country_identifier', 'vote', 'date', 'year', 'issue']
        optional_context_cols = ['country_name', 'country_code', 'descr', 'session', 'resolution', 'agenda', 'undl_link']
        if 'importantvote' in df.columns:
            optional_context_cols.append('importantvote')
            
        final_cols = final_cols_needed + [col for col in optional_context_cols if col in df.columns]
        missing_final = [col for col in final_cols_needed if col not in df.columns]
        if missing_final:
            raise ValueError(f"Essential columns missing before final selection: {', '.join(missing_final)}")
            
        df_final = df[final_cols].copy()
        logger.info(f"Loaded and processed {len(df_final)} records.")
        return df_final, unique_issues
        
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return None, None
    except ValueError as ve:
        logger.error(f"Data loading/validation error: {ve}")
        return None, None
    except Exception as e:
        logger.error(f"Unexpected error loading/processing {filepath}: {e}")
        return None, None

# Load the data
un_data, unique_issues = load_un_votes_data(UN_VOTES_CSV_PATH)
if un_data is not None:
    st.session_state['un_data'] = un_data
    st.session_state['unique_issues'] = unique_issues
    st.session_state['data_loaded'] = True
    logger.info("Data loaded successfully into session state")
else:
    logger.error("Failed to load data")

# --- Sidebar Definition ---
st.sidebar.header("⚙️ Data Source")
st.sidebar.markdown(f"""**Using Local CSV File:** `{UN_VOTES_CSV_PATH}`""")
if not st.session_state.data_loaded:
    st.sidebar.error("Data loading failed.")
else:
    st.sidebar.success("Data loaded successfully.")

st.sidebar.header("🔬 Analysis Parameters")
st.sidebar.subheader("Analysis Period & Clustering")
data_available_for_sidebar = st.session_state.data_loaded and not st.session_state.get('un_data', pd.DataFrame()).empty
default_start_year = datetime.now().year - 5
default_end_year = datetime.now().year
min_year_allowed = 1946
max_year_allowed = datetime.now().year + 1
single_year_data = False

if data_available_for_sidebar:
    try:
        rc_dates = st.session_state['un_data']['date']
        valid_dates = rc_dates.dropna()
        if not valid_dates.empty:
            min_year_data = int(valid_dates.dt.year.min())
            max_year_data = int(valid_dates.dt.year.max())
            min_year_allowed = min_year_data
            max_year_allowed = max_year_data
            default_start_year = max(min_year_data, datetime.now().year - 5)
            default_end_year = max_year_data
            if min_year_allowed == max_year_allowed:
                single_year_data = True
                st.sidebar.info(f"Data only for {min_year_allowed}.")
            max_year_allowed_slider = max(min_year_allowed + 1, max_year_allowed)
    except Exception as e:
        st.sidebar.warning(f"Year range parse error: {e}")
else:
    max_year_allowed_slider = max_year_allowed

CLUSTER_START_YEAR = st.sidebar.slider(
    "Analysis Start Year",
    min_year_allowed,
    max_year_allowed_slider - 1,
    default_start_year,
    key="cluster_start",
    disabled=(single_year_data or not data_available_for_sidebar)
)

CLUSTER_END_YEAR = st.sidebar.slider(
    "Analysis End Year",
    CLUSTER_START_YEAR,
    max_year_allowed_slider,
    default_end_year,
    key="cluster_end",
    disabled=(single_year_data or not data_available_for_sidebar)
)

if single_year_data:
    CLUSTER_START_YEAR = min_year_allowed
    CLUSTER_END_YEAR = min_year_allowed

NUM_CLUSTERS = st.sidebar.number_input(
    "Number of Clusters",
    min_value=2,
    max_value=50,
    value=10,
    step=1,
    key="num_clusters",
    disabled=not data_available_for_sidebar
)

st.sidebar.subheader("Prediction / Simulation / Polarity")
ml_min_year = min_year_allowed
ml_max_year = max_year_allowed
default_split_year = max(ml_min_year, ml_max_year - 1) if ml_max_year > ml_min_year else ml_min_year

ML_DATA_SPLIT_YEAR = st.sidebar.number_input(
    "Year to Split Data (Train <= Year < Test)",
    min_value=ml_min_year,
    max_value=ml_max_year,
    value=default_split_year,
    step=1,
    key="ml_split_year",
    disabled=(single_year_data or not data_available_for_sidebar)
)

ML_TRAIN_END_YEAR = ML_DATA_SPLIT_YEAR
ML_TEST_YEAR_START = ML_DATA_SPLIT_YEAR + 1

SIMULATE_ISSUE = st.sidebar.selectbox(
    "Issue for Simulation & Polarity Chart",
    options=st.session_state.get('unique_issues', ["Data not loaded"]),
    index=0,
    key="simulate_issue_updated",
    disabled=not data_available_for_sidebar
)

# --- Main App Tabs ---
tab_cluster, tab_predict = st.tabs(["📊 Analysis & Visualization", "🤖 Prediction"])

current_data = st.session_state.get('un_data', pd.DataFrame())
data_available = st.session_state.get('data_loaded', False)

with tab_cluster:
    st.header(f"📊 Analysis & Visualization ({CLUSTER_START_YEAR}-{CLUSTER_END_YEAR})")
    if not data_available:
        st.warning("Data not loaded. Check file path and console for errors.")
    elif current_data.empty:
        st.warning("Loaded data is empty.")
    else:
        st.markdown(f"*Using data loaded from `{UN_VOTES_CSV_PATH}`*")
        
        # Get vote matrix, country list, and filtered data for analysis
        vote_matrix, country_list, df_filtered_period = preprocess_for_similarity(
            current_data, CLUSTER_START_YEAR, CLUSTER_END_YEAR
        )

        if vote_matrix is not None and not vote_matrix.empty and len(country_list) > 1:
            st.write(f"Analyzing {len(country_list)} countries/entities in {CLUSTER_START_YEAR}-{CLUSTER_END_YEAR}.")
            similarity_matrix = calculate_similarity(vote_matrix)

            if similarity_matrix is not None:
                # --- Clustering ---
                st.subheader("Voting Blocs (Hierarchical Clustering)")
                clusters, final_num_clusters, cluster_labels = perform_clustering(
                    similarity_matrix, NUM_CLUSTERS, country_list
                )

                if clusters:
                    st.write(f"Identified {final_num_clusters} clusters (requested {NUM_CLUSTERS}).")

                    # --- Visualization 1: Cluster Vote Distribution ---
                    st.subheader("Average Vote Distribution per Cluster")
                    st.caption("Shows the proportion of Yes/No/Abstain/Absent votes within each cluster for the selected period.")
                    vote_dist_cols = st.columns(min(final_num_clusters, 4))
                    col_idx = 0
                    for cluster_id in sorted(clusters.keys()):
                        cluster_countries = clusters[cluster_id]
                        cluster_votes = df_filtered_period[df_filtered_period['country_identifier'].isin(cluster_countries)]['vote']
                        vote_counts = cluster_votes.value_counts(normalize=True).reindex(['Y', 'N', 'A', ' '], fill_value=0)
                        vote_counts.index = ['Yes', 'No', 'Abstain', 'Absent/NV']
                        with vote_dist_cols[col_idx % len(vote_dist_cols)]:
                            st.markdown(f"**Cluster {cluster_id + 1}** ({len(cluster_countries)} members)")
                            st.bar_chart(vote_counts, height=200)
                            with st.expander("View Members"):
                                st.write(", ".join(sorted(cluster_countries)))
                        col_idx += 1
                    st.divider()

                    # --- Visualization 2: PCA Scatter Plot ---
                    st.subheader("Voting Pattern Similarity (PCA Projection)")
                    st.caption("Projects high-dimensional voting records onto 2D space. Countries closer together voted more similarly. Colors indicate cluster membership.")
                    try:
                        n_components = 2
                        pca = PCA(n_components=n_components, random_state=42)
                        pca_result = pca.fit_transform(vote_matrix)
                        pca_df = pd.DataFrame(
                            data=pca_result,
                            columns=[f'PC{i+1}' for i in range(n_components)],
                            index=country_list
                        )
                        pca_df['Cluster'] = cluster_labels + 1
                        pca_df['Cluster'] = pca_df['Cluster'].astype(str)
                        pca_df['Country'] = pca_df.index

                        fig_pca, ax_pca = plt.subplots(figsize=(10, 8))
                        sns.scatterplot(
                            x='PC1', y='PC2',
                            hue='Cluster',
                            palette=sns.color_palette("viridis", n_colors=final_num_clusters),
                            data=pca_df,
                            ax=ax_pca,
                            s=50,
                            alpha=0.8,
                            legend='full'
                        )
                        ax_pca.set_title(f'PCA of Voting Patterns ({CLUSTER_START_YEAR}-{CLUSTER_END_YEAR})')
                        ax_pca.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance explained)')
                        ax_pca.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance explained)')
                        ax_pca.grid(True, linestyle='--', alpha=0.5)
                        st.pyplot(fig_pca)
                        plt.close(fig_pca)

                        with st.expander("View PCA Data"):
                            st.dataframe(pca_df[['Country', 'Cluster', 'PC1', 'PC2']])

                    except Exception as pca_e:
                        st.error(f"Could not generate PCA plot: {pca_e}")
                    st.divider()

                else:
                    st.warning("Clustering could not be performed. Cannot display cluster-based visualizations.")

            else:
                st.warning("Similarity calculation failed.")
            st.divider()

            # --- Visualization 3: Timeline of Issue Salience ---
            st.subheader("Timeline of Issue Salience")
            st.caption("Shows the number of votes cast per year on the most frequent issues.")
            n_top_issues = st.slider(
                "Number of Top Issues to Show:",
                min_value=3,
                max_value=20,
                value=10,
                key="top_issues_slider"
            )
            try:
                issue_counts = df_filtered_period.groupby(['year', 'issue'])['rcid'].nunique().reset_index(name='vote_count')
                top_issues = issue_counts.groupby('issue')['vote_count'].sum().nlargest(n_top_issues).index.tolist()
                top_issue_counts = issue_counts[issue_counts['issue'].isin(top_issues)]
                issue_timeline = top_issue_counts.pivot(index='year', columns='issue', values='vote_count').fillna(0)
                if not issue_timeline.empty:
                    st.line_chart(issue_timeline)
                else:
                    st.info("No issue data to plot for the selected period and top issues.")
            except Exception as e:
                st.error(f"Could not generate issue salience timeline: {e}")
            st.divider()

            # --- Visualization 4: Resolution Polarity Chart ---
            st.subheader(f"Vote Polarity for Issue: '{SIMULATE_ISSUE}'")
            st.caption(f"Shows the distribution of votes specifically for resolutions related to the selected issue '{SIMULATE_ISSUE}' within the {CLUSTER_START_YEAR}-{CLUSTER_END_YEAR} period.")
            try:
                issue_data = df_filtered_period[df_filtered_period['issue'] == SIMULATE_ISSUE]
                if not issue_data.empty:
                    issue_vote_counts = issue_data['vote'].value_counts(normalize=True).reindex(['Y', 'N', 'A', ' '], fill_value=0)
                    issue_vote_counts.index = ['Yes', 'No', 'Abstain', 'Absent/NV']
                    fig_pol, ax_pol = plt.subplots()
                    issue_vote_counts.plot(kind='bar', ax=ax_pol, rot=0, title=f"Vote Distribution for '{SIMULATE_ISSUE}'")
                    ax_pol.set_ylabel("Proportion of Votes")
                    st.pyplot(fig_pol)
                    plt.close(fig_pol)
                else:
                    st.info(f"No votes found for the issue '{SIMULATE_ISSUE}' in the selected period.")
            except Exception as e:
                st.error(f"Could not generate resolution polarity chart: {e}")
            st.divider()

            # --- Visualization 5: Vote Entropy by Country ---
            st.subheader("Voting Predictability (Entropy)")
            st.caption("Shannon entropy measures vote predictability. Higher entropy means less predictable (more varied votes). Lower entropy means more consistent voting.")
            try:
                entropy_scores = df_filtered_period.groupby('country_identifier')['vote'].apply(calculate_country_entropy).sort_values()
                entropy_df = entropy_scores.reset_index()
                entropy_df.columns = ['Country Code', 'Vote Entropy']

                top_n_entropy = 10
                col_low, col_high = st.columns(2)
                with col_low:
                    st.write(f"**Top {top_n_entropy} Most Consistent (Lowest Entropy):**")
                    st.dataframe(entropy_df.head(top_n_entropy).round(3), use_container_width=True)
                with col_high:
                    st.write(f"**Top {top_n_entropy} Least Consistent (Highest Entropy):**")
                    st.dataframe(entropy_df.tail(top_n_entropy).iloc[::-1].round(3), use_container_width=True)

            except Exception as e:
                st.error(f"Could not calculate vote entropy: {e}")

        elif vote_matrix is not None:
            st.warning(f"Only {len(country_list)} countries found. Need >= 2 for analysis.")
        else:
            st.info(f"No data available for analysis in the selected year range ({CLUSTER_START_YEAR}-{CLUSTER_END_YEAR}).")

with tab_predict:
    st.header("🤖 Vote Prediction & Simulation")
    if not data_available:
        st.warning("Data not loaded. Check file path and console for errors.")
    elif current_data.empty:
        st.warning("Loaded data is empty.")
    else:
        st.markdown(f"*Using data loaded from `{UN_VOTES_CSV_PATH}`*")
        st.write(f"Train <= {ML_TRAIN_END_YEAR}, Test >= {ML_TEST_YEAR_START}.")
        
        model_pipeline, accuracy, report, all_countries, train_size, test_size = train_vote_predictor(
            current_data, ML_TRAIN_END_YEAR, ML_TEST_YEAR_START
        )
        
        if model_pipeline and all_countries is not None:
            # --- Model Performance Display ---
            st.subheader("Model Performance")
            m_col1, m_col2 = st.columns(2)
            m_col1.metric(
                label=f"Accuracy on Test Set ({ML_TEST_YEAR_START}+)",
                value=f"{accuracy:.2%}" if accuracy and test_size > 0 else ("N/A" if test_size > 0 else "No Test Data")
            )
            m_col2.metric(label="Training Samples", value=train_size)
            m_col2.metric(label="Test Samples", value=test_size)

            if report and test_size > 0:
                with st.expander("Metrics (Test Set)"):
                    report_df = pd.DataFrame(report).transpose().round(3)
                    st.dataframe(report_df[['precision', 'recall', 'f1-score', 'support']])
            elif train_size > 0 and test_size == 0:
                st.info("Model trained, no test data.")
            elif train_size == 0:
                st.warning("No training data found.")

            st.subheader(f"Simulation: '{SIMULATE_ISSUE}'")
            if train_size == 0:
                st.warning("Model untrained, results unreliable.")
            if SIMULATE_ISSUE not in st.session_state.get('unique_issues', []):
                st.warning(f"Issue '{SIMULATE_ISSUE}' may not be in data.")
            
            if all_countries is not None and len(all_countries) > 0:
                simulation_df = pd.DataFrame({
                    'country_identifier': all_countries,
                    'issue': [SIMULATE_ISSUE] * len(all_countries)
                })
                try:
                    predicted_votes = model_pipeline.predict(simulation_df)
                    vote_counts = Counter(predicted_votes)
                    results_summary = pd.DataFrame.from_dict(
                        vote_counts,
                        orient='index',
                        columns=['Count']
                    ).sort_index()
                    
                    sim_col1, sim_col2 = st.columns([1, 2])
                    with sim_col1:
                        st.write("Predicted Dist:")
                        st.dataframe(results_summary)
                    with sim_col2:
                        fig_bar, ax_bar = plt.subplots()
                        results_summary.plot(kind='bar', ax=ax_bar, legend=None)
                        ax_bar.set_ylabel("Countries")
                        ax_bar.set_title(f"Sim Vote Distribution")
                        plt.xticks(rotation=0)
                        st.pyplot(fig_bar)
                        plt.close(fig_bar)
                    
                    with st.expander("Detailed Predictions"):
                        pred_df = pd.DataFrame({
                            'Country': all_countries,
                            'Predicted Vote': predicted_votes
                        }).sort_values('Country')
                        st.dataframe(pred_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Simulation error: {e}")
            else:
                st.warning("Cannot run simulation: No countries identified.")
        else:
            st.warning("Model training failed/skipped.")

st.sidebar.info("Analysis based on local CSV file.") 