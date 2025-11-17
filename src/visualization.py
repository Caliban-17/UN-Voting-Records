import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import logging
from sklearn.decomposition import PCA
from src.config import (
    DEFAULT_FIG_SIZE,
    DEFAULT_MARKER_SIZE,
    DEFAULT_ALPHA,
    PLOT_COLORS
)

logger = logging.getLogger(__name__)

def plot_cluster_vote_distribution(
    clusters: Dict[int, List[str]],
    df_filtered: pd.DataFrame,
    n_cols: int = 4
) -> Optional[plt.Figure]:
    """Plots vote distribution for each cluster."""
    try:
        n_clusters = len(clusters)
        n_rows = (n_clusters + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten()
        
        for i, (cluster_id, countries) in enumerate(sorted(clusters.items())):
            cluster_votes = df_filtered[df_filtered['country_identifier'].isin(countries)]['vote']
            vote_counts = cluster_votes.value_counts(normalize=True).reindex(['Y', 'N', 'A', ' '], fill_value=0)
            vote_counts.index = ['Yes', 'No', 'Abstain', 'Absent/NV']
            
            ax = axes[i]
            vote_counts.plot(kind='bar', ax=ax, rot=0)
            ax.set_title(f'Cluster {cluster_id + 1} ({len(countries)} members)')
            ax.set_ylabel('Proportion')
            
        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        return fig
        
    except Exception as e:
        logger.error(f"Error plotting cluster vote distribution: {str(e)}")
        return None

def plot_pca_projection(
    vote_matrix: pd.DataFrame,
    cluster_labels: np.ndarray,
    n_clusters: int
) -> Optional[plt.Figure]:
    """Plots PCA projection of voting patterns."""
    try:
        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(vote_matrix)
        
        pca_df = pd.DataFrame(
            data=pca_result,
            columns=['PC1', 'PC2'],
            index=vote_matrix.index
        )
        pca_df['Cluster'] = cluster_labels + 1
        pca_df['Cluster'] = pca_df['Cluster'].astype(str)
        
        fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
        sns.scatterplot(
            x='PC1', y='PC2',
            hue='Cluster',
            palette=sns.color_palette(PLOT_COLORS, n_colors=n_clusters),
            data=pca_df,
            ax=ax,
            s=DEFAULT_MARKER_SIZE,
            alpha=DEFAULT_ALPHA
        )
        
        ax.set_title('PCA of Voting Patterns')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        logger.error(f"Error plotting PCA projection: {str(e)}")
        return None

def plot_issue_salience(
    df_filtered: pd.DataFrame,
    n_top_issues: int = 10
) -> Optional[plt.Figure]:
    """Plots timeline of issue salience."""
    try:
        issue_counts = df_filtered.groupby(['year', 'issue'])['rcid'].nunique().reset_index(name='vote_count')
        top_issues = issue_counts.groupby('issue')['vote_count'].sum().nlargest(n_top_issues).index.tolist()
        top_issue_counts = issue_counts[issue_counts['issue'].isin(top_issues)]
        issue_timeline = top_issue_counts.pivot(index='year', columns='issue', values='vote_count').fillna(0)
        
        fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
        issue_timeline.plot(ax=ax)
        ax.set_title('Timeline of Issue Salience')
        ax.set_ylabel('Number of Votes')
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        logger.error(f"Error plotting issue salience: {str(e)}")
        return None

def plot_resolution_polarity(
    df_filtered: pd.DataFrame,
    issue: str
) -> Optional[plt.Figure]:
    """Plots vote distribution for a specific issue."""
    try:
        issue_data = df_filtered[df_filtered['issue'] == issue]
        if issue_data.empty:
            logger.warning(f"No data found for issue: {issue}")
            return None
            
        vote_counts = issue_data['vote'].value_counts(normalize=True).reindex(['Y', 'N', 'A', ' '], fill_value=0)
        vote_counts.index = ['Yes', 'No', 'Abstain', 'Absent/NV']
        
        fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
        vote_counts.plot(kind='bar', ax=ax, rot=0)
        ax.set_title(f"Vote Distribution for '{issue}'")
        ax.set_ylabel('Proportion of Votes')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        logger.error(f"Error plotting resolution polarity: {str(e)}")
        return None

def plot_entropy_distribution(
    entropy_scores: pd.Series,
    top_n: int = 10
) -> Optional[plt.Figure]:
    """Plots entropy distribution for countries."""
    try:
        entropy_df = entropy_scores.reset_index()
        entropy_df.columns = ['Country', 'Entropy']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot lowest entropy
        lowest_entropy = entropy_df.nsmallest(top_n, 'Entropy')
        sns.barplot(x='Entropy', y='Country', data=lowest_entropy, ax=ax1)
        ax1.set_title(f'Top {top_n} Most Consistent Voters')
        
        # Plot highest entropy
        highest_entropy = entropy_df.nlargest(top_n, 'Entropy')
        sns.barplot(x='Entropy', y='Country', data=highest_entropy, ax=ax2)
        ax2.set_title(f'Top {top_n} Least Consistent Voters')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        logger.error(f"Error plotting entropy distribution: {str(e)}")
        return None 