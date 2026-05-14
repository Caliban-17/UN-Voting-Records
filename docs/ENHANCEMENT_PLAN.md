# UN Voting Records - Enhancement Plan

## 1. Missing API Endpoints to Add

### 1.1 Vote Prediction Endpoint

Add to `web_app.py`:

```python
@app.route('/api/prediction/train', methods=['POST'])
def train_prediction_model():
    """Train the vote prediction model"""
    if df_global is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    try:
        data = request.json or {}
        train_end_year = int(data.get('train_end_year', 2018))
        test_start_year = int(data.get('test_start_year', 2019))
        
        from src.model import train_vote_predictor
        
        model, accuracy, report, countries, train_size, test_size = train_vote_predictor(
            df_global, train_end_year, test_start_year
        )
        
        # Store model in app context for predictions
        app.config['PREDICTION_MODEL'] = model
        app.config['ALL_COUNTRIES'] = countries
        
        return jsonify({
            'accuracy': accuracy,
            'report': report,
            'train_samples': train_size,
            'test_samples': test_size,
            'countries_available': len(countries) if countries is not None else 0
        })
        
    except Exception as e:
        logger.error(f"Model training error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/prediction/predict', methods=['POST'])
def predict_votes():
    """Predict votes for an issue"""
    model = app.config.get('PREDICTION_MODEL')
    countries = app.config.get('ALL_COUNTRIES')
    
    if model is None:
        return jsonify({'error': 'Model not trained. Call /api/prediction/train first'}), 400
    
    try:
        data = request.json
        issue = data.get('issue', 'Human rights')
        selected_countries = data.get('countries', list(countries)[:50])
        
        from src.model import predict_votes as do_predict
        
        summary, detailed = do_predict(model, selected_countries, issue)
        
        return jsonify({
            'summary': summary.to_dict('records') if summary is not None else [],
            'predictions': detailed.to_dict('records') if detailed is not None else []
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/prediction/issues', methods=['GET'])
def get_available_issues():
    """Get list of issues for prediction"""
    if df_global is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    issues = df_global['issue'].value_counts().head(50).to_dict()
    return jsonify({'issues': issues})
```

### 1.2 Divergence Analysis Endpoint

```python
@app.route('/api/analysis/divergence', methods=['POST'])
def analyze_divergence():
    """Analyze voting divergence between countries or within clusters"""
    if df_global is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    try:
        data = request.json
        start_year, end_year = validate_year_range(data)
        country_a = data.get('country_a')
        country_b = data.get('country_b')
        
        # Build similarity matrix
        vote_matrix, country_list, df_filtered = preprocess_for_similarity(
            df_global, start_year, end_year
        )
        similarity_matrix = calculate_similarity(vote_matrix)
        
        from src.divergence_analysis import DivergenceDetector
        
        detector = DivergenceDetector(df_filtered, similarity_matrix)
        
        if country_a and country_b:
            # Pairwise divergence
            anomalies = detector.detect_vote_anomalies(country_a, country_b)
            return jsonify({
                'type': 'pairwise',
                'country_a': country_a,
                'country_b': country_b,
                'anomalies': anomalies[:20]  # Top 20 divergent votes
            })
        else:
            return jsonify({'error': 'Provide country_a and country_b'}), 400
            
    except Exception as e:
        logger.error(f"Divergence analysis error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/analysis/country-compare', methods=['POST'])
def compare_countries():
    """Compare two countries' voting patterns"""
    if df_global is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    try:
        data = request.json
        start_year, end_year = validate_year_range(data)
        country_a = data.get('country_a')
        country_b = data.get('country_b')
        
        if not country_a or not country_b:
            return jsonify({'error': 'Provide both country_a and country_b'}), 400
        
        # Filter data
        df_period = df_global[
            (df_global['year'] >= start_year) & 
            (df_global['year'] <= end_year)
        ]
        
        # Get votes for both countries
        votes_a = df_period[df_period['country_identifier'] == country_a]
        votes_b = df_period[df_period['country_identifier'] == country_b]
        
        # Merge on resolution
        merged = votes_a.merge(
            votes_b, on='rcid', suffixes=('_a', '_b')
        )
        
        # Calculate agreement rate
        agreements = (merged['vote_a'] == merged['vote_b']).sum()
        total = len(merged)
        agreement_rate = agreements / total if total > 0 else 0
        
        # Vote breakdown
        vote_comparison = {
            'both_yes': ((merged['vote_a'] == 1) & (merged['vote_b'] == 1)).sum(),
            'both_no': ((merged['vote_a'] == -1) & (merged['vote_b'] == -1)).sum(),
            'both_abstain': ((merged['vote_a'] == 0) & (merged['vote_b'] == 0)).sum(),
            'a_yes_b_no': ((merged['vote_a'] == 1) & (merged['vote_b'] == -1)).sum(),
            'a_no_b_yes': ((merged['vote_a'] == -1) & (merged['vote_b'] == 1)).sum(),
        }
        
        # Disagreements by issue
        disagreements = merged[merged['vote_a'] != merged['vote_b']]
        issues_disagreed = disagreements.groupby('issue_a').size().nlargest(10).to_dict()
        
        return jsonify({
            'country_a': country_a,
            'country_b': country_b,
            'period': f"{start_year}-{end_year}",
            'total_common_votes': int(total),
            'agreement_rate': round(agreement_rate, 3),
            'vote_breakdown': vote_comparison,
            'top_disagreement_issues': issues_disagreed
        })
        
    except Exception as e:
        logger.error(f"Country comparison error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/analysis/country-detail/<country_code>', methods=['GET'])
def get_country_detail(country_code):
    """Get detailed analysis for a single country"""
    if df_global is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    try:
        start_year = int(request.args.get('start_year', 2010))
        end_year = int(request.args.get('end_year', 2020))
        
        df_period = df_global[
            (df_global['year'] >= start_year) & 
            (df_global['year'] <= end_year) &
            (df_global['country_identifier'] == country_code)
        ]
        
        if df_period.empty:
            return jsonify({'error': f'No data for {country_code}'}), 404
        
        # Vote distribution
        vote_dist = df_period['vote'].value_counts().to_dict()
        vote_labels = {1: 'Yes', -1: 'No', 0: 'Abstain'}
        vote_dist_labeled = {vote_labels.get(k, 'Other'): v for k, v in vote_dist.items()}
        
        # Votes by year
        yearly = df_period.groupby('year')['vote'].value_counts().unstack(fill_value=0)
        
        # Top issues voted on
        top_issues = df_period['issue'].value_counts().head(10).to_dict()
        
        # Calculate predictability (entropy)
        from scipy.stats import entropy
        vote_probs = df_period['vote'].value_counts(normalize=True)
        predictability = 1 - (entropy(vote_probs, base=2) / 2)  # Normalize to 0-1
        
        return jsonify({
            'country': country_code,
            'period': f"{start_year}-{end_year}",
            'total_votes': int(len(df_period)),
            'vote_distribution': vote_dist_labeled,
            'yearly_votes': yearly.to_dict(),
            'top_issues': top_issues,
            'predictability_score': round(predictability, 3),
            'yes_rate': round(vote_dist.get(1, 0) / len(df_period), 3),
            'no_rate': round(vote_dist.get(-1, 0) / len(df_period), 3),
            'abstain_rate': round(vote_dist.get(0, 0) / len(df_period), 3)
        })
        
    except Exception as e:
        logger.error(f"Country detail error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500
```

---

## 2. Visualization Improvements

### 2.1 Enhanced PCA with Cluster Colors and Labels

Update the `/api/visualization/pca` endpoint:

```python
@app.route('/api/visualization/pca', methods=['POST'])
def get_pca_plot():
    """Get enhanced PCA projection with cluster coloring"""
    # ... existing validation code ...
    
    # Perform PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(vote_matrix)
    
    # Get cluster assignments
    similarity_matrix = calculate_similarity(vote_matrix)
    clusters, _, cluster_labels = perform_clustering(
        similarity_matrix, 
        int(request.json.get('num_clusters', 8)), 
        country_list
    )
    
    # Create enhanced DataFrame
    pca_df = pd.DataFrame({
        'PC1': components[:, 0],
        'PC2': components[:, 1],
        'Country': country_list,
        'Cluster': [f'Bloc {l+1}' for l in cluster_labels]
    })
    
    # Create figure with better styling
    fig = px.scatter(
        pca_df, 
        x='PC1', 
        y='PC2',
        color='Cluster',
        hover_name='Country',
        text='Country',  # Add country labels
        title=f'Voting Alignment Map ({start_year}-{end_year})',
        template='plotly_white',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    # Improve layout
    fig.update_traces(
        textposition='top center',
        textfont_size=8,
        marker=dict(size=12, line=dict(width=1, color='white'))
    )
    
    fig.update_layout(
        xaxis_title=f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
        yaxis_title=f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
        legend_title='Voting Bloc',
        height=600,
        font=dict(family='Inter, sans-serif')
    )
    
    # Add annotation explaining the visualization
    fig.add_annotation(
        text="Countries closer together vote more similarly",
        xref="paper", yref="paper",
        x=0.5, y=-0.15,
        showarrow=False,
        font=dict(size=11, color='gray')
    )
    
    return jsonify(json.loads(fig.to_json()))
```

### 2.2 Enhanced Network Graph with Labels

```python
def plot_network_interactive_enhanced(
    graph: nx.Graph,
    layout: str = 'force',
    communities: Optional[Dict[int, List[str]]] = None,
    highlight_countries: Optional[List[str]] = None
) -> go.Figure:
    """Enhanced network visualization with country labels and better colors"""
    
    # Use a better color palette for communities
    community_colors = [
        '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
        '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe'
    ]
    
    # Calculate positions
    if layout == 'force':
        pos = nx.spring_layout(graph, k=2, iterations=100, seed=42)
    # ... other layouts ...
    
    # Create node trace with labels
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',  # Add text mode
        text=[node for node in graph.nodes()],  # Country codes as labels
        textposition='top center',
        textfont=dict(size=9, color='#333'),
        hoverinfo='text',
        hovertext=node_hover_text,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            colorscale='Viridis',
            line=dict(width=2, color='white'),
            opacity=0.9
        )
    )
    
    # Add legend for communities
    fig.update_layout(
        title=dict(
            text='UN Voting Alliance Network',
            subtitle=dict(text='Node size = number of connections, Color = voting bloc')
        ),
        annotations=[
            dict(
                text="<b>How to read:</b> Connected countries vote similarly (>70% agreement)",
                xref="paper", yref="paper",
                x=0, y=1.08, showarrow=False,
                font=dict(size=11)
            )
        ]
    )
    
    return fig
```

### 2.3 Soft Power Visual Ranking (Replace Table with Horizontal Bar)

```python
@app.route('/api/visualization/soft-power-ranking', methods=['POST'])
def get_soft_power_ranking_viz():
    """Get soft power as a visual ranking chart"""
    # ... existing soft power calculation ...
    
    # Create horizontal bar chart
    top_20 = soft_power_scores.head(20)
    
    fig = go.Figure()
    
    # Add bars with gradient colors
    fig.add_trace(go.Bar(
        y=top_20.index[::-1],  # Reverse for top-to-bottom
        x=top_20.values[::-1],
        orientation='h',
        marker=dict(
            color=top_20.values[::-1],
            colorscale='Blues',
            line=dict(width=1, color='#2c3e50')
        ),
        text=[f'{v:.3f}' for v in top_20.values[::-1]],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Soft Power Score: %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='🏆 Soft Power Rankings',
            subtitle=dict(text='Based on network centrality and vote influence')
        ),
        xaxis_title='Soft Power Score (0-1)',
        yaxis_title='',
        height=600,
        margin=dict(l=100),
        showlegend=False
    )
    
    # Add rank numbers
    for i, country in enumerate(top_20.index[::-1]):
        fig.add_annotation(
            x=-0.02, y=country,
            text=f"#{20-i}",
            showarrow=False,
            font=dict(size=10, color='gray'),
            xanchor='right'
        )
    
    return jsonify(json.loads(fig.to_json()))
```

---

## 3. New UI Components

### 3.1 Add Prediction Tab to HTML

```html
<!-- Add to tabs-nav -->
<div class="tab-link" data-tab="predictions">🔮 Predictions</div>
<div class="tab-link" data-tab="compare">🔍 Compare</div>

<!-- Predictions Tab -->
<div id="predictions" class="tab-pane">
    <div class="viz-grid">
        <!-- Model Training Card -->
        <div class="viz-card">
            <div class="viz-header">
                <span class="viz-title">Train Prediction Model</span>
            </div>
            <div class="prediction-controls">
                <div class="form-group">
                    <label>Training Data End Year</label>
                    <input type="number" id="trainEndYear" value="2018" min="1950" max="2019">
                </div>
                <div class="form-group">
                    <label>Test Data Start Year</label>
                    <input type="number" id="testStartYear" value="2019" min="1951" max="2020">
                </div>
                <button id="trainModelBtn" class="btn btn-primary">Train Model</button>
                <div id="modelMetrics" class="metrics-display"></div>
            </div>
        </div>
        
        <!-- Prediction Card -->
        <div class="viz-card">
            <div class="viz-header">
                <span class="viz-title">Predict Vote Outcomes</span>
            </div>
            <div class="prediction-controls">
                <div class="form-group">
                    <label>Select Issue</label>
                    <select id="predictionIssue" class="form-control">
                        <option>Loading issues...</option>
                    </select>
                </div>
                <button id="predictBtn" class="btn btn-primary" disabled>Predict Votes</button>
            </div>
            <div id="predictionResults"></div>
        </div>
        
        <!-- Prediction Visualization -->
        <div class="viz-card full-width">
            <div class="viz-header">
                <span class="viz-title">Predicted Vote Distribution</span>
            </div>
            <div id="predictionChart"></div>
        </div>
    </div>
</div>

<!-- Compare Tab -->
<div id="compare" class="tab-pane">
    <div class="compare-header">
        <div class="country-selector">
            <label>Country A</label>
            <select id="countryA" class="form-control country-select"></select>
        </div>
        <div class="vs-badge">VS</div>
        <div class="country-selector">
            <label>Country B</label>
            <select id="countryB" class="form-control country-select"></select>
        </div>
        <button id="compareBtn" class="btn btn-primary">Compare</button>
    </div>
    
    <div class="viz-grid">
        <div class="viz-card">
            <div class="viz-header">
                <span class="viz-title">Agreement Overview</span>
            </div>
            <div id="agreementGauge"></div>
        </div>
        <div class="viz-card">
            <div class="viz-header">
                <span class="viz-title">Vote Breakdown</span>
            </div>
            <div id="voteBreakdown"></div>
        </div>
        <div class="viz-card full-width">
            <div class="viz-header">
                <span class="viz-title">Key Disagreements</span>
            </div>
            <div id="disagreementList"></div>
        </div>
    </div>
</div>
```

---

## 4. CSS Enhancements for Professional Academic Tone

```css
/* Add to style.css */

/* Typography improvements */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+Pro:wght@400;600&display=swap');

:root {
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-serif: 'Source Serif Pro', Georgia, serif;
    --color-un-blue: #009edb;
    --color-diplomatic-navy: #1a365d;
    --color-soft-gold: #d4a574;
}

/* Academic section headers */
.section-title {
    font-family: var(--font-serif);
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--color-diplomatic-navy);
    border-bottom: 2px solid var(--color-un-blue);
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
}

/* Insight callouts */
.insight-box {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    border-left: 4px solid var(--color-un-blue);
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    border-radius: 0 6px 6px 0;
}

.insight-box h4 {
    font-size: 0.85rem;
    text-transform: uppercase;
    color: var(--color-un-blue);
    margin-bottom: 0.5rem;
    letter-spacing: 0.5px;
}

.insight-box p {
    font-family: var(--font-serif);
    color: #334155;
    line-height: 1.6;
}

/* Country comparison styles */
.compare-header {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 2rem;
    padding: 2rem;
    background: white;
    border-radius: 8px;
    margin-bottom: 2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

.vs-badge {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--color-diplomatic-navy);
    padding: 0.5rem 1rem;
    background: #f1f5f9;
    border-radius: 50%;
}

.country-selector {
    text-align: center;
}

.country-select {
    font-size: 1.1rem;
    padding: 0.75rem 1rem;
    min-width: 200px;
}

/* Metrics display */
.metrics-display {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-top: 1rem;
}

.metric-item {
    text-align: center;
    padding: 1rem;
    background: #f8fafc;
    border-radius: 6px;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--color-diplomatic-navy);
}

.metric-label {
    font-size: 0.8rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Soft power ranking styles */
.ranking-item {
    display: flex;
    align-items: center;
    padding: 0.75rem 1rem;
    border-bottom: 1px solid #e2e8f0;
    transition: background 0.2s;
}

.ranking-item:hover {
    background: #f8fafc;
}

.ranking-position {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--color-soft-gold);
    width: 40px;
}

.ranking-country {
    flex: 1;
    font-weight: 500;
}

.ranking-score {
    font-family: 'JetBrains Mono', monospace;
    color: #64748b;
}

.ranking-bar {
    height: 8px;
    background: linear-gradient(90deg, var(--color-un-blue) 0%, var(--color-soft-gold) 100%);
    border-radius: 4px;
    margin-top: 0.25rem;
}

/* Tooltip styles for explanations */
.info-tooltip {
    display: inline-block;
    width: 16px;
    height: 16px;
    background: #e2e8f0;
    border-radius: 50%;
    text-align: center;
    line-height: 16px;
    font-size: 10px;
    color: #64748b;
    cursor: help;
    margin-left: 4px;
}

.info-tooltip:hover::after {
    content: attr(data-tooltip);
    position: absolute;
    background: #1e293b;
    color: white;
    padding: 0.5rem 0.75rem;
    border-radius: 4px;
    font-size: 0.75rem;
    max-width: 250px;
    z-index: 100;
    margin-top: 1.5rem;
    margin-left: -100px;
}
```

---

## 5. JavaScript Enhancements

Add to `app.js`:

```javascript
// Prediction functionality
async function loadPredictionIssues() {
    const response = await axios.get('/api/prediction/issues');
    const select = document.getElementById('predictionIssue');
    select.innerHTML = Object.entries(response.data.issues)
        .map(([issue, count]) => `<option value="${issue}">${issue} (${count} votes)</option>`)
        .join('');
}

async function trainModel() {
    const btn = document.getElementById('trainModelBtn');
    btn.disabled = true;
    btn.textContent = 'Training...';
    
    try {
        const response = await axios.post('/api/prediction/train', {
            train_end_year: parseInt(document.getElementById('trainEndYear').value),
            test_start_year: parseInt(document.getElementById('testStartYear').value)
        });
        
        const data = response.data;
        document.getElementById('modelMetrics').innerHTML = `
            <div class="metrics-display">
                <div class="metric-item">
                    <div class="metric-value">${(data.accuracy * 100).toFixed(1)}%</div>
                    <div class="metric-label">Accuracy</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">${data.train_samples.toLocaleString()}</div>
                    <div class="metric-label">Training Samples</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">${data.test_samples.toLocaleString()}</div>
                    <div class="metric-label">Test Samples</div>
                </div>
            </div>
        `;
        
        document.getElementById('predictBtn').disabled = false;
        
    } catch (error) {
        alert('Training failed: ' + error.message);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Train Model';
    }
}

async function predictVotes() {
    const issue = document.getElementById('predictionIssue').value;
    
    try {
        const response = await axios.post('/api/prediction/predict', { issue });
        
        // Create pie chart of predictions
        const summary = response.data.summary;
        const fig = {
            data: [{
                type: 'pie',
                labels: summary.map(s => s.Vote === 1 ? 'Yes' : s.Vote === -1 ? 'No' : 'Abstain'),
                values: summary.map(s => s.Count),
                marker: {
                    colors: ['#22c55e', '#ef4444', '#f59e0b']
                },
                textinfo: 'label+percent',
                hole: 0.4
            }],
            layout: {
                title: `Predicted Votes: ${issue}`,
                showlegend: true,
                height: 400
            }
        };
        
        Plotly.newPlot('predictionChart', fig.data, fig.layout);
        
    } catch (error) {
        console.error('Prediction failed:', error);
    }
}

// Country comparison
async function compareCountries() {
    const countryA = document.getElementById('countryA').value;
    const countryB = document.getElementById('countryB').value;
    
    try {
        const response = await axios.post('/api/analysis/country-compare', {
            country_a: countryA,
            country_b: countryB,
            start_year: state.startYear,
            end_year: state.endYear
        });
        
        const data = response.data;
        
        // Agreement gauge
        Plotly.newPlot('agreementGauge', [{
            type: 'indicator',
            mode: 'gauge+number',
            value: data.agreement_rate * 100,
            title: { text: 'Agreement Rate' },
            gauge: {
                axis: { range: [0, 100] },
                bar: { color: '#009edb' },
                steps: [
                    { range: [0, 50], color: '#fee2e2' },
                    { range: [50, 75], color: '#fef3c7' },
                    { range: [75, 100], color: '#dcfce7' }
                ]
            }
        }], { height: 300 });
        
        // Vote breakdown
        const breakdown = data.vote_breakdown;
        Plotly.newPlot('voteBreakdown', [{
            type: 'bar',
            x: ['Both Yes', 'Both No', 'Both Abstain', `${countryA} Yes/${countryB} No`, `${countryA} No/${countryB} Yes`],
            y: [breakdown.both_yes, breakdown.both_no, breakdown.both_abstain, breakdown.a_yes_b_no, breakdown.a_no_b_yes],
            marker: { color: ['#22c55e', '#ef4444', '#f59e0b', '#8b5cf6', '#ec4899'] }
        }], { height: 300 });
        
    } catch (error) {
        console.error('Comparison failed:', error);
    }
}

// Add event listeners
document.getElementById('trainModelBtn')?.addEventListener('click', trainModel);
document.getElementById('predictBtn')?.addEventListener('click', predictVotes);
document.getElementById('compareBtn')?.addEventListener('click', compareCountries);
```

---

## 6. Summary of New Features

### Prediction Features (Currently Missing)
1. ✅ Train ML model on historical data
2. ✅ Predict votes for any issue
3. ✅ Show model accuracy metrics
4. ✅ Visualize predicted vote distribution

### Divergence Features (Currently Missing)
1. ✅ Detect anomalous votes between countries
2. ✅ Compare any two countries
3. ✅ Identify divisive issues
4. ✅ Track alliance changes over time

### Visualization Improvements
1. ✅ PCA with cluster colors and country labels
2. ✅ Network graph with country labels and better sizing
3. ✅ Soft power as visual ranking bar chart
4. ✅ Agreement gauge for country comparison
5. ✅ Better color palettes (UN blue, diplomatic tones)

### UX Improvements
1. ✅ Info tooltips explaining each metric
2. ✅ Insight callout boxes highlighting key findings
3. ✅ Better typography (serif for titles, sans for data)
4. ✅ Hover states and transitions
5. ✅ Loading states with progress indicators

