# Quick Start Guide - UN Voting Network Analysis

## 🚀 Getting Started (3 Steps)

### Step 1: Launch the App
The app is **already running** at: **http://localhost:8080**

### Step 2: Load Data (Main Page)
1. Go to **http://localhost:8080** (main page)
2. Wait 5-10 seconds for data to load
3. You'll see "Analysis & Visualization" tabs appear
4. Data is now in memory

### Step 3: Explore Network Features
Click sidebar links to access new features:
- **🌐 Network Graph** - Interactive voting alliance network
- **💪 Soft Power** - Influence rankings over time

---

## 📊 Network Graph Page

### What It Shows
- **Nodes** = Countries (193 UN members)
- **Edges** (connections) = Voting similarity
- **Colors** = Voting blocs/clusters (auto-detected)
- **Node size** = Number of connections

### Controls

**Year Range** (Sidebar)
- Select time period for analysis
- Default: Last 10 years
- Tip: Shorter periods = faster rendering

**Similarity Threshold** (0.5 - 1.0)
- **0.5**: Dense network (many connections)
- **0.7**: Moderate (clear clusters) ← Default
- **0.9**: Sparse (only very similar countries)

**Network Layout**
- **Force**: Dynamic, spring-based (best for exploration)
- **Circular**: Countries arranged in circle
- **Kamada-Kawai**: Balanced, aesthetic
- **Spring**: Similar to force but different algorithm

**Color Nodes By**
- **Cluster**: Automatic voting bloc detection
- **Degree**: Number of connections (centrality)

**Timeline Animation**
- Shows how alliances evolved over time
- Use slider to scrub through years
- Press Play to animate

### How to Use

1. **Adjust Year Range** to your period of interest
2. **Set Similarity Threshold** (start with 0.7)
3. **Choose Layout** (force is best for first view)
4. **Interact with graph:**
   - **Drag** nodes to rearrange
   - **Hover** to see country details
   - **Zoom** with scroll wheel
   - **Pan** by dragging background

5. **Explore Communities:**
   - Scroll down to see detected voting blocs
   - Each cluster shows member countries

---

## 💪 Soft Power Page

### What It Shows
Rankings of UN member influence based on:
- **PageRank** (40%) - Overall network influence
- **Betweenness** (30%) - Broker/mediator power
- **Eigenvector** (20%) - Power through connections
- **Vote Swaying** (10%) - Ability to influence others

### Features

**Current Rankings Table**
- Top N countries (adjustable)
- Shows all metric components
- Sortable columns

**Historical Trends** (Optional)
- Click "Calculate historical soft power trends"
- Select start year and frequency
- View line chart of power shifts over time
- Download data as CSV

**Country Detail View**
- Select any country from dropdown
- See metric breakdown chart
- View rank and total connections

---

## 💡 Tips & Tricks

### Performance
- **First load**: Large dataset (352MB) takes 5-10 seconds
- **Subsequent loads**: Faster with caching
- **Tip**: Use smaller year ranges for faster network rendering

### Understanding the Network

**High Similarity (0.8+)**
- Countries vote together >80% of the time
- Strong alliance

**Medium Similarity (0.6-0.8)**
- Frequent alignment
- Same regional/political bloc

**Low Similarity (<0.6)**
- Different voting patterns
- Opposing interests

### Temporal Weighting
All calculations use **exponential decay**:
- Recent votes weighted more heavily
- Formula: `0.95^(years_ago)`
- Example: 2020 vote = 77% weight in 2024

---

## 🐛 Troubleshooting

### "No data loaded" Error
**Solution**: Navigate to main page (http://localhost:8080) first, wait for data to load, then click Network Graph in sidebar.

### Page Loads Slowly
**Solution**: 
- Use smaller year range (5 years instead of 20)
- Increase similarity threshold (0.8 instead of 0.7)
- Close timeline animation

### Network Graph Too Dense
**Solution**: Increase similarity threshold to 0.8 or 0.9

### Can't See My Country
**Solution**: 
- Check year range (country may not have voted in that period)
- Lower similarity threshold (country may be isolated)

---

## 🎯 Example Workflows

### Find Voting Blocs (2020-2024)
1. Set year range: 2020-2024
2. Similarity threshold: 0.7
3. Layout: Force
4. Color by: Cluster
5. Look for tight groups of same-colored nodes

### Track Power Shifts
1. Go to Soft Power page
2. Click "Calculate historical soft power trends"
3. Start year: 2000
4. Frequency: 2Y
5. Click Calculate
6. View trend lines

### Identify Alliance Breaks
1. Network page, year range: 2010-2015
2. Note clusters
3. Change year range: 2015-2020  
4. Compare cluster membership
5. Countries that switched = divergence

---

## 📚 Understanding the Metrics

### PageRank
- Like Google's algorithm
- Influenced by popular countries connected to you
- High = Central to network

### Betweenness
- How often you're on shortest path between others
- High = Bridge between different groups
- "Broker" power

### Eigenvector
- Power from having powerful connections
- High = Connected to other powerful countries
- "Elite network" metric

### Vote Swaying
- How often others vote like you
- Calculated from actual voting patterns
- High = Others follow your lead

---

## 🔧 Configuration

Want to customize? Edit `.env`:

```bash
# Temporal weighting (higher = more recent bias)
TEMPORAL_DECAY_FACTOR=0.95

# Network density
DEFAULT_SIMILARITY_THRESHOLD=0.7

# Soft power formula
PAGERANK_WEIGHT=0.4
BETWEENNESS_WEIGHT=0.3
EIGENVECTOR_WEIGHT=0.2
VOTE_SWAYING_WEIGHT=0.1
```

---



---

## 🎉 You're Ready!

Start exploring at: **http://localhost:8080**

**Quick Navigation:**
- Main Page → Load data
- Network Graph → See alliances
- Soft Power → Track influence
