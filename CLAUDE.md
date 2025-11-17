# CLAUDE.md - UN Voting Records Analyzer

## Project Overview

The **UN Voting Records Analyzer** is a Streamlit-based web application for analyzing and visualizing United Nations General Assembly voting patterns. The application provides:

- Voting pattern analysis and visualization using hierarchical clustering
- Vote prediction using machine learning (Random Forest)
- Interactive visualizations (PCA projections, cluster analysis, entropy calculations)
- Issue salience timelines and resolution polarity analysis

**Tech Stack**: Python 3.x, Streamlit, pandas, scikit-learn, matplotlib, seaborn, scipy

## Repository Structure

```
UN-Voting-Records/
├── src/                          # Source code modules
│   ├── __init__.py              # Package initialization
│   ├── app.py                   # Streamlit web application (entry point)
│   ├── main.py                  # Core analysis logic & async model training
│   ├── config.py                # Configuration settings and constants
│   ├── data_processing.py       # Data loading, preprocessing, matrix creation
│   ├── model.py                 # ML models for vote prediction
│   └── visualization.py         # Plotting and charting functions
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── conftest.py              # Pytest configuration and fixtures
│   ├── test_data_processing.py  # Tests for data processing
│   ├── test_main.py             # Tests for main module
│   ├── test_model.py            # Tests for ML models
│   └── test_visualization.py    # Tests for visualizations
├── data/                         # Data directory (gitignored)
│   └── 2025_03_31_ga_voting_corr1.csv  # Expected CSV file location
├── requirements.txt              # Python dependencies
├── README.md                     # User-facing documentation
├── .gitignore                    # Git ignore rules
└── CLAUDE.md                     # This file - AI assistant guide

```

## Codebase Architecture

### Modular Design Philosophy

The codebase follows a **separation of concerns** pattern:

1. **config.py** - Single source of truth for all configuration
2. **data_processing.py** - Pure data transformation functions
3. **model.py** - Machine learning logic isolated from UI
4. **visualization.py** - Plotting functions with consistent signatures
5. **app.py** - Streamlit UI layer (presentation only)
6. **main.py** - CLI/script entry point for batch processing

### Key Design Patterns

- **Configuration-driven**: All magic numbers, paths, and settings in `config.py`
- **Type hints**: Functions use Python type hints for clarity
- **Logging**: Comprehensive logging using Python's `logging` module
- **Error handling**: Try-except blocks with informative error messages
- **Async/batch processing**: Memory-efficient model training with batch processing

## Module Descriptions

### src/config.py
**Purpose**: Central configuration file

**Key Constants**:
- `UN_VOTES_CSV_PATH`: Path to the CSV data file (data/2025_03_31_ga_voting_corr1.csv)
- `VOTE_MAP`: Mapping from vote values to numeric scores (Yes=1, No=-1, Abstain/Other/Absent=0)
- `COLUMN_RENAME_MAP`: Standardizes CSV column names
- `ESSENTIAL_COLUMNS`: Required columns for analysis
- `OPTIONAL_COLUMNS`: Additional context columns
- `DEFAULT_*`: Default values for clustering, plotting, etc.
- `PLOT_COLORS`: Visualization color schemes

**When to modify**: When adding new configuration parameters, default values, or file paths

### src/data_processing.py
**Purpose**: Data loading, cleaning, and transformation

**Key Functions**:
- `load_and_preprocess_data(file_path)`: Loads CSV, handles malformed rows, standardizes columns
  - Returns: (DataFrame, list of unique issues)
  - Handles: Date parsing, vote mapping, column renaming, missing values

- `create_vote_matrix(df, start_year, end_year)`: Creates country x resolution voting matrix
  - Returns: (vote_matrix DataFrame, country_list, filtered_df)
  - Creates pivot table with countries as rows, resolution IDs as columns

- `calculate_country_entropy(df)`: Calculates Shannon entropy for voting predictability
  - Higher entropy = more varied/unpredictable voting

- `get_issue_statistics(df)`: Generates statistics per issue
  - Returns: Dict with vote_counts, distributions, entropy

**Error Handling Pattern**:
```python
try:
    # Main logic
    logger.info("Starting operation...")
    # ... operation code ...
except Exception as e:
    logger.error(f"Error in operation: {str(e)}")
    return None  # or appropriate default
```

### src/model.py
**Purpose**: Machine learning for vote prediction

**Key Functions**:
- `train_vote_predictor(df, train_yr_end, test_yr_start)`: Trains Random Forest classifier
  - Features: country_identifier, issue (one-hot encoded)
  - Returns: (model_pipeline, accuracy, classification_report, countries, train_size, test_size)
  - Uses `ColumnTransformer` for preprocessing

- `predict_votes(model, countries, issue)`: Predicts votes for given issue
  - Returns: (vote_counts summary, detailed_predictions DataFrame)

- `save_model(model, filepath)`: Persists model to disk using joblib
- `load_model(filepath)`: Loads saved model

**ML Pipeline**:
1. OneHotEncoder for categorical features (country_identifier, issue)
2. RandomForestClassifier (100 estimators, balanced class weights)
3. Training on years <= train_yr_end
4. Testing on years >= test_yr_start

### src/visualization.py
**Purpose**: Matplotlib/seaborn visualization functions

**Key Functions**:
- `plot_cluster_vote_distribution(clusters, df_filtered, n_cols=4)`: Bar charts per cluster
- `plot_pca_projection(vote_matrix, cluster_labels, n_clusters)`: 2D PCA scatter plot
- `plot_issue_salience(df_filtered, n_top_issues=10)`: Timeline of issue frequency
- `plot_resolution_polarity(df_filtered, issue)`: Vote distribution for specific issue
- `plot_entropy_distribution(entropy_scores, top_n=10)`: Most/least consistent voters

**Visualization Standards**:
- All functions return `Optional[plt.Figure]` (None on error)
- Use constants from `config.py` for sizes, colors, alpha
- Include error handling with logging
- Call `plt.tight_layout()` before returning
- Close figures after use in Streamlit: `plt.close(fig)`

### src/main.py
**Purpose**: Core analysis logic and CLI entry point

**Key Components**:
- `train_vote_predictor_async(df, batch_size=5000)`: Memory-efficient batch training with Logistic Regression
  - Uses `CountVectorizer` for text features
  - Trains incrementally using warm_start
  - Monitors memory usage with `psutil`

- `preprocess_for_similarity(df, start_year, end_year)`: Wrapper for vote matrix creation
- `calculate_similarity(vote_matrix)`: Computes cosine similarity between countries
- `perform_clustering(similarity_matrix, n_clusters, country_list)`: Hierarchical clustering
  - Uses `AgglomerativeClustering` with precomputed distance

- `load_un_votes_data(filepath)`: Custom CSV loader with validation
- `main()`: Async main function for testing/batch processing

**Async/Memory Management**:
- Global `shutdown_event` for graceful termination
- Garbage collection (`gc.collect()`) after memory-intensive operations
- Progress logging every N batches
- Numerical stability: `np.clip()` to prevent division by zero

### src/app.py
**Purpose**: Streamlit web interface

**Structure**:
1. **Session State Management**: Caches loaded data, issues, flags
2. **Sidebar**: Data source info, analysis parameters (year ranges, n_clusters, split year, issue selection)
3. **Tab 1 - Analysis & Visualization**:
   - Clustering results with vote distribution bar charts
   - PCA scatter plot showing voting similarity
   - Issue salience timeline (line chart)
   - Resolution polarity for selected issue
   - Vote entropy distribution (most/least consistent voters)
4. **Tab 2 - Prediction**:
   - Model performance metrics (accuracy, precision, recall, F1)
   - Simulated vote predictions for selected issue
   - Detailed predictions per country

**Streamlit Patterns**:
- `st.session_state` for data persistence across reruns
- `st.cache_data` (implicit via session_state) to avoid reloading
- `st.columns()` for layout
- `st.expander()` for collapsible details
- `st.pyplot(fig)` + `plt.close(fig)` to display and cleanup matplotlib figures

### tests/
**Purpose**: Comprehensive test coverage using pytest

**Test Organization**:
- `conftest.py`: Shared fixtures and test configuration
- `test_data_processing.py`: Tests for data loading, vote matrix, entropy, statistics
- `test_model.py`: Tests for model training, prediction, save/load
- `test_visualization.py`: Tests for plotting functions
- `test_main.py`: Integration tests for main module

**Testing Conventions**:
- Use `pytest` framework
- Test functions named `test_<functionality>()`
- Use `tempfile` for temporary CSV files in tests
- Test happy paths, edge cases, and error conditions
- Assertions check return types, shapes, and values
- Mock heavy dependencies where appropriate

## Data Requirements

### Expected CSV Format

The application expects a CSV file at `data/2025_03_31_ga_voting_corr1.csv` with these columns:

**Source Columns** (as they appear in CSV):
- `undl_id`: UN document library ID (renamed to `rcid`)
- `ms_code`: Member state code (renamed to `country_code`)
- `ms_name`: Member state name (renamed to `country_name`)
- `ms_vote`: Vote value - 'Y', 'N', 'A', ' ' (renamed to `vote`)
- `date`: Vote date (YYYY-MM-DD format)
- `session`: UN session number
- `title`: Resolution title (renamed to `descr`)
- `subjects`: Issue/subject classification (renamed to `issue`)
- `resolution`: Resolution number
- `agenda_title`: Agenda item title (renamed to `agenda`)
- `undl_link`: Link to UN document

**Processed Columns** (after transformation):
- `rcid`: Resolution/vote ID (unique identifier)
- `country_code`: Country code (e.g., 'US', 'UK')
- `country_name`: Full country name
- `country_identifier`: Derived from country_code (used for analysis)
- `vote`: Standardized vote value ('Yes', 'No', 'Abstain', 'Absent', 'Other')
- `date`: Parsed datetime
- `year`: Extracted year (for filtering)
- `issue`: Issue category/subject
- `descr`: Resolution description

### Vote Value Mapping

```python
# In CSV (ms_vote column)
'Y' -> 'Yes'  (numeric: 1)
'N' -> 'No'   (numeric: -1)
'A' -> 'Abstain' (numeric: 0)
' ' -> 'Absent' (numeric: 0)
Other -> 'Other' (numeric: 0)
```

## Development Workflows

### Initial Setup

```bash
# Clone the repository
git clone <repository-url>
cd UN-Voting-Records

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Ensure data file exists
mkdir -p data
# Place 2025_03_31_ga_voting_corr1.csv in data/ directory
```

### Running the Application

```bash
# Activate virtual environment
source venv/bin/activate

# Run Streamlit app
streamlit run src/app.py

# Or run main.py for CLI testing
python src/main.py
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_data_processing.py

# Run specific test function
pytest tests/test_data_processing.py::test_create_vote_matrix

# Run with verbose output
pytest -v

# Run with print statements visible
pytest -s
```

### Adding New Features

#### 1. Adding a New Visualization

**Steps**:
1. Add configuration constants to `src/config.py` if needed
2. Create function in `src/visualization.py`:
   ```python
   def plot_new_chart(data: pd.DataFrame, param: int) -> Optional[plt.Figure]:
       """Description of what this plots."""
       try:
           fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
           # Plotting logic
           plt.tight_layout()
           return fig
       except Exception as e:
           logger.error(f"Error plotting new chart: {str(e)}")
           return None
   ```
3. Add tests in `tests/test_visualization.py`
4. Import and use in `src/app.py`:
   ```python
   from src.visualization import plot_new_chart

   fig = plot_new_chart(data, param)
   if fig:
       st.pyplot(fig)
       plt.close(fig)
   ```

#### 2. Adding a New Data Processing Function

**Steps**:
1. Add function to `src/data_processing.py`:
   ```python
   def process_new_feature(df: pd.DataFrame) -> pd.DataFrame:
       """Description of processing."""
       try:
           logger.info("Processing new feature...")
           # Processing logic
           return processed_df
       except Exception as e:
           logger.error(f"Error in process_new_feature: {str(e)}")
           return pd.DataFrame()
   ```
2. Add tests in `tests/test_data_processing.py`:
   ```python
   def test_process_new_feature():
       sample_data = pd.DataFrame({...})
       result = process_new_feature(sample_data)
       assert isinstance(result, pd.DataFrame)
       assert len(result) > 0
       # More assertions
   ```
3. Import and use in main logic

#### 3. Adding a New ML Model

**Steps**:
1. Add model function to `src/model.py`:
   ```python
   def train_new_model(df: pd.DataFrame, **params) -> Tuple[Pipeline, float]:
       """Train new ML model."""
       # Model training logic
       return model_pipeline, accuracy
   ```
2. Add tests in `tests/test_model.py`
3. Update `src/app.py` to include new model option

## Code Style and Conventions

### Python Style

- **PEP 8 compliant**: Follow PEP 8 style guide
- **Type hints**: Use type hints for function signatures
  ```python
  def function_name(param1: str, param2: int) -> Optional[pd.DataFrame]:
  ```
- **Docstrings**: Include docstrings for modules, classes, and functions
  ```python
  def example_function(data: pd.DataFrame) -> pd.DataFrame:
      """
      Brief description of function.

      Args:
          data (pd.DataFrame): Description of data parameter

      Returns:
          pd.DataFrame: Description of return value
      """
  ```
- **Naming conventions**:
  - Variables/functions: `snake_case`
  - Constants: `UPPER_CASE`
  - Classes: `PascalCase`
  - Private/internal: `_leading_underscore`

### Logging Practices

```python
# Module-level logger
logger = logging.getLogger(__name__)

# Usage patterns
logger.info("Informational message about normal operation")
logger.warning("Warning about potential issue")
logger.error(f"Error occurred: {str(e)}")
logger.debug("Detailed debug information")
```

### Error Handling Pattern

```python
def process_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Process data with error handling."""
    try:
        logger.info("Starting data processing...")

        # Validation
        if df is None or df.empty:
            logger.warning("Empty dataframe provided")
            return None

        # Main logic
        result = df.copy()
        # ... processing steps ...

        logger.info(f"Processing complete: {len(result)} rows")
        return result

    except KeyError as e:
        logger.error(f"Missing column: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in process_data: {str(e)}")
        return None
```

### Import Organization

```python
# Standard library imports
import os
import sys
import logging
from typing import Tuple, Optional, List, Dict

# Third-party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Local imports
from src.config import VOTE_MAP, DEFAULT_N_CLUSTERS
from src.data_processing import load_and_preprocess_data
```

## Common Patterns and Anti-Patterns

### ✅ DO: Use Configuration Constants

```python
# Good
from src.config import DEFAULT_N_CLUSTERS
n_clusters = DEFAULT_N_CLUSTERS

# Bad
n_clusters = 10  # Magic number
```

### ✅ DO: Return Consistent Types

```python
# Good
def get_data() -> Optional[pd.DataFrame]:
    if error:
        return None
    return df

# Bad
def get_data():
    if error:
        return False  # Inconsistent type
    return df
```

### ✅ DO: Log Before and After Operations

```python
logger.info("Starting clustering...")
clusters = perform_clustering(data)
logger.info(f"Clustering complete: {len(clusters)} clusters found")
```

### ✅ DO: Validate Inputs

```python
def process(df: pd.DataFrame):
    if df is None or df.empty:
        logger.warning("Empty dataframe provided")
        return None
    # Continue processing
```

### ❌ DON'T: Mix UI and Logic

```python
# Bad - Streamlit code in data processing
def load_data():
    st.write("Loading data...")  # Don't do this

# Good - Separate concerns
def load_data():
    logger.info("Loading data...")
    # Pure data loading logic
```

### ❌ DON'T: Hardcode Paths

```python
# Bad
df = pd.read_csv("data/votes.csv")

# Good
from src.config import UN_VOTES_CSV_PATH
df = pd.read_csv(UN_VOTES_CSV_PATH)
```

### ❌ DON'T: Ignore Errors Silently

```python
# Bad
try:
    result = risky_operation()
except:
    pass  # Silent failure

# Good
try:
    result = risky_operation()
except Exception as e:
    logger.error(f"Operation failed: {str(e)}")
    return None
```

## Testing Guidelines

### Test Structure

```python
def test_feature_name():
    """Test description."""
    # Arrange - Set up test data
    sample_data = pd.DataFrame({...})

    # Act - Execute the function
    result = function_under_test(sample_data)

    # Assert - Verify results
    assert isinstance(result, pd.DataFrame)
    assert len(result) == expected_length
    assert 'expected_column' in result.columns
```

### Test Coverage Requirements

- **Happy path**: Test normal, expected behavior
- **Edge cases**: Empty data, single row, boundary values
- **Error conditions**: Invalid inputs, missing columns, type errors
- **Integration**: Test component interactions

### Using Fixtures (conftest.py)

```python
# tests/conftest.py
@pytest.fixture
def sample_vote_data():
    """Fixture providing sample voting data."""
    return pd.DataFrame({
        'country_identifier': ['US', 'UK', 'FR'],
        'vote': ['Y', 'N', 'A'],
        'year': [2020, 2020, 2020]
    })

# tests/test_something.py
def test_with_fixture(sample_vote_data):
    result = process_votes(sample_vote_data)
    assert len(result) == 3
```

## Performance Considerations

### Memory Management

- **Batch processing**: Used in `train_vote_predictor_async()` for large datasets
- **Garbage collection**: Call `gc.collect()` after memory-intensive operations
- **Sparse matrices**: Use scipy sparse matrices for large vote matrices
- **DataFrame copying**: Use `.copy()` when modifying to avoid SettingWithCopyWarning

### Optimization Tips

1. **Filter early**: Filter DataFrames as early as possible
   ```python
   # Good
   df_filtered = df[df['year'] >= 2020]
   processed = expensive_operation(df_filtered)

   # Bad
   processed = expensive_operation(df)
   filtered = processed[processed['year'] >= 2020]
   ```

2. **Use vectorized operations**: Prefer pandas/numpy over loops
   ```python
   # Good
   df['vote_numeric'] = df['vote'].map(VOTE_MAP)

   # Bad
   for i, row in df.iterrows():
       df.loc[i, 'vote_numeric'] = VOTE_MAP.get(row['vote'])
   ```

3. **Lazy loading**: Only load data when needed (used in Streamlit session_state)

4. **Numerical stability**: Add epsilon to prevent division by zero
   ```python
   similarity_matrix = vote_matrix.replace(0, 1e-10)
   ```

## Troubleshooting Guide

### Common Issues

#### Issue: Data file not found
**Symptoms**: FileNotFoundError when running app
**Solution**:
1. Ensure data directory exists: `mkdir -p data`
2. Place CSV file in `data/2025_03_31_ga_voting_corr1.csv`
3. Check file permissions: `ls -l data/`

#### Issue: Missing columns error
**Symptoms**: ValueError about missing columns
**Solution**:
1. Verify CSV has all required source columns (see Data Requirements section)
2. Check column names match exactly (case-sensitive)
3. Review `COLUMN_RENAME_MAP` in config.py

#### Issue: Memory errors during model training
**Symptoms**: MemoryError or system slowdown
**Solution**:
1. Reduce `batch_size` in `train_vote_predictor_async()`
2. Filter data to smaller year range
3. Use smaller sample of data for development

#### Issue: Clustering produces unexpected results
**Symptoms**: All countries in one cluster or too many singleton clusters
**Solution**:
1. Check year range has sufficient data
2. Adjust `n_clusters` parameter
3. Verify vote matrix has non-zero variance
4. Check for NaN values in vote matrix

#### Issue: Tests failing
**Symptoms**: pytest failures
**Solution**:
1. Run `pytest -v` for detailed output
2. Check if dependencies are installed: `pip install -r requirements.txt`
3. Verify test data paths are correct
4. Check for environment-specific issues (paths, data files)

## Git Workflow

### Branch Strategy

- `main`: Production-ready code
- `claude/claude-md-*`: AI assistant feature branches
- Feature branches: Short-lived branches for specific features

### Commit Guidelines

- **Commit messages**: Descriptive, present tense
  - Good: "Add entropy visualization to analysis tab"
  - Bad: "fixed stuff"
- **Commit size**: Atomic commits (one logical change per commit)
- **Before committing**:
  1. Run tests: `pytest`
  2. Check code style
  3. Update documentation if needed

### Pull Request Process

1. Create feature branch from main
2. Make changes and commit
3. Push to remote: `git push -u origin branch-name`
4. Create pull request with description
5. Ensure tests pass
6. Request review if working with team

## AI Assistant Guidelines

### When Analyzing This Codebase

1. **Start with config.py**: Understand all constants and settings
2. **Read module docstrings**: Each file has clear purpose
3. **Follow imports**: Trace function calls through modules
4. **Check tests**: Tests show expected behavior and edge cases

### When Making Changes

1. **Preserve patterns**: Follow existing code style and conventions
2. **Update tests**: Add/modify tests for any code changes
3. **Update docs**: Update CLAUDE.md if architecture changes
4. **Log appropriately**: Add logging for new operations
5. **Handle errors**: Use try-except with informative logging

### When Adding Features

1. **Configuration first**: Add any constants to config.py
2. **Core logic**: Implement in appropriate module (data_processing, model, visualization)
3. **Tests**: Write tests before or alongside implementation
4. **UI integration**: Add to app.py only after core logic is tested
5. **Documentation**: Update this file if adding major functionality

### Reference Locations for Common Tasks

| Task | Primary File(s) | Key Functions/Sections |
|------|----------------|----------------------|
| Modify data loading | `data_processing.py` | `load_and_preprocess_data()` |
| Change vote mapping | `config.py`, `data_processing.py` | `VOTE_MAP` constant |
| Add clustering method | `main.py` | `perform_clustering()` |
| Change ML algorithm | `model.py` | `train_vote_predictor()` |
| Add chart type | `visualization.py` | Create new `plot_*()` function |
| Modify UI layout | `app.py` | Streamlit sections, tabs |
| Add configuration | `config.py` | Add constant |
| Fix data issues | `data_processing.py` | Preprocessing functions |
| Memory optimization | `main.py` | `train_vote_predictor_async()` |
| Add tests | `tests/test_*.py` | Create `test_*()` function |

### Quick Command Reference

```bash
# Development
streamlit run src/app.py          # Run web app
python src/main.py                # Run CLI version
pytest                            # Run all tests
pytest -v                         # Verbose test output
pytest --cov=src                  # Test with coverage

# Environment
python -m venv venv               # Create virtual environment
source venv/bin/activate          # Activate (Linux/Mac)
venv\Scripts\activate             # Activate (Windows)
pip install -r requirements.txt   # Install dependencies

# Git
git status                        # Check status
git add .                         # Stage changes
git commit -m "message"           # Commit
git push -u origin branch-name    # Push to remote
```

## Additional Resources

- **Streamlit Docs**: https://docs.streamlit.io/
- **pandas Docs**: https://pandas.pydata.org/docs/
- **scikit-learn Docs**: https://scikit-learn.org/stable/
- **pytest Docs**: https://docs.pytest.org/
- **matplotlib Docs**: https://matplotlib.org/stable/contents.html

## Changelog

### Current State (2025-11-17)
- Modular architecture with separate modules for config, data, models, visualization
- Streamlit web interface with two main tabs (Analysis & Prediction)
- Machine learning vote prediction using Random Forest
- Hierarchical clustering for voting bloc identification
- Comprehensive test suite with pytest
- Memory-efficient batch processing for large datasets
- Configuration-driven design
- Comprehensive logging throughout

---

**Last Updated**: 2025-11-17
**Maintained by**: AI Assistant (Claude)
**Version**: 1.0
