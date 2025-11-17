# UN Voting Records Analyzer

A Streamlit application for analyzing and visualizing United Nations voting patterns, predicting future votes, and identifying voting blocs.

## Features

- Voting pattern analysis and visualization
- Hierarchical clustering of voting blocs
- Vote prediction using machine learning
- Issue salience timeline
- Resolution polarity analysis
- Vote entropy calculation

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Ensure the data file is in the correct location:
   - Place `2025_03_31_ga_voting_corr1.csv` in the `data/` directory
2. Run the application:
   ```bash
   streamlit run src/main.py
   ```

## Data Requirements

The application expects a CSV file with the following columns:
- undl_id (renamed to rcid)
- ms_code (renamed to country_code)
- ms_name (renamed to country_name)
- ms_vote (renamed to vote)
- date
- session
- title (renamed to descr)
- subjects (renamed to issue)
- resolution
- agenda_title (renamed to agenda)
- undl_link

## Project Structure

```
.
├── data/               # Data files
├── src/               # Source code
├── tests/             # Test files
├── venv/              # Virtual environment
├── requirements.txt   # Dependencies
└── README.md          # Documentation
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 