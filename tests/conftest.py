import pytest
from unittest.mock import MagicMock

@pytest.fixture(autouse=True)
def mock_streamlit():
    """Mock streamlit to prevent it from running during tests."""
    mock_st = MagicMock()
    
    # Mock common streamlit functions
    mock_st.write = MagicMock()
    mock_st.error = MagicMock()
    mock_st.warning = MagicMock()
    mock_st.info = MagicMock()
    mock_st.success = MagicMock()
    mock_st.caption = MagicMock()
    mock_st.markdown = MagicMock()
    mock_st.metric = MagicMock()
    mock_st.columns = MagicMock(return_value=[MagicMock()])
    mock_st.expander = MagicMock()
    mock_st.pyplot = MagicMock()
    mock_st.line_chart = MagicMock()
    mock_st.bar_chart = MagicMock()
    mock_st.dataframe = MagicMock()
    mock_st.slider = MagicMock(return_value=10)
    mock_st.number_input = MagicMock(return_value=2020)
    mock_st.selectbox = MagicMock(return_value="Issue 1")
    
    # Create a context manager that returns the mock
    class MockExpander:
        def __enter__(self):
            return mock_st
        def __exit__(self, *args):
            pass
    mock_st.expander.return_value = MockExpander()
    
    # Patch streamlit
    import sys
    sys.modules['streamlit'] = mock_st
    return mock_st 