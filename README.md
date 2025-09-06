# Recommendation System Project

## Setup Instructions

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate    # On Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model and generate artifacts:
   ```bash
   python run_pipeline.py
   ```

4. Start the Streamlit app:
   ```bash
   streamlit run src/app.py
   ```

## Dataset
- By default, the project uses a synthetic dataset.
- To use MovieLens 100k, place `u.data` at `data/ml-100k/u.data`.
