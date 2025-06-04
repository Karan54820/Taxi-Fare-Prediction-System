# NYC Taxi Fare Prediction

This project predicts taxi fares in New York City using a machine learning pipeline and a user-friendly Streamlit web app.

## Features
- Trained XGBoost regression model for fare prediction
- Scaler for input normalization
- Streamlit web interface for easy predictions

## Setup
1. **Clone or download this repository.**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Ensure the following files are in the project directory:**
   - `xgb_model.joblib` (trained model)
   - `scaler.joblib` (fitted scaler)
   - `app.py` (Streamlit app)

## Running the App
Start the Streamlit app with:
```bash
streamlit run app.py
```
This will open the web interface in your browser.

## Usage
- Enter the required trip details in the form.
- Click **Predict Fare** to get the estimated fare amount.

## Notes
- The model expects features in the same order and format as used during training.
- For best results, use realistic values for NYC taxi trips.

## Requirements
See `requirements.txt` for all dependencies. 