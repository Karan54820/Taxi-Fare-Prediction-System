import streamlit as st
import numpy as np
import joblib
import datetime
import pandas as pd

# Load model and scaler
model = joblib.load('xgb_model.joblib')
scaler = joblib.load('scaler.joblib')

st.set_page_config(page_title='NYC Taxi Fare Prediction', page_icon='🚕', layout='centered')

# Sidebar content
with st.sidebar:
    st.title('🚕 NYC Taxi Fare')
    st.markdown('''
    **NYC Taxi Fare Prediction**
    
    Predict your taxi fare in New York City with a few simple details!
    ''')
    st.info('''
    **How to use:**
    - Fill in your trip details in the main form.
    - Hover over the ⓘ icon for help on each field.
    - Click **Predict Fare** to get your estimate.
    ''')
    st.markdown('---')
    st.caption('Made with ❤️ by [Karan Yadav](https://github.com/Karan54820/Taxi-Fare-Prediction-System)')

# Custom background and header styling
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #232946 0%, #121629 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #232946 0%, #121629 100%);
    }
    .header-card {
        background: #232946;
        color: #eebf63;
        border-radius: 18px;
        box-shadow: 0 4px 24px 0 rgba(0,0,0,0.18);
        padding: 2rem 2rem 1rem 2rem;
        margin-bottom: 2rem;
        border: 2px solid #eebf63;
    }
    .form-card {
        background: #181c2fdd;
        color: #f4f4f4;
        border-radius: 16px;
        box-shadow: 0 2px 16px 0 rgba(0,0,0,0.18);
        padding: 2rem 2rem 1rem 2rem;
        margin-bottom: 2rem;
        border: 1.5px solid #eebf63;
    }
    .stButton>button {
        background-color: #0099ff;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 2em;
    }
    .stButton>button:hover {
        background-color: #007acc;
        color: white;
    }
    .stMarkdown, .stCaption, .stSubheader, .stTextInput, .stSelectbox, .stNumberInput, .stDateInput, .stTimeInput {
        color: #f4f4f4 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Attractive header section
st.markdown(
    """
    <div class="header-card">
        <h1 style="text-align:center; font-size:2.8rem; color:#0099ff; margin-bottom:0.2em;">🚕 NYC Taxi Fare Predictor</h1>
        <h3 style="text-align:center; color:#f4f4f4; font-weight:400; margin-top:0;">Get your New York City taxi fare in seconds!</h3>
        <p style="text-align:center; color:#f4f4f4; font-size:1.1rem; margin-top:0.5em;">
            Enter your trip details below and discover your estimated fare instantly.<br>
            <span style="color:#ffb300; font-weight:bold;">Fast</span> • <span style="color:#00b894; font-weight:bold;">Accurate</span> • <span style="color:#ff7675; font-weight:bold;">User-Friendly</span>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Optionally, add a taxi image/banner (royalty-free image URL)
st.image('https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=800&q=80', use_container_width=True, caption='NYC Yellow Taxi')

# Main form in a visually distinct card
with st.container():
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    st.caption('Enter your trip details:')

    feature_names = [
        'rate_code',
        'Pickup_longitude',
        'Pickup_latitude',
        'Dropoff_longitude',
        'Dropoff_latitude',
        'Passenger_count',
        'Trip_distance',
        'Extra',
        'Improvement_surcharge',
        'Trip_type',
        'pickup_datetime_day',
        'pickup_datetime_hour',
        'pickup_datetime_minute',
        'pickup_datetime_second',
        'pickup_datetime_weekday',
        'dropoff_datetime_day',
        'dropoff_datetime_hour',
        'dropoff_datetime_minute',
        'dropoff_datetime_second',
        'dropoff_datetime_weekday',
    ]

    feature_help = {
        'rate_code': 'Rate code assigned by the taxi meter (1=Standard, 2=JFK, 3=Newark, 4=Nassau/Westchester, 5=Negotiated, 6=Group Ride)',
        'Pickup_longitude': 'Longitude of the pickup location (e.g., -73.985428)',
        'Pickup_latitude': 'Latitude of the pickup location (e.g., 40.748817)',
        'Dropoff_longitude': 'Longitude of the dropoff location (e.g., -73.985428)',
        'Dropoff_latitude': 'Latitude of the dropoff location (e.g., 40.748817)',
        'Passenger_count': 'Number of passengers (1-8)',
        'Trip_distance': 'Distance of the trip in miles (e.g., 2.5)',
        'Extra': 'Extra charges (e.g., 0.5 for rush hour, 1.0 for overnight, 0 for none)',
        'Improvement_surcharge': 'Improvement surcharge (usually 0.3)',
        'Trip_type': 'Trip type (1=Street-hail, 2=Dispatch)',
    }

    rate_code_options = [
        ('Standard Rate (1)', 1),
        ('JFK (2)', 2),
        ('Newark (3)', 3),
        ('Nassau/Westchester (4)', 4),
        ('Negotiated Fare (5)', 5),
        ('Group Ride (6)', 6),
    ]
    passenger_count_options = [f'{i} passenger' if i == 1 else f'{i} passengers' for i in range(1, 9)]
    trip_type_options = [
        ('Street-hail (1)', 1),
        ('Dispatch (2)', 2),
    ]

    def get_inputs():
        values = []
        col1, col2 = st.columns(2)
        with col1:
            rate_code_label = st.selectbox('Rate Code', options=rate_code_options, format_func=lambda x: x[0], help=feature_help['rate_code'])
            values.append(rate_code_label[1])
            values.append(st.number_input('Pickup Longitude', value=-73.98, format='%.6f', help=feature_help['Pickup_longitude']))
            values.append(st.number_input('Pickup Latitude', value=40.75, format='%.6f', help=feature_help['Pickup_latitude']))
            passenger_count = st.selectbox('Passenger Count', options=list(range(1, 9)), format_func=lambda x: passenger_count_options[x-1], help=feature_help['Passenger_count'])
            values.append(passenger_count)
            values.append(st.number_input('Trip Distance (miles)', min_value=0.0, value=1.0, format='%.2f', help=feature_help['Trip_distance']))
            values.append(st.number_input('Extra', value=0.0, format='%.2f', help=feature_help['Extra']))
            values.append(st.number_input('Improvement Surcharge', value=0.3, format='%.2f', help=feature_help['Improvement_surcharge']))
            trip_type_label = st.selectbox('Trip Type', options=trip_type_options, format_func=lambda x: x[0], help=feature_help['Trip_type'])
            values.append(trip_type_label[1])
        with col2:
            values.append(st.number_input('Dropoff Longitude', value=-73.98, format='%.6f', help=feature_help['Dropoff_longitude']))
            values.append(st.number_input('Dropoff Latitude', value=40.75, format='%.6f', help=feature_help['Dropoff_latitude']))

        st.divider()
        st.subheader('Pickup Date & Time')
        pickup_date = st.date_input('Pickup Date', value=datetime.date.today(), help='Select the pickup date')
        pickup_time = st.time_input('Pickup Time', value=datetime.time(0, 0), help='Select the pickup time')
        values.append(pickup_date.day)
        values.append(pickup_time.hour)
        values.append(pickup_time.minute)
        values.append(pickup_time.second)
        values.append(pickup_date.weekday())

        st.divider()
        st.subheader('Dropoff Date & Time')
        dropoff_date = st.date_input('Dropoff Date', value=datetime.date.today(), help='Select the dropoff date')
        dropoff_time = st.time_input('Dropoff Time', value=datetime.time(0, 0), help='Select the dropoff time')
        values.append(dropoff_date.day)
        values.append(dropoff_time.hour)
        values.append(dropoff_time.minute)
        values.append(dropoff_time.second)
        values.append(dropoff_date.weekday())
        return values

    user_input = get_inputs()

    st.divider()

    if st.button('Predict Fare', use_container_width=True):
        X_df = pd.DataFrame([user_input], columns=feature_names)
        X_scaled = scaler.transform(X_df)
        fare_pred = model.predict(X_scaled)[0]
        st.success(f'Predicted Fare Amount: ${fare_pred:.2f}')
        st.balloons()
        st.info('This is an estimate. Actual fares may vary due to traffic, tolls, and other factors.')

    st.markdown("""
    ---
    <small>Made with ❤️ for NYC Taxi Fare Prediction</small>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True) 