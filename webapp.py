import numpy as np
import pickle
import streamlit as st 

# Load the model
model_path = 'performance_model.sav'  # Update the path to your model file
loaded_model = pickle.load(open(model_path, 'rb'))

# Creating a function for Prediction
def stock_prediction(input_data):
    # Changing the input_data to numpy array and converting to float
    input_data_as_numpy_array = np.asarray(input_data).astype(float)
    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    return prediction

def main():
    # Setting page title and background color
    st.set_page_config(page_title="Stock Price Prediction Web App", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(
        """
        <style>
            body {
                background-color: #f0f2f6;
            }
            .title {
                font-size: 36px;
                font-weight: bold;
                color: #2c3e50;
                padding-top: 20px;
                padding-bottom: 20px;
                text-align: center;
            }
            .input-label {
                font-size: 20px;
                color: #34495e;
                margin-bottom: 10px;
            }
            .predict-button {
                font-size: 24px;
                font-weight: bold;
                background-color: #27ae60;
                color: white;
                padding: 10px 20px;
                border-radius: 5px;
                margin-top: 20px;
                margin-bottom: 20px;
                cursor: pointer;
            }
            .prediction {
                font-size: 24px;
                font-weight: bold;
                color: #2980b9;
                text-align: center;
                margin-top: 20px;
            }
        </style>
        """
    )

    # Title
    st.markdown("<h1 class='title'>Stock Price Prediction Web App</h1>", unsafe_allow_html=True)
    
    # Getting the input data from the user
    st.markdown("<h2 class='input-label'>Enter the Following Parameters:</h2>", unsafe_allow_html=True)
    Simplemovingaverage_3 = st.text_input('Simple Moving Average 3', '0.0')
    Simplemovingaverage_4 = st.text_input('Simple Moving Average 4', '0.0')
    Exponentialmovingaverage = st.text_input('Exponential Moving Average', '0.0')
    Relativestrengthindex = st.text_input('Relative Strength Index', '0.0')
    Momentum = st.text_input('Momentum', '0.0')
    Previousdaycloseprice = st.text_input('Previous Day Close Price', '0.0')
    
    # Code for Prediction
    if st.button('Predict Stock Price', key='prediction_button', help='Click to predict the stock price based on the input parameters'):
        prediction = stock_prediction([Simplemovingaverage_3,
                                       Simplemovingaverage_4,
                                       Exponentialmovingaverage,
                                       Relativestrengthindex,
                                       Momentum,
                                       Previousdaycloseprice])
        st.markdown(f"<p class='prediction'>Predicted Stock Price: {prediction[0]}</p>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
