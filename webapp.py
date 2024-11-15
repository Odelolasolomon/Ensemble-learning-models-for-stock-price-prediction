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
    # Giving a title
    st.title('Stock Price Prediction Web App')
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Stock Price Prediction Web App </h2>
    </div>
    """
    
    # Getting the input data from the user
    Simplemovingaverage_3 = st.text_input('Simple Moving Average 3', '0.0')
    Simplemovingaverage_4 = st.text_input('Simple Moving Average 4', '0.0')
    Exponentialmovingaverage = st.text_input('Exponential Moving Average', '0.0')
    Relativestrengthindex = st.text_input('Relative Strength Index', '0.0')
    Momentum = st.text_input('Momentum', '0.0')
    Previousdaycloseprice = st.text_input('Previous Day Close Price', '0.0')
    
    # Code for Prediction
    if st.button('Predict stock price'):
        prediction = stock_prediction([Simplemovingaverage_3,
                                       Simplemovingaverage_4,
                                       Exponentialmovingaverage,
                                       Relativestrengthindex,
                                       Momentum,
                                       Previousdaycloseprice])
        st.success(f'Predicted Stock Price is : {prediction[0]}')

if __name__ == '__main__':
    main()
