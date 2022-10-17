import streamlit as st
import numpy as np
import pickle
# loading the saved model
loaded_model = pickle.load(open("trained_model.csv", 'rb'))

html_temp='''
<style>
[data-testid="stAppViewContainer"]
{
background-image: url("https://images.unsplash.com/photo-1602615576820-ea14cf3e476a?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=870&q=80");
background-size: cover;
font-family:Courier; color:green; font-size: 20px;
}
</style>
'''
def iris_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    # print(prediction)

    if (prediction[0] == 0):
        return 'The species is Setosa!'
    elif (prediction[0] == 1):
        return "The species is Versicolor!"
    else:
        return 'The species is Virginica!'

def main():

    st.title("IRIS SPECIES PREDICTION!!!")
    st.markdown(html_temp, unsafe_allow_html=True)
    col1,  col2  =st.columns(2)
    with col1:
        sepal_length = st.number_input('sepal length in cm')
        sepal_width = st.number_input('sepal width in cm')

    with col2:
        petal_length = st.number_input('petal length in cm')
        petal_width = st.number_input('petal width in cm')
        species_iris=""

        if st.button('Submit'):
            species_iris = iris_prediction(
                [sepal_length, sepal_width, petal_length, petal_width])
            st.success(species_iris)
if __name__ == '__main__':
    main()