# Machine Learning Pipeline for Loan Delinquency Prediction as an analog of Deliquency on P2P Lending

This is a Python code for a complete machine learning pipeline to predict loan delinquency using the Ridge regression algorithm.The larger aim of this model is to act as a proof of concept for a Peer to Peer Lending on Dataset provided by Payments apps + apps like splitwise.

## Dataset

The LendingClub dataset is a publicly available dataset consisting of data from LendingClub, an American peer-to-peer lending company. The dataset contains information on loans issued from 2007 to 2018 and includes information on loan characteristics such as loan amount, interest rate, loan grade, loan purpose, employment length, and borrower credit score. It also contains information on borrower demographic characteristics such as age, income, and home ownership status.The dataset original aswell as a tailored version is uploaded. The dataset has been tailored in accordance to the fact that all features applicable to Loan borrowing are not applicable in a P2P Lending scenario.

## Dependencies

This code requires the following libraries:

    numpy
    pandas
    scikit-learn
    pickle
    
## Data Cleaning and Preprocessing

The read_data() function reads the CSV file into a pandas DataFrame. The data_clean() function removes rows with missing values. The data_encoding() function encodes categorical features using one-hot encoding. The data_normalization() function scales the data and performs principal component analysis (PCA) for dimensionality reduction.

## Model Training

The train() function splits the preprocessed data into training and testing sets, applies a polynomial degree to eliminate linearity problems, and trains a Ridge regression model using cross-validation to optimize hyperparameters. The trained model is saved as a pickle file.

## Model Prediction

The predict() function loads the trained model and makes predictions on the test set. Predictions are clipped to a maximum value of 100 to represent percentage values.

## API

app.py contains information to build an API using the model above to build web apps etc.

## Hugging Face

Hugging Face contains a visualiazation of the data and predictions. [Here is a link to the Hugging Face space](https://huggingface.co/spaces/Amaysood/P2P_Lending_Deliquency)

## Usage

To use this code, simply run the train() function on your own loan data CSV file and adjust any hyperparameters as desired. Then, run the predict() function on the resulting trained model to make predictions on new loan data.To us the API using app.py run the code and open the speified url to run the api.

## Disclaimer

This code is intended for educational purposes only and should not be used for actual loan delinquency prediction without careful consideration and validation.
