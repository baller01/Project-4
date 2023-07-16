# Project-4

# Deep Learning

## Task Definition: 
Carbon emmissions in motor vehicles that is above 255g/km is considered as high. The objective of this task is to create a predictive model, a binary classifier, that predicts high or not high. 

## Tool Used:
Deep learning using neural network was used. 

## Process Steps:
- Load data from csv file
- preprocessing
- splitting the data into X_test, X_train, y_test, y_train
- create model
- compiling the model
- Train: fitting the model into the data
- Evaluating the accuracy of the model

## Results
The model had accuracy of 98.2% using 1 layer, 20 neurons and 50epochs
Some columns were which were dependant and highly corelated were removed. These include 'Make', 'Model', 'Fuel Consumption Comb (L/100 km)' and 'Fuel Consumption Comb (mpg)'
Optimization was done by binning the "Transmission" layer and increasing layers to 2. 
The accuracy of the optimized model was 98.2%

## Conclusion:
The model was successful as a binary classifier to identify a car as a higher emitter or not. 
