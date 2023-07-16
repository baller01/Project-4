# Project-4

# Unsupervised Machine Learning

## Task Definition: 
In this study we investigate the Carbon Emissions in Canada by car properties by and check if we can create clusters by using unsupervised machine learning k-means method.

## Tool Used:
We used sklearn libirary together with hvplot and pandas.

## Process Steps:
1- Identifing database data types

<img width="206" alt="image" src="https://github.com/baller01/Project-4/assets/118228120/07e6e7ab-7242-4468-b2d1-e19452a5e9c4">

2- Scale all numeric data

<img width="311" alt="image" src="https://github.com/baller01/Project-4/assets/118228120/62358bb0-50db-4f3d-9138-57c64bd02fd2">

3- Scale nonnumeric data by using dummies

<img width="485" alt="image" src="https://github.com/baller01/Project-4/assets/118228120/820ad634-7809-4b0a-8010-77ecc4debf97">

4- With the updated dataset included scaled data, calculate and visualise the inertia values

<img width="368" alt="image" src="https://github.com/baller01/Project-4/assets/118228120/10081288-44c2-4be0-8bfd-7397ca86ae74">

5- Optimum cluster quantity defined as 3 based on graph above. Clusters assigned to the scaled dataset and graphs created for the desired variable - output data.

<img width="335" alt="image" src="https://github.com/baller01/Project-4/assets/118228120/d2a438d2-ca1a-4b07-b089-ebe6c5ae5dad">

<img width="335" alt="image" src="https://github.com/baller01/Project-4/assets/118228120/7735b26a-dc33-422d-b908-8344d71918b2">

## Conclusion:
Using unsupervised Machine Learning K-means model, we can categorise optimum 3 clusters based on the relationship among “Fuel Consumption City” and “Fuel Consumption Highway” with “CO2 Emission”.
