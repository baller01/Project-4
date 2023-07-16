# Project-4
## Prediction of Carbon Emissions in Canada by car properties

## Description and objectives
To Relationship between car properties and carbon emission.
Correlation between car properties and carbon emission.
Presenting machine learning models to predict the carbon emissions from known properties of cars according to the availble data.

## Activities carried out
- Presentation and visusualization of availed data on tableau
- Clustering through unsupervised learning
- Multiple linear regression using Supervised Machine Learning
- Binary clasification using deep learning

## 1. Tableau Insights: CO2 Emissions Analysis

![Screenshot 2023-07-16 at 20 04 53](https://github.com/baller01/Project-4/assets/123272517/08c7dd75-4ac1-4ffc-8d04-42121f3c535f)

This repository contains insights derived from the analysis of 7,134 vehicles in order to understand the quantity of CO2 emissions. The analysis takes into consideration various variables, including fuel type, vehicle class, engine size, and number of cylinders.

# 1.1 CO2 Emissions Overview

![Screenshot 2023-07-16 at 20 55 34](https://github.com/baller01/Project-4/assets/123272517/43960858-40f5-442d-88fe-a566910e3e90)

Total number of vehicles analyzed: 7,134
Average CO2 emissions (g/km): 250.59
Note: The average CO2 emissions are above the normal range and require high attention. The European Commission's target by 2024 is to reach emission levels of 95g CO2/km for cars and 147g CO2/km for vans.
# 1.2 Highest and Lowest Polluting Vehicles

![Screenshot 2023-07-16 at 20 56 45](https://github.com/baller01/Project-4/assets/123272517/4ada47af-875c-4b8f-b674-7720fb56443a)

![Screenshot 2023-07-16 at 20 56 45](https://github.com/baller01/Project-4/assets/123272517/1b80d57c-672b-4fd8-b9d3-d7258ddbb84e)

The analysis revealed significant differences in CO2 emissions among different vehicle types:

Highest polluting vehicles:

Supercars and passenger vans recorded the highest CO2 emissions, with a staggering figure of more than 400g CO2/km.
Lowest polluting vehicles:

Hybrid vehicles, such as Ioniq, Prius, Corolla from Toyota, and Hyundai, exhibited significantly lower CO2 emissions, measuring less than 100g CO2/km.
# 1.3 Contribution of Vehicle Models

Among the various vehicle manufacturers, the following models were identified as significant contributors to CO2 emissions:

Mercedes
BMW
Ford
Chevrolet

## 1.4 Fuel Types and Usage

In terms of fuel types, the analysis revealed the following insights:

Premium Gasoline and Regular Gasoline were the most commonly used fuels among the top 100 polluting vehicles.

# 2. Unsupervised Machine Learning

## 2.1 Task Definition: 
In this study we investigate the Carbon Emissions in Canada by car properties by and check if we can create clusters by using unsupervised machine learning k-means method.

## 2.2 Tool Used:
We used sklearn libirary together with hvplot and pandas.

## 2.3 Process Steps:
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

Using unsupervised Machine Learning K-means model, we can categorise optimum 3 clusters based on the relationship among “Fuel Consumption City” and “Fuel Consumption Highway” with “CO2 Emission”.

# 3. MULTIPLE LINEAR REGRESSION ANALYSIS WITH SUPERVISED MACHINE LEARNING

In order to apply a model to analyse and make prediction to “CO2_Emissions_Canada.csv” data (7385 rows (non null) * 12 columns) , firstly some statistical informations have been identified to get first insight of data via mean, standart deviation, min- max, and percentiles.

![image](https://github.com/baller01/Project-4/assets/121508137/003ab59e-273f-47b8-b1d9-b497611442aa)

Then numerical columns have been extracted from the main dataframe, which can be seen below according to data types;

Engine size (L): Continuous variable (float64)

Cylinders: Discrete numerical variable (int64)

Fuel Consumption City (L/100 km): Continuous variable (float64)

Fuel Consumption Hwy (L/100 km): Continuous variable (float64)

Fuel Consumption Comb (L/100 km): Continuous variable (float64)

Fuel Consumption Comb (mpg):  ): Continuous variable  (int64)

CO2 Emissions(g/km):   Continuous variable (int64)

![image](https://github.com/baller01/Project-4/assets/121508137/c623e1e8-ad9b-4f1a-b85e-e32c7740a812)

Secondly their correlation have been checked and below heatmap has been created. In order to create graphs and statistical data visualisation  “matplotlib.pyplot” and “seaborn”  libraries have been used.

![image](https://github.com/baller01/Project-4/assets/121508137/1b2be6c2-c9bb-4ce8-96cb-3b32448f22ac)


As can be seen from the heatmap all numeric variables are correlated to each other and also to CO2 Emission at least %70 or higher. The correlations with CO2 Emissions are below:

CO2 Emissions(g/km) - Engine Size(L) = 0.85

CO2 Emissions(g/km) - Cylinders = 0.83

CO2 Emissions(g/km) - Fuel Consumption City (L/100 km) = 0.92

CO2 Emissions(g/km) - Fuel Consumption Hwy (L/100 km) = 0.88

*CO2 Emissions(g/km) - Fuel Consumption Comb (L/100 km) = 0.92

*CO2 Emissions(g/km) - Fuel Consumption Comb (mpg) = -0.91

Since “CO2 Emissions” is correlated with more than one variable Multiple Linear Regression model has been chosen for analysis and prediction (The two columns which are combinations of Fuel Consumptions have been excluded from the model). For this purpose, the normality of the distributions of variables has been checked. As seen from the below graphs the continuous variables are -fairly- normally distributed.


![image](https://github.com/baller01/Project-4/assets/121508137/b365fd63-28ca-4393-bef7-2fec7f033f06)

![image](https://github.com/baller01/Project-4/assets/121508137/60e9fbe6-6f20-49c0-ac5d-75713ab6c67b)

![image](https://github.com/baller01/Project-4/assets/121508137/c81bbf3b-8dbb-45bf-8100-f2d6fe986ed0)



## 3.1 MULTIPLE LINEAR REGRESSION MODEL; APPLICATION, EVALUATION AND PREDICTION WITH SUPERVISED MACHINE LEARNING


1-	From “sklearn” library for predictive data analysis, linear_model, train_test_split and LinearRegression have been imported.

2-	
       Dependent variable is:
	
 	y = "CO2 Emissions(g/km)"
  
        Independent variables are:

        X= "Engine Size(L)", "Cylinders", "Fuel Consumption City (L/100 km)", "Fuel Consumption Hwy (L/100 km)"

      	

3-	Data has been splitted as train and test sets. Regression model has been applied to training data set and prediction has been 
        made on X_test data.

4-	To evaluate the model, from sklearn.metrics , r2_score, mean_squared_error, mean_absolute_error were imported.

	R2 score: 0.8797165993674141
 
	Mean Squared Error: 409.5475270722951
 
	Mean Absolute Error: 13.499776395662566
       

•	R2: 88% of variance in our dependent variable can be explained by our independent variables. Which is quite high and shows that data fits to regression model and model predicts the data well.

•	MSE and MAE:  The difference between actual and predicted is small.


	Coefficients: [5.75078492 6.65332716 6.8317655 6.48691586]

	Intercept: 50.52369559592012
 

•	Regression Equation:

y = X1*5.75 + X2*6.66 + X3*6.83 + X4*6.49 + 50.52

X1 = Engine Size(L)

X2 = Cylinders

X3 = Fuel Consumption City (L/100 km)

X4 = Fuel Consumption Hwy (L/100 km)

y = CO2 Emissions(g/km)

All independent variables are positively related to “y” (when they increase “y” also increases) and intercept is also positive. 

•	predict the CO2 emission of a car, where:

Engine Size(L) =  2.4
	
Cylinders =  4 
  
Fuel Consumption City (L/100 km) = 11.2 
 
Fuel Consumption Hwy (L/100 km) = 7.7
	
(predictedCO2 = regr_model.predict([[2.4, 4, 11.2, 7.7]]))
 
predictedCO2=  217.4 
 
actual CO2 Emissions = 221 g/km


The Line of Linear Regression

A straight line that starts from 50.2 on y axis.

![image](https://github.com/baller01/Project-4/assets/121508137/231e4857-d045-45c2-9098-f0e4ae7b6a86)


Residuals plot

Residuals randomly scatter around “0”, which means linear regression model is a good fit for the data.

![image](https://github.com/baller01/Project-4/assets/121508137/d8080275-cf6f-4bfd-9227-70ede721418e)


# 4. Deep Learning

## 4.1 Task Definition: 
Carbon emmissions in motor vehicles that is above 255g/km is considered as high. The objective of this task is to create a predictive model, a binary classifier, that predicts high or not high. 

## 4.2 Tool Used:
Deep learning using neural network was used. 

## 4.3 Process Steps:
- Load data from csv file
- preprocessing
- splitting the data into X_test, X_train, y_test, y_train
- create model
- compiling the model
- Train: fitting the model into the data
- Evaluating the accuracy of the model

## 4.4 Results
The model had accuracy of 98.2% using 1 layer, 20 neurons and 50epochs
Some columns were which were dependant and highly corelated were removed. These include 'Make', 'Model', 'Fuel Consumption Comb (L/100 km)' and 'Fuel Consumption Comb (mpg)'
Optimization was done by binning the "Transmission" layer and increasing layers to 2. 
The accuracy of the optimized model was 98.2%

The model was successful as a binary classifier to identify a car as a higher emitter or not. 

# CONCLUSIONS

In our project as well as exploring the data through “Tableu” we used unsupervised Machine Learning for clustering and supervised Machine Learning and Deep Learning for making predictions.

Exploring the data through “Tableu” we identified some relationships between car properties and “CO2 Emissions” from data, such as identifying average CO2 Emission according to car model or finding most polluting fuel types.

Using unsupervised Machine Learning we have trained the model by using similarities in data by K-Means method and created sample clusters according to CO Emissions and Fuel Consumptions. Considering the purpose of an analysis more clusters can be done by changing the variables in the data.

Both of our models (multiple linear regression and logistic regression) have high accuracy, short training time, and make good predictions.

1.	With multiple linear regression by changing multiple properties of a car, CO2 emissions can be predicted with 88% accuracy.

2.	With logistics regression by changing the threshold value of CO2 emissions it can be predicted/tested with 98% accuracy and classified as lower or higher from that specific value.

Our models have high accuracy and less training time because:

There is no irrelevant information in the data set.

We don’t have any redundant features in our model. 

We don’t have any random data value.

We have enough continuous numeric data.

We have well defined categorical data with valuable information.

The features are highly correlated to target variable.

We haven’t dropped the outliners deliberately because every data has its importance with its own value in regression analysis also the car values are not random and belong to real existing cars. This also increases our model’s efficiency by recognizing cars in similar data sets. For instance, we have analyzed Canada’s car properties and CO2 emissions, but our models can also give high accuracy in other countries’ data sets in same topic.

We did not include Fuel Consumption Comb (L/100 km) and Fuel Consumption Comb (mpg) columns to our models since they are combination of Fuel Consumption City and Hwy features (55% city, 45% Hwy).
	
# RECOMMENDATIONS

In our project as well as exploring the data through “Tableu” we wanted to use supervised, unsupervised Machine Learning and Deep Learning as tools. They all created valuable outcomes to solve real life problems and make predictions related to “CO2 Emissions and cars’ properties” so they can be used for making predictions in different data sets.

However, during Logistics Regression modelling in Deep Learning we have realized that the task is not too complex for deep learning since it reaches the high accuracy with min epoch and even without in need of a second layer. Since deep learning is about non - linearity and complexity, we think that this is due to the highly linear nature of our data and high correlations between dependent and independent variables. Thus, instead of using Deep Learning, supervised Machine Learning application can be well enough for this type of linear model – data relationship.


