# Project-4

MULTIPLE LINEAR REGRESSION ANALYSIS WITH SUPERVISED MACHINE LEARNING

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



MULTIPLE LINEAR REGRESSION MODEL; APPLICATION, EVALUATION AND PREDICTION WITH SUPERVISED MACHINE LEARNING


1-	From “sklearn” library for predictive data analysis, linear_model, train_test_split and LinearRegression have been imported.

2-	Dependent variable:

 	y = "CO2 Emissions(g/km)"
  
    Independent variables:
    
  X= "Engine Size(L)", "Cylinders", "Fuel Consumption City (L/100 km)", "Fuel Consumption Hwy (L/100 km)"

3-	Data has been splitted as train and test sets. Regression model has been applied to training data set and prediction has been made on X_test data.

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



