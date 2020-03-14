## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## General info
### Given
Meal Data and No Meal Data of 5 subjects.
Ground truth labels of Meal and No Meal for 5 subjects.
### Todo
1. Extract features from Meal and No Meal data
2. Make sure that the features are discriminatory
3. Each student trains a separate machine to recognize Meal or No Meal data
4. Use k fold cross validation on the training data to evaluate your recognition system
5. Each student write a function that takes one test sample as input and outputs 1 if it predicts the test sample as meal or 0 if it predicts test sample as No meal. 
	
## Technologies
Project is created with:
* Python version: 3.7
* PyCharm version: 2019.3.3
* PyCharm Runtime version: 11.0.5 *64
	
## Setup
Before running this project, here are some notes:
* The Meal and NoMeal data are separated into two folders which are called "MealData" and "NoMealData".
* To modify the path of the .csv data, please do it accordingly.
* It is assumed that all the data files are in .csv format.
* There is a .pkl file created while running the trainingData.py file for later testing use.
* Since it is not mentioned, the project only trained the model using SVM. However, Bagging, Random Forest or Adaboost can be used as well.
* The accuracy and other scores are meaningful but expected a drop while new data are trained.

##### Note: Run this project using PyCharm will be the best.
