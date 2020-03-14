import os
from sklearn.metrics import *
from sklearn.decomposition import PCA
import pandas as pd
from sklearn import svm
from scipy import stats
from scipy.stats import entropy
import numpy as np
from sklearn.model_selection import KFold
import joblib

# traverse to the directory
def directory(dir):
    fileList = []
    for directory, dirs, name in os.walk(dir):
        [fileList.append('{0}/{1}'.format(dir, n)) for n in name]
    return fileList

# get the data for meal and nomeal from the folders
def getData(files):
    df = pd.read_csv(files, names=list(range(31)))
    return fillNans(df)

# remove all the NaN values from the pandas series
def removeNans(pdseries):
    filled_data = []
    # using interpolate to remove the NaN values forwardly
    for series in pdseries:
        filled_data.append(pd.Series(series).interpolate(method='linear', limit_direction='forward').to_list())
    return pd.DataFrame(filled_data)

# fill the NaN values of the dataframe
def fillNans(dataframe):
    # 1. remove NaNs
    cleaned_data = removeNans(dataframe.values.tolist())

    # 2. get the first column and check the NaN value
    first_column = cleaned_data.columns[0]

    # 3. check if it contains null values
    check_isnull = cleaned_data[first_column].isnull()

    # 4. if yes, get its index and create a list of those
    nan_idx_list = cleaned_data[check_isnull].index.tolist()

    # 5. using the median (more accurate) to fill those null values
    values = cleaned_data.median(axis=0).tolist()
    for i in nan_idx_list:
        cleaned_data.loc[i] = values

    return np.array(cleaned_data)


# create a class that can pass the cleaned data into all the features
class features:
    def __init__(self, data):
        self.data = data

    # feature 1. covariance (10 of those to make the accuracy higher)
    def covariance(self):
        cov1 = np.cov(self.data[:, 1:3])
        cov2 = np.cov(self.data[:, 4:6])
        cov3 = np.cov(self.data[:, 7:9])
        cov4 = np.cov(self.data[:, 10:12])
        cov5 = np.cov(self.data[:, 13:15])
        cov6 = np.cov(self.data[:, 16:18])
        cov7 = np.cov(self.data[:, 19:21])
        cov8 = np.cov(self.data[:, 22:24])
        cov9 = np.cov(self.data[:, 25:27])
        cov10 = np.cov(self.data[:, 28:30])


        return (cov1 + cov2 + cov3 + cov4 + cov5 + cov6 + cov7 + cov8 + cov9 + cov10) / 10

    # feature 2. entropy values (calculate the error rate)
    def entropy(self, base=None):
        array = []
        for i in range(len(self.data)):
            value, counts = np.unique(self.data, return_counts=True)
            array.append(entropy(self.data[i, :]))
        np.array(array).reshape((len(self.data), 1)) # reshape to the matrix that can be concatenated
        return np.array(array)

    # feature 3. skewness values (calculate the bias rate) using the built-in function
    def skewness(self):
        array = []
        for i in range(len(self.data)):
            array.append(stats.skew(np.array(self.data[i, :])))
        np.array(array).reshape((len(self.data), 1))
        return np.array(array)

    # feature 4. kurtosis values (calculate the flatness or peakness)
    def kurtosis(self):
        array = []
        for i in range(len(self.data)):
            array.append(stats.kurtosis(np.array(self.data[i, :])))
        np.array(array).reshape((len(self.data), 1))
        return np.array(array)

    # feature 5. polyfit values (calculate the ralation between two data sets)
    def polyfit(self, meal_df, nomeal_df):
        array = []
        for i in range(31):
            arr = np.polyfit(meal_df[0:30][i], nomeal_df[0:30][i], 3)
            array.append(arr)
        return np.array(array)

    # feature 6. chi-2 values (calculate the usefulness)
    def chisquare(self):
        array = []
        for i in range(len(self.data)):
            array.append(stats.chisquare(self.data, ddof=1))

        x = np.array(array)[:, 0, 0]  # cut the 3-dimension to 1 dimension for matrix concatenate
        return np.array(x)


# KFold function
def K_Fold(matrix):
    result_list = []

    # apply kfold by spliting 10 times of the whole training set
    k_fold = KFold(n_splits=10, shuffle=True, random_state=1)

    # split the matrix and get training set and testing set in every folding
    for idx_train, idx_test in k_fold.split(matrix):
        X_train, X_test = matrix[idx_train], matrix[idx_test]

        # Y train and test using the last column
        Y_train, Y_test = matrix[idx_train][:,-1], matrix[idx_test][:,-1]

        # apply SVM and fit the model with training sets
        model = svm.SVC(kernel='rbf')
        model.fit(X_train, Y_train)

        # predict the model and get the scores
        Y_pred = model.predict(X_test)
        result_list.append(classification_report(Y_test, Y_pred))

    print(result_list[0]) # the first result has the greatest accuracy

def modelpkl(model):
    joblib.dump(model, 'ModelForTesting.pkl')
    print('Model Successfully Built!')


def main():
    # MealData and NoMealData directory for the csv files
    meal_data = directory('./MealNoMealData/MealData')
    nomeal_data = directory('./MealNoMealData/NoMealData')

    # get the meal and nomeal data from separate folders
    meal = np.concatenate([getData(i) for i in meal_data])
    nomeal = np.concatenate([getData(i) for i in nomeal_data])

    # transform them into dataframes
    meal_df = pd.DataFrame(meal)
    nomeal_df = pd.DataFrame(nomeal)

    # take the last column as the labeling column
    # mark nomeal as 0 and meal as 1
    nomeal_df[meal_df.shape[1]] = [0 for _ in range(len(nomeal))]
    meal_df[meal_df.shape[1]] = [1 for _ in range(len(meal))]

    # concat the meal and nomeal dataframes
    data = pd.concat([meal_df, nomeal_df])

    # trim the dataframe to 510 * 31 dataframe
    first_31 = data[[i for i in range(31)]]

    # Extract Features
    feat = features(first_31.values)
    feat1 = feat.covariance()
    feat2 = feat.entropy()
    feat3 = feat.skewness()
    feat4 = feat.kurtosis()
    feat5 = feat.polyfit(meal_df, nomeal_df)
    feat6 = feat.chisquare()

    # create a new matrix for PCA
    dataMatrix = np.concatenate((feat2[:, None], feat3[:, None], feat4[:, None], feat6[:, None], feat1), axis=1)

    # create a PCA with 5 components
    pca = PCA(n_components=5)
    principalComponents = pca.fit_transform(dataMatrix)  # fit the model
    principalDf = pd.DataFrame(data=principalComponents)
    array = pca.explained_variance_ratio_  # get the variance ratio

    # loop through the variance ratio array and find the values that are greater than 0.2
    temp = []
    for i in range(len(array)):
        t = array[i]
        if t > 0.20:
            temp.append(i)

    # create a label for columns
    label = data[31]

    # apply kfold using the pca dataframe and the label
    K_Fold(np.concatenate((principalDf[[i for i in temp]], label[:, None]), axis=1))

    # Y_test data and train the model with SVM and put in to pkl file
    Y_t = np.concatenate((principalDf[[i for i in temp]], label[:, None]), axis=1)[:, -1]
    model = svm.SVC(kernel='rbf')
    model.fit(principalDf[[i for i in temp]], Y_t)
    modelpkl(model)

if __name__ == '__main__':
    main()
