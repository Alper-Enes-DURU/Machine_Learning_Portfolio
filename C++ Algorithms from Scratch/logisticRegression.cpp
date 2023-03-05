/**
 * A C++ program that is an implementation of logistic regression using
 * gradient descent optimization. It reads the CSV file named "titanic_project.csv",
 * fits the regression model on the data, and makes new predictions on new data.
 * 
 * @author Alper Duru
 * @date   03/04/2023
 */


// Include libraries
#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <string>
#include <sstream>

using namespace std::chrono;
using namespace std;

// Include function headers
vector<vector<double>> readDataFile (string inputFile_);
vector<string> separateData (string val_, char delimiter_);
vector<double> logRegression (vector<vector<double>> dataVal_, 
                                      int numOfIteration_, 
                                      double rateOfLearning_);
vector<double> gradDescent (vector<vector<double>> xVal_, 
                                   vector<double> yVal_, 
                                   vector<double> coefficient_, 
                                   int numOfIteration_, 
                                   double rateOfLearning_);
vector<double> calculatePrediction (vector<vector<double>> dataVal_, vector<double> coefficient_);
double calculateAccuracy (vector<double> predictionVal_, vector<double> actualVal_);
double calculateSensitivity (vector<double> predictionVal_, vector<double> targetVal_);
double calculateSpecificity (vector<double> predictionVal_, vector<double> targetVal_);


int main()
{
    // Read the data file
    vector<vector<double>> data = readDataFile("titanic_project.csv");

    // Split the data as train/test
    int trainDataSize = 800;
    vector<vector<double>> trainingDataList (data.begin(), data.begin() + trainDataSize);
    vector<vector<double>> testingDataList (data.begin() + trainDataSize, data.end());

    vector<double> trainingData_y(trainDataSize, 0.0);
    vector<double> trainingData_x(trainDataSize, 0.0);

    for (int ind = 0; ind < trainDataSize; ind++)
    {
        // Our target variable
        trainingData_y[ind] = trainingDataList[ind][0];
        // Our predictor (sex)
        trainingData_x[ind] = trainingDataList[ind][3];
    }

    // Start for logistic regression
    int numOfIteration_ = 1000;
    double rateOfLearning_ = 0.01;
    auto opStartTime = high_resolution_clock::now();

    vector<vector<double>> trainingDataList_x (trainDataSize, vector<double> (2, 0.0));

    for (int i = 0; i < trainDataSize; i++)
    {
        trainingDataList_x[i][0] = 1.0;
        trainingDataList_x[i][1] = trainingData_x[i];
    }

    // coefficients
    vector<double> initCoefficientsList = {0.0, 0.0};
    vector<double> coefficientsList = gradDescent(trainingDataList_x, 
                                                   trainingData_y, 
                                                   initCoefficientsList, 
                                                   numOfIteration_, 
                                                   rateOfLearning_);

    auto opStopTime = high_resolution_clock::now();
    auto totalTrainingTime = duration_cast < microseconds > (opStopTime - opStartTime).count();

    cout << "Total training time -- " << totalTrainingTime << " microseconds" << endl;
    cout << "Coefficients -- " << coefficientsList[0] << " " << coefficientsList[1] << endl;

    // Predictions on the test data
    vector<double> testData_y( testingDataList.size(), 0.0 );
    vector<double> testData_x( testingDataList.size(), 0.0 );
    
    for (int i = 0; i < testingDataList.size(); i++)
    {
        testData_y[i] = testingDataList[i][0];
        testData_x[i] = testingDataList[i][3];
    }

    vector<vector<double>> testingDataList_x(testingDataList.size(), vector<double> (2, 0.0));

    for (int i = 0; i < testingDataList.size(); i++)
    {
        testingDataList_x[i][0] = 1.0;
        testingDataList_x[i][1] = testData_x[i];
    }
    vector<double> test_predictions = calculatePrediction (testingDataList_x, coefficientsList);

    // Metrics on test data
    double testDataAccuracy = calculateAccuracy(test_predictions, testData_y);
    double testDataSensitivity = calculateSensitivity(test_predictions, testData_y);
    double testDataSpecificity = calculateSpecificity(test_predictions, testData_y);

    cout << "Testing data accuracy --> " << testDataAccuracy << endl;
    cout << "Testing data sensitivity --> " << testDataSensitivity << endl;
    cout << "Testing data specificity --> " << testDataSpecificity << endl;

    return 0;
}

/**
 * This function read the data from a file and returns a vector of vectors, 
 * where each inner vector represents a row of data.
 */
vector<vector<double>> readDataFile (string inputFile_)
{
    ifstream infile(inputFile_.c_str());

    if (!infile.good())
    {
        cerr << "Error in opening the file!" << inputFile_ << endl;
        exit(1);
    }

    vector<vector<double>> dataVal;
    string inputLine;

    while (getline(infile, inputLine))
    {
        vector<double> row;
        istringstream inputStream(inputLine);
        string token_;

        while (getline(inputStream, token_, ','))
        {
            try
            {
                row.push_back(stod(token_));
            }
            catch (const invalid_argument & exception_)
            {
                row.push_back(0.0);
            }
        }
        dataVal.push_back(row);
    }

    return dataVal;
}


vector<string> separateData (string val_, char delimiter_)
{
    vector<string> dataList;
    stringstream inputStream(val_);
    string token_;

    while (getline(inputStream, token_, delimiter_))
    {
        dataList.push_back(token_);
    }

    return dataList;
}

vector<double> logRegression (vector<vector<double>> dataVal_, int numOfIteration_, double rateOfLearning_)
{
    int numOfData = dataVal_[0].size() - 1;
    vector<double> coefficientsList(numOfData, 0.0);
    int numOfSampleData = dataVal_.size();

    for (int ind = 0; ind < numOfIteration_; ind++)
    {
        double cost = 0.0;
        vector<double> gradientList (numOfData, 0.0);

        for (int a = 0; a < numOfSampleData; a++)
        {
            double y = dataVal_[a][0];
            vector<double> dataList (numOfData, 0.0);

            for (int b = 1; b <= numOfData; b++)
            {
                dataList[b - 1] = dataVal_[a][b];
            }

            double z = 0.0;

            for (int b = 0; b < numOfData; b++)
            {
                z += coefficientsList[b] * dataList[b];
            }

            double h = 1.0 / (1.0 + exp(-z));
            cost += y * log(h) + (1.0 - y) * log(1.0 - h);

            for (int b = 0; b < numOfData; b++)
            {
                gradientList[b] += (h - y) * dataList[b];
            }
        }

        cost /= -numOfSampleData;

        for (int a = 0; a < numOfData; a++)
        {
            coefficientsList[a] -= rateOfLearning_ * gradientList[a] / numOfSampleData;
        }
    }
    return coefficientsList;
}


vector<double> gradDescent (vector<vector<double>> x, vector<double> y, 
                                 vector<double> coefficients, int num_iterations, 
                                 double learning_rate)
{
    int n = x.size();
    int m = x[0].size();

    for (int i = 0; i < num_iterations; i++)
    {
        vector<double> y_hat(n, 0.0);

        for (int j = 0; j < n; j++)
        {
            double z = 0.0;

            for (int k = 0; k < m; k++)
            {
                z += coefficients[k] * x[j][k];
            }
            y_hat[j] = 1.0 / (1.0 + exp(-z));
        }

        vector<double> errors(n, 0.0);

        for (int j = 0; j < n; j++)
        {
            errors[j] = y_hat[j] - y[j];
        }

        vector<double> new_coefficients(m, 0.0);

        for (int j = 0; j < m; j++)
        {
            double gradient = 0.0;

            for (int k = 0; k < n; k++)
            {
                gradient += errors[k] * x[k][j];
            }
            new_coefficients[j] = coefficients[j] - learning_rate * gradient;
        }
        coefficients = new_coefficients;
    }
    return coefficients;
}

/**
 * This function calculates predictions based on logical regression and
 * returns a vector of predicted values.
 */
vector<double> calculatePrediction (vector<vector<double>> dataList_, vector<double> coefficientList_)
{
    int numOfSamples = dataList_.size();
    int numOfFeatures = dataList_[0].size();
    vector<double> predictionList_(numOfSamples, 0.0);

    for (int a = 0; a < numOfSamples; a++)
    {
        double calc = 0.0;

        for (int b = 0; b < numOfFeatures; b++)
        {
            calc += coefficientList_[b] * dataList_[a][b];
        }

        double h = 1.0 / (1.0 + exp(-calc));
        predictionList_[a] = round(h);
    }

    return predictionList_;
}


/**
 * This function calculates the accuracy of a model given its precition
 * and the corresponding target values. Accuracy measures the proportion
 * of instances that were correctly classified by the model.
 */
double calculateAccuracy (vector<double> predictionList_, vector<double> actualList_)
{
    int numOfSamples = predictionList_.size();
    int numOfCorrectVal = 0;

    for (int a = 0; a < numOfSamples; a++)
    {
        if (predictionList_[a] == actualList_[a]) 
        {
            numOfCorrectVal++;
        }
    }
    return static_cast<double> (numOfCorrectVal) / numOfSamples;
}


/**
 * This function is used to calculate the sensitivity of a model given its predictions
 * and the corresponding target values.
 */
double calculateSensitivity (vector<double> predictionList_, vector<double> targetList_)
{
    int truePositives_ = 0;
    int falseNegatives_ = 0;

    for (int i = 0; i < predictionList_.size(); i++) 
    {
        if (predictionList_[i] == 1.0 && targetList_[i] == 1.0)
        {
            truePositives_++;
        }
        else if (predictionList_[i] == 0.0 && targetList_[i] == 1.0)
        {
            falseNegatives_++;
        }
    }

    if (truePositives_ == 0 && falseNegatives_ == 0)
    {
        return 1.0;
    }
    else
    {
        return (double) truePositives_ / (truePositives_ + falseNegatives_);
    }
}

/**
 * This function calculates the specificty of a model given its predictions and
 * the corresponding target values.
 */
double calculateSpecificity (vector<double> predictionList_, vector<double> targetList_)
{
    int trueNegatives_ = 0;
    int falsePositives_ = 0;

    for (int a = 0; a < predictionList_.size(); a++)
    {
        if (predictionList_[a] == 0.0 && targetList_[a] == 0.0)
        {
            trueNegatives_++;
        }
        else if (predictionList_[a] == 1.0 && targetList_[a] == 0.0)
        {
            falsePositives_++;
        }
    }

    if (trueNegatives_ == 0 && falsePositives_ == 0)
    {
        return 1.0;
    }
    else
    {
        return (double) trueNegatives_ / (trueNegatives_ + falsePositives_);
    }
}