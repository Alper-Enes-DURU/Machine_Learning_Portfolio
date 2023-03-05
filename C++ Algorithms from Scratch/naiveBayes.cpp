/**
 * A C++ program that is an implementation of Naive Bayes algorithm. 
 * It reads the CSV file named "titanic_project.csv".
 * 
 * @author Alper Duru
 * @date   03/04/2023
 */

// Include libraries
#include <vector>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <sstream>

using namespace std;

// Include function headers
vector<string> separateData (const string & str_, char delimiter_);
vector<vector<double>> readDataFile (string inputFile_, char delimiter = ',');
vector<double> convertStringToDouble (vector<string> val_);
double calculateMean (const vector<double> & val_);
double calculateVariance (const vector<double> & val_, double mean_);
double stdev (const vector<double> & val_, double mean_);
double calculateGaussian (double x_, double mean_, double stdev_);
vector<vector<double>> naiveBayes (const vector<vector<double>> & trainData_,
                                          const vector<double> & train_y);
double predict (const vector<vector<double>> & model_params,
                            const vector<double> & val_);
vector<double> evaluate (const vector<vector<double>> & testData_,
                                     const vector<double> & test_y,
                                     const vector<vector<double>> & model_params);


int main()
{
    // Read the data file
    vector<vector<double>> data = readDataFile("titanic_project.csv");

    vector<vector<double>> predictors(data.size() - 1);
    vector<double> response(data.size() - 1);

    for (int i = 1; i < data.size(); i++)
    {
        vector<double> row(data[i].size());

        for (int j = 0; j < data[i].size(); j++)
        {
            row[j] = stod(to_string(data[i][j]));
        }

        predictors[i - 1] = { row[2], row[3], row[4] };
        response[i - 1] = row[1];
    }

    int num_train = 800;
    vector<vector<double>> train_data(predictors.begin(), predictors.begin() + num_train);
    vector<double> train_y(response.begin(), response.begin() + num_train);
    vector<vector<double>> test_data(predictors.begin() + num_train, predictors.end());
    vector<double> test_y(response.begin() + num_train, response.end());

    vector<vector<double>> model_params = naiveBayes(train_data, train_y);
    vector<double> metrics = evaluate(test_data, test_y, model_params);

    cout << "Accuracy --> " << metrics[0] << endl;
    cout << "Sensitivity --> " << metrics[1] << endl;
    cout << "Specificity --> " << metrics[2] << endl;
    return 0;
}

/**
 * This function separates the string into substrings based on
 * the delimiter and returns a vector of those substrings.
 */
vector<string> separateData (const string & str_, char delimiter_)
{
    vector<string> tokenList;
    string token_;
    istringstream inputStream(str_);

    while (getline(inputStream, token_, delimiter_))
    {
        tokenList.push_back(token_);
    }

    return tokenList;
}

/**
 * This function read the data from a file and returns a vector of vectors, 
 * where each inner vector represents a row of data.
 */
vector<vector<double>> readDataFile (string filename, char delimiter_ = ',')
{
    ifstream inputFile (filename);

    if (!inputFile)
    {
        throw runtime_error("Error in opening the file!");
    }

    vector<vector<double>> dataList;
    string inputLine_;

    while (getline(inputFile, inputLine_))
    {
        vector<double> row;
        stringstream ss (inputLine_);
        string token;

        while (getline(ss, token, delimiter_))
        {
            if ( !token.empty() )
            {
                char * endptr;
                double value = strtod(token.c_str(), & endptr);

                if ( * endptr == '\0')
                {
                    row.push_back(value);
                }
            }
        }
        dataList.push_back(row);
    }
    return dataList;
}

/**
 * This function takes a vector of strings as input, converts
 * each string element to a double, and returns a vector of doubles. 
 */
vector<double> convertStringToDouble (vector<string> dataList_)
{
    vector<double> dataList2_(dataList_.size());

    transform(dataList_.begin(), dataList_.end(), 
              dataList2_.begin(), [](const string & val_)
              { return stod(val_); });

    return dataList2_;
}

/**
 * This function calcultes the mean value of a vector of doubles.
 */
double calculateMean (const vector<double> & dataList_)
{
    double totalVal = 0.0;

    for (int a = 0; a < dataList_.size(); a++)
    {
        totalVal += dataList_[a];
    }
    
    return totalVal / dataList_.size();
}

/**
 * This function calculates the variance of a vector of doubles, 
 * given the mean of a vector.
 */
double calculateVariance (const vector<double> & dataList_, double meanVal_)
{
    double totalVal = 0.0;

    for (int a = 0; a < dataList_.size(); a++)
    {
        totalVal += pow(dataList_[a] - meanVal_, 2);
    }

    return totalVal / dataList_.size();
}

/**
 * This function calculates the standard deviation of a vector of doubles,
 * given the mean of the vector.
 */
double stdev (const vector<double> & dataList_, double meanVal_)
{
    double variance = calculateVariance(dataList_, meanVal_);
    return sqrt (variance);
}

/**
 * This function calculates the probability density of a Gaussian
 * distribution for a gievn value, mean, and standar deviation.
 */
double calculateGaussian (double val_, double meanVal_, double stdev_)
{
    double expVal = exp(-pow(val_ - meanVal_, 2) / (2 * pow(stdev_, 2)));
    double denominatorVal_ = sqrt(2 * M_PI) * stdev_;
    return expVal / denominatorVal_;
}

vector<vector<double>> naiveBayes (const vector<vector<double>> & trainDataList,
                                          const vector<double> & trainData_y)
{
    vector<vector<vector<double>>> class_data(2);

    for (int i = 0; i < trainDataList.size(); i++)
    {
        int cls = (int) trainData_y[i];
        class_data[cls].push_back(trainDataList[i]);
    }

    int num_samples = trainData_y.size();
    double prior_0 = (double) class_data[0].size() / num_samples;
    double prior_1 = (double) class_data[1].size() / num_samples;

    int num_predictors = trainDataList[0].size();

    vector<vector<double>> meanList (2, vector<double> (num_predictors));
    vector<vector<double>> stdevList (2, vector<double> (num_predictors));

    for (int cls = 0; cls < 2; cls++)
    {
        for (int j = 0; j < num_predictors; j++)
        {
            vector<double> valueList;

            for (int i = 0; i < class_data[cls].size(); i++)
            {
                valueList.push_back(class_data[cls][i][j]);
            }

            double meanValue = calculateMean(valueList);
            double stdevVal = stdev(valueList, meanValue);
            meanList[cls][j] = meanValue;
            stdevList[cls][j] = stdevVal;
        }
    }

    vector<vector<double>> model_params = { meanList[0], stdevList[0], meanList[1], stdevList[1], { prior_0, prior_1} };
    return model_params;
}

double predict (const vector<vector<double>> & model_params,
                            const vector<double> & x)
{
    vector<double> mean_0 = model_params[0];
    vector<double> stdev_0 = model_params[1];
    vector<double> mean_1 = model_params[2];
    vector<double> stdev_1 = model_params[3];

    double prior_0 = model_params[4][0];
    double prior_1 = model_params[4][1];

    double likelihood_0 = 1.0;
    double likelihood_1 = 1.0;

    for (int j = 0; j < x.size(); j++)
    {
        double pdf_0 = calculateGaussian(x[j], mean_0[j], stdev_0[j]);
        double pdf_1 = calculateGaussian(x[j], mean_1[j], stdev_1[j]);
        likelihood_0 *= pdf_0;
        likelihood_1 *= pdf_1;
    }

    double posterior_0 = likelihood_0 * prior_0;
    double posterior_1 = likelihood_1 * prior_1;

    return posterior_1 > posterior_0 ? 1.0 : 0.0;
}

vector<double> evaluate (const vector<vector<double>> & testDataList,
                                     const vector<double> & testData_y,
                                     const vector<vector<double>> & model_params)
{
    int num_correct = 0;
    int truePositives = 0;
    int falsePositives = 0;
    int trueNegatives = 0;
    int falseNegatives = 0;

    for (int i = 0; i < testDataList.size(); i++)
    {
        double y_true = testData_y[i];
        double y_pred = predict(model_params, testDataList[i]);

        if (y_true == y_pred)
        {
            num_correct++;

            if (y_true == 1.0)
            {
                truePositives++;
            }
            else
            {
                trueNegatives++;
            }
        }
        else
        {
            if (y_true == 1.0)
            {
                falseNegatives++;
            }
            else
            {
                falsePositives++;
            }
        }
    }

    double accuracyVal = (double) num_correct / testDataList.size();
    double sensitivityVal = (double) truePositives / (truePositives + falseNegatives);
    double specificityVal = (double) trueNegatives / (trueNegatives + falsePositives);

    vector<double> metricList = { accuracyVal, sensitivityVal, specificityVal };
    return metricList;
}