/*
 * This program is a C++ implementation of statistical analysis of data 
 * stored in a CSV file named "Boston.csv". It computes the sum, mean,
 * median, range, covariance, and correlation of two sets of data, "rm" and "medv".
 * 
 * @author Alper Duru
 * @date 02/04/2023
 */

// Include libraries needed
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <string>
#include <math.h>
#include <numeric>

using namespace std;

// Function headers
double sumOf_vectorData( vector<double> &vec );
double meanOf_vectorData( vector<double> &vec );
double medianOf_vectorData( vector<double> vec );
double rangeOf_vectorData( vector<double> &vec );
double covarianceOf_vectorData( vector<double> &rm, vector<double> &medv );
double correlationOf_vectorData( vector<double> &rm, vector<double> &medv );
void print_stats( vector<double> &vec );

int main(int arg, char **argv)
{
    ifstream inFS;
    // Input file stream
    string line;
    string rm_in, medv_in;
    const int MAX_LEN = 1000;
    vector<double> rm(MAX_LEN);
    vector<double> medv(MAX_LEN);

    // Try to open file
    cout << "\nOpening file Boston.csv." << endl;
    inFS.open("Boston.csv");

    if (!inFS.is_open())
    {
        cout << "Could not open file Boston.csv." << endl;
        return 1; // 1 indicates error
    }

    // Can now use inFS stream like cin stream
    // Boston.csv should contain two doubles
    cout << "Reading line 1" << endl;
    getline(inFS, line);

    // echo heading
    cout << "heading: " << line << endl;
    int numObservations = 0;

    while (inFS.good())
    {
        getline(inFS, rm_in, ',');
        getline(inFS, medv_in, '\n');
        rm.at(numObservations) = stof(rm_in);
        medv.at(numObservations) = stof(medv_in);
        numObservations++;
    }

    rm.resize(numObservations);
    medv.resize(numObservations);

    cout << "New length " << rm.size() << endl;
    cout << "Closing file Boston.csv." << endl;

    inFS.close(); // Done with file, so close it

    cout << "Number of records: " << numObservations << endl;
    cout << "\nStats for rm" << endl;

    print_stats(rm);

    cout << "\nStats for medv" << endl;

    print_stats(medv);

    cout << "In Covariance = " << covarianceOf_vectorData(rm, medv) << endl;
    cout << "In Correlation = " << correlationOf_vectorData(rm, medv) << endl;
    cout << "\nProgram terminated.\n";

    return 0;
}

/*
 *   This function uses a for-each loop to iterate through the elements 
 *   of the vector and add each element to the sumVal variable, which is initialized to zero.
 *   The sumVal variable holds the accumulated sum of the elements.
 */
double sumOf_vectorData( vector<double> &vec_ )
{
    double sumVal = 0.0;
    for (const double &val : vec_)
    {
        sumVal += val;
    }

    return sumVal;
}

/*
 * This function calculates the mean of a vector of double values. 
 * It uses the function sumOf_vectorData() to find the sum of the 
 * values in the vector and then divides the sum by the number of 
 * elements in the vector (found using vec.size()) to find the mean. 
 * The result is casted to a double value before being returned.
 */
double meanOf_vectorData( vector<double> &vec_ )
{
    return static_cast<double>( sumOf_vectorData( vec_ ) ) / vec_.size();
}

/*
 * This function calculates the median of a vector of double values. 
 * It sorts the vector using the sort() function from the algorithm 
 * library, with the vec.begin() and vec.end() arguments representing 
 * the range of elements to sort. The number of elements in the vector 
 * is found using vec.size() and stored in the n variable.
 */
double medianOf_vectorData( vector<double> vec_ )
{
    sort(vec_.begin(), vec_.end());
    size_t vectorSize = vec_.size();

    if (vectorSize % 2 == 0)
    {
        return (vec_[vectorSize / 2 - 1] + vec_[vectorSize / 2]) / 2;
    }
    else
    {
        return vec_[vectorSize / 2];
    }
}

/*
 * This function calculates the range of a vector of double values 
 * by finding the minimum and maximum values in the vector and 
 * subtracting the minimum from the maximum.
 */
double rangeOf_vectorData( vector<double> &vec_ )
{
    double minVal = vec_[0];
    double maxVal = vec_[0];

    for (const double &val : vec_)
    {
        if (val < minVal)
        {
            minVal = val;
        }
        if (val > maxVal)
        {
            maxVal = val;
        }
    }

    return maxVal - minVal;
}

/*
 * This function calculates the covariance between two data sets, 
 * represented by the input vectors rm and medv. The mean of each 
 * data set is calculated, and then the covariance is determined 
 * by summing the product of the deviations of each data point 
 * from its respective mean. The final covariance value is normalized by dividing by n-1.
 */
double covarianceOf_vectorData( vector<double> &rm_, vector<double> &medv_ )
{
    int n = rm_.size();
    double rm_mean = 0.0;
    double medv_mean = 0.0;

    for (int i = 0; i < n; i++)
    {
        rm_mean += rm_[i];
        medv_mean += medv_[i];
    }

    rm_mean /= n;
    medv_mean /= n;

    double covariance = 0.0;

    for (int i = 0; i < n; i++)
    {
        covariance += (rm_[i] - rm_mean) * (medv_[i] - medv_mean);
    }

    covariance /= (n - 1);

    return covariance;
}

/*
 * This function calculates the correlation between two data sets (rm and medv)
 * by determining their mean, standard deviation, and covariance. 
 * The correlation is calculated as the normalized covariance.
 */
double correlationOf_vectorData( vector<double> &rm_, vector<double> &medv_ )
{
    int n = rm_.size();
    double rm_mean = 0.0;
    double medv_mean = 0.0;

    for (int i = 0; i < n; i++)
    {
        rm_mean += rm_[i];
        medv_mean += medv_[i];
    }

    rm_mean /= n;
    medv_mean /= n;

    double rm_stddev = 0.0;
    double medv_stddev = 0.0;
    double covariance = 0.0;

    for (int i = 0; i < n; i++)
    {
        rm_stddev += (rm_[i] - rm_mean) * (rm_[i] - rm_mean);
        medv_stddev += (medv_[i] - medv_mean) * (medv_[i] - medv_mean);
        covariance += (rm_[i] - rm_mean) * (medv_[i] - medv_mean);
    }

    rm_stddev = sqrt(rm_stddev / (n - 1));
    medv_stddev = sqrt(medv_stddev / (n - 1));
    covariance /= (n - 1);

    return covariance / (rm_stddev * medv_stddev);
}

/*
 * This function takes a vector as a parameter and prints the
 * data of the given vector by calling the functions defined in this file.
 */
void print_stats( vector<double> &vec )
{
    cout << "------------------------------------" << endl;
    cout << "Sum Of The Vector Data = " << sumOf_vectorData( vec ) << endl;
    cout << "Mean Of The Vector Data = " << meanOf_vectorData( vec ) << endl;
    cout << "Median Of The Vector Data = " << medianOf_vectorData( vec ) << endl;
    cout << "Range Of The Vector Data = " << rangeOf_vectorData( vec ) << endl;
}