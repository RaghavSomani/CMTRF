#ifndef INIT_H
#define INIT_H

#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include <map>
#include <random>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <cstdio>
#include <Eigen/Dense>
#include <Eigen/Sparse> // life saviour _/\_
#include <iterator>
using namespace std;
using namespace Eigen;
#include "spline.h"

// typedefs for some most common sparse data types
typedef SparseMatrix<double,ColMajor> SpMatC;
typedef SparseMatrix<double,RowMajor> SpMatR;
typedef Triplet<double> T;

// uniform random number generator in between 0 and 1 to partition dataset into train and test
default_random_engine generator;
uniform_real_distribution<double> uniform(0.0,1.0);

// uniform constant margin of eps in the sigma functions
#define eps 0.5

// measure class to store hyperparameters corresponding to different types of measures
class measure
{
public:
	double l1;
	double l2;
	int K;
	void set_params(double L1, double L2, int k=0)
	{
		l1 = L1;
		l2 = L2;
		K = k;
	}
};

// low_rank class for with different measures as its member objects
class low_rank
{
public:
	measure KT;
	measure TL;	
};

// algo class for different algorithm objects
// each with a map of k -> lowrank object storing optimal parameters for each measure
class algo
{
public:
	map<int,low_rank> Kmap;
};

// Dataset class with name and 4 algo objects
class Dataset
{
public:
	string name;
	algo No_sigma;
	algo sigma_1;
	algo sigma_N;
	algo sigma_K;
};


#endif