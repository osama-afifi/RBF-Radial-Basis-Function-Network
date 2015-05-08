#ifndef _UTIL_H_
#define _UTIL_H_


#include <vector>
#include <assert.h>

class Utility
{

public:

	/*
	Multiply equal sized vectors and return the result in double type
	*/
	static double multiplyVectors(const std::vector<double> &a , const std::vector<double> &b)
	{
		assert(a.size() == b.size());
		double res=0;
		for(int i = 0 ; i<a.size() ; i++)
			res += a[i] * b[i];
		return res;
	}

	/*
		Multiplies a vector with a given constant and return the result vector
	*/
	static std::vector<double> multiplyVecConst(const std::vector<double>&input , const double c)
	{
		std::vector<double>res(input.size());
		for(int i = 0 ; i<input.size() ; i++)
				res[i] = input[i] * c;
		return res;
	}

	/*
		adds a vector to a given input (passed by reference)
	*/
	static void AddVectors(std::vector<double>&input , const std::vector<double>&addend)
	{
		assert(input.size() == addend.size());
		for(int i = 0 ; i<input.size() ; i++)
				input[i] += addend[i];
	}

	/*
		calculates the covariance between two vectors
	*/
static double coVariance(datapoint x, datapoint y)
{
	assert(x.size() == y.size());
	double a = 0, b = 0, c = 0;
	for (int i = 0; i < x.size(); ++i)
	{
		a += x[i];
		b += y[i];
		c += x[i] * y[i];
	}

	return (double)(c / x.size()) - (double)((a / x.size()) * (b / x.size()));
}

};

#endif