#pragma once

#include<vector>
#include<math.h>
#include "KmeansPP.h"
#include "Util.h"


typedef std::vector<double> datapoint;
class RBFNetwork
{
public:
	RBFNetwork(const std::vector<datapoint> &training_data, const std::vector<int> &training_labels, int num_of_labels);
	~RBFNetwork(void);

	/* Start Training the Radial Basis Function network
		Takes the number of RBF centroids, the learning rate, the number of iteration and a print flag as input
		Saves the output model to be used in testing and single predictions 
		return accuracy and mse (by reference)
		*/
	double startTraining(int num_rbf_units, double learning_rate, int num_iterations, double &mse, bool print_flag = false);

	/* Start Testing the RBF Network to make sure it's not overfitting
		(Should be done after training) */
	void startTesting(const std::vector<datapoint> &testing_data, const std::vector<int> &testing_labels);

	
	/* Predict a single data point support multi-classes (One vs. All method)
		(Should be done after training of course) */
	int predictLabel(const datapoint &data_point, double &error);

	//TODO
	void saveModel();
	void loadModel();

private:
	int num_of_labels;
	std::vector<datapoint> training_data;
	std::vector<int> training_labels;
	std::vector< std::vector<double> > rbf_units;
	std::vector< std::vector<double> > layer2_weights;
	std::vector<datapoint> rbf_centroids;
	std::vector<double>total_centroids_dist;

	// Random Number seed devices/engines/distributions
    std::random_device rd;
	std::default_random_engine random_engine;
	std::uniform_real_distribution<double> random_real_gen;

	void buildRBFUnits();
	double distance(const datapoint &a ,const datapoint &b);
	double basisFunction(const datapoint &data_point, const datapoint &centroid, const double total_centroid_dist);
	
};

