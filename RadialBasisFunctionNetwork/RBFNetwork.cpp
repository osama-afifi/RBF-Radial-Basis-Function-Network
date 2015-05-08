#include "RBFNetwork.h"

using namespace std;

RBFNetwork::RBFNetwork(const vector<datapoint> &training_data, const vector<int> &training_labels, int num_of_labels)
	: training_data(training_data), training_labels(training_labels), num_of_labels(num_of_labels)
	, random_real_gen(0, 1), random_engine(rd())
{}

RBFNetwork::~RBFNetwork(void)
{
}

double RBFNetwork::startTraining(int num_rbf_units, double learning_rate, int num_iterations, double &mse, bool print_flag)
{
	if(print_flag)
	{
		printf("Started RBF Network Training...\n");
		printf("Getting RBF Centroids using K-means++...\n");
	}


	// Calculate RBF Centroids
	KmeansPP KMPP(training_data);
	rbf_centroids.clear();
	KMPP.RunKMeansPP(num_rbf_units, rbf_centroids);

	if(print_flag)
		printf("Building the RBF Units...\n");
	// Build First (Fixed) Layer Units
	buildRBFUnits();

	// Init the Second Layer with values [-1,1]
	layer2_weights.assign(num_of_labels, vector<double>(num_rbf_units,0));
	for(auto &label: layer2_weights)
		for(auto &arc: label)
			arc = random_real_gen(random_engine);

	// Train the second layer weights
	mse = 0;
	double accuracy=0;
	for(int iter = 0 ; iter<num_iterations ; iter++)
	{	
		for(int i = 0 ; i<training_data.size() ; i++)
		{
			for(int label = 0 ; label<num_of_labels ; label++)
			{
				double hypothesis = Utility::multiplyVectors(rbf_units[i], layer2_weights[label]);
				double truth_base = (training_labels[i]==label) ? +1.0 : -1.0;
				double error_direction = (truth_base-hypothesis) * truth_base;
				vector<double> delta = Utility::multiplyVecConst(rbf_units[i], -error_direction * learning_rate);
				Utility::AddVectors(layer2_weights[label], delta);	
			}
		}

		// Gathering Statistics
		mse = 0.0;
		accuracy=0;
		for(int i = 0 ; i<training_data.size() ; i++)
		{
			double error_dir=0;
			int prediction = predictLabel(training_data[i],error_dir);
			if(prediction == training_labels[i])accuracy++;
			mse += error_dir * error_dir;
		}
		mse *= (double)(1.0/(double)training_data.size());
		accuracy *= (double)(1.0/(double)training_data.size());

		if(print_flag)
		{
		 printf("Training Epoch (%d / %d), MSE=[%.3f], Acc=[%.3f], Progress [%.2f]\r",
			 (iter+1), num_iterations, mse, accuracy*100.0 , ((double)((double)(iter+1)/(double)num_iterations) * 100.0));
		}
			
			
		if(mse < 1e-9)
			break;
	}
	if(print_flag)
		printf("\n----------------------------\n");
	return accuracy;
}

void RBFNetwork::buildRBFUnits()
{
	total_centroids_dist.clear();
	total_centroids_dist.assign(rbf_centroids.size(),0);
	// Calculate total distance for each centroid
	for(int i = 0 ; i<rbf_centroids.size() ; i++)
		for(auto &data_point : training_data)
			total_centroids_dist[i] += distance(data_point,rbf_centroids[i]);

	// Build RBF Units
	rbf_units.assign(training_data.size(),vector<double>());
	for(int i = 0 ; i<training_data.size() ; i++)
	{
		for(int j = 0 ; j<rbf_centroids.size() ; j++)
			rbf_units[i].push_back(basisFunction(training_data[i], rbf_centroids[j], total_centroids_dist[j]));
	}
}

double RBFNetwork::basisFunction(const datapoint &data_point, const datapoint &centroid, const double total_centroid_dist)
{ 
	return exp(-distance(data_point,centroid) / (total_centroid_dist));
}

double RBFNetwork::distance(const datapoint &a, const datapoint &b)
{
	double dist=0;
	for(int i = 0 ; i< a.size() ; i++)
		dist += (a[i]-b[i])*(a[i]-b[i]);
	return dist;
}

int RBFNetwork::predictLabel(const datapoint &data_point, double &error)
{
	double maxi=-1e18;
	int bestLabel=-1;
	vector<double> cur_rbf_unit;
	for(int j = 0 ; j<rbf_centroids.size() ; j++)
		cur_rbf_unit.push_back(basisFunction(data_point, rbf_centroids[j], total_centroids_dist[j]));
	// Get the label with maximum hypothesis
	for(int label = 0 ; label<num_of_labels ; label++)
	{
		double hypothesis = Utility::multiplyVectors(cur_rbf_unit, layer2_weights[label]);
		if(maxi<hypothesis)
		{
			maxi = hypothesis;
			bestLabel = label;
		}
	}
	assert(bestLabel!=-1);
	error = (double)bestLabel-maxi;
	return bestLabel;
}