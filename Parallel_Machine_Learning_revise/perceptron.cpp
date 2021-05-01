#pragma once
#include "Data.h"

void dataset::Execute_perceptron(int features, int num_train, double alpha,int epochs) {
	// Establish a vector of length |features.|
	double bias = 0;
	double* vector = (double*)malloc(sizeof(double) * features);
	double output = 0;
	double accuracy = 0.0;
	// Run over a set of set training points 
	for (int h = 0;epochs > h;h++) {
		for (int i = 0; i < this->num_datapoints - this->num_testpoint; i++) {
			output = 0;
			for (int j = 0; j < features; j++) {
				output = vector[j] * this->Data[i].features[j];
			}
			// See if a mistake was made 
			if (output <= 0 && this->Data[i].binary_class == true) {
				bias += alpha;
				for (int v = 0; v < features; v++) {//Correct weights in vector
					vector[v] += alpha * vector[v] * this->Data[i].features[v];
				}
			}
			if (output >= 0 && this->Data[i].binary_class == false) {
				bias -= alpha;
				for (int v = 0;v<features;v++) {//Correct weights in vector
					vector[v] -= alpha * vector[v] * this->Data[i].features[v];
				}
			}
		}
	}
	// Run over a set of test points- Keep track of accuracy in training 
	for (int i = 0; i < this->num_datapoints - this->num_testpoint; i++) {
		output = 0;
		for (int j = 0; j < features; j++) {
			output = vector[j] * this->Data[i].features[j];
		}
		if ((output > 0 && this->Data[i].binary_class) || (output < 0 && this->Data[i].binary_class == false)) {
			accuracy += 1;//Increment accuracy
		}
	}
	accuracy /= double(this->num_testpoint);
	// Run over the test points
	//Now print out test accuracy andthe resulting training vector
	printf("Test accuracy is %lf", accuracy);

	free(vector);
}

/*
void dataset::Execute_knn_parallel() {
	return;
}*/