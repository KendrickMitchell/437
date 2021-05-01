#pragma once
#include "Data.h"


double Euc_dist(datapoint* A, datapoint* B, int dimensions) {
	int i = 0;
	double distance = 0.0;
	while (i < dimensions) {
		distance += pow(A->features[i] - B->features[i], 2);
		i++;
	}
	return sqrtl(distance);
}

double boolean_to_num(bool bolean) {
	if (bolean == true) { return 1; }
	return -1;
}

void sort_neighbors(int k, datapoint* kclosest) {
	datapoint temp;
	for (size_t i = 0; i < k; i++) {
		for (size_t j = 0; j < k - i - 1; j++)
		{
			if (kclosest[j].distance < kclosest[j + 1].distance) {
				temp = kclosest[j + 1];
				kclosest[j + 1] = kclosest[j];
				kclosest[j] = temp;
			}
		}
	}
	return;
}

double dataset::find_knn( int k, datapoint* testpoint,datapoint* trainpoint,int numpoints) {//For a single point
	datapoint* closest = nullptr;//Will need to be reordered after replacing a nearest neighbor. Bubble sort it.
	closest = (datapoint*)malloc(k * sizeof(datapoint));
	double classifier = 0;
	int i = 0;
	// Take k points to reside in the closest list without checking. Then order them
	for (i=0;i<k;i++) {
		closest[i] = this->Data[trainindexes[i]];
		closest[i].distance = Euc_dist(testpoint,closest+i,6);
	}
	sort_neighbors(k,closest);
	//Then check to see if the kth - 1 is further than a new point. If so replace it. Reorder list.
	for (;i<numpoints;i++) {
		if (Euc_dist(trainpoint + i, testpoint, 6) > Euc_dist(closest+k-1,testpoint,6)) {
			closest[k - 1] = this->Data[this->trainindexes[i]];
			sort_neighbors(k, closest);
		}
	}
	//Take a vote to see which class is most represented. Return the predicted class. 
	for (i=0;i<k;i++) {
		classifier += closest[i].binary_class;//Classes are either -1 or 1
	}
	free(closest);

	if (classifier > 0) { return -1; }
	if (classifier < 0) { return 1; }

	return -1;//<---A far larger portion of the data belongs to the negative class.
}

void dataset::Execute_knn(int k) {
	double accuracy = 0.0;
	int result = 0;
//Read in data while generating features. 
	//this->allocate_data(size);
	//this->file_read();
//for each test point run a knn test on it and see if it is correct. 
	 
	for (int i = 0;i < this->num_testpoint;i++) {//Now to test our test datapoints.
		result = find_knn(k, this->Data + this->testindexes[i], this->Data, this->num_datapoints - this->num_testpoint);
		if ((Data[this->testindexes[i]].binary_class && result==1) || (!Data[this->testindexes[i]].binary_class && result==-1)) {
			accuracy++;
		}
	}
	accuracy /= this->num_testpoint;
	printf("Accuracy is %lf", accuracy);
}














double dataset::find_knn_p( int k, datapoint* testpoint, datapoint* trainpoints, int numpoints, int exec, double* return_index) {//For a single point
	if (exec >= this->num_testpoint) { return 0.0; }//Now to just send back results to a pointer.
	
	datapoint* closest = nullptr;//Will need to be reordered after replacing a nearest neighbor. Bubble sort it.
	closest = (datapoint*)malloc(k * sizeof(datapoint));
	double classifier = 0;
	int i = 0;
	// Take k points to reside in the closest list without checking. Then order them
	for (i = 0; i < k; i++) {
		closest[i] = this->Data[trainindexes[i]];
		closest[i].distance = Euc_dist(testpoint, closest + i, 6);
	}
	sort_neighbors(k, closest);
	//Then check to see if the kth - 1 is further than a new point. If so replace it. Reorder list.
	for (; i < numpoints; i++) {
		if (Euc_dist(trainpoints + i, testpoint, 6) > Euc_dist(closest + k - 1, testpoint, 6)) {
			closest[k - 1] = this->Data[this->trainindexes[i]];
			closest[k - 1].distance = Euc_dist(trainpoints + i, testpoint, 6);
			sort_neighbors(k, closest);
		}
	}
	//Take a vote to see which class is most represented. Return the predicted class. 
	for (i = 0; i < k; i++) {
		classifier += closest[i].binary_class;//Classes are either -1 or 1
	}
	free(closest);

	if (classifier > 0) { *return_index = -1; }
	if (classifier < 0) { *return_index = 1; }

	return 1;//<---K better be an odd value.
}

void dataset::Execute_knn_parallel(int k) {
	std::thread threads[8];
	double accuracy = 0.0;
	int result = 0.0;
	int exec = 0;
	int i = 0;
	//Read in data while generating features. 
	//this->allocate_data(size);
	//this->file_read();
	//for each test point run a knn test on it and see if it is correct. 
	double* results = (double*)malloc(sizeof(double) * this->num_testpoint);///<---Used to store parralel results

	for (i = 0; i < this->num_testpoint; i++) {//Now to test our test datapoints.
		for (exec = 0; exec < this->num_threads; exec++) {
			threads[exec] = std::thread(&dataset::find_knn_p,this, k, this->Data + this->testindexes[i], this->Data, this->num_datapoints - this->num_testpoint, i, results + i);
		}//Ok apparently you need a pointer the object to access a member function.

		for (int exec = 0; exec < this->num_threads; exec++) {
			threads[exec].join();
		}
	}
	
	///Calculate accuracy. given the results
	for (i=0;i<this->num_testpoint;i++) {
		if ((this->Data[this->testindexes[i]].binary_class == true && results[i]==1) || (this->Data[this->testindexes[i]].binary_class == false && results[i] == -1)) {
			accuracy++;
		}
	}
	
	free(results);
	accuracy /= this->num_testpoint;
	printf("Accuracy is %lf ", accuracy);
}//The find_knn function needs a new version revision. Perhaps send a value to an array index holding a 1 or -1.

//Certainly learned something. Multiple processes need extra considerations like shared memory spaces, process independence
//Additional conditions to prevent collisions or out of bounds errors on shared memory.