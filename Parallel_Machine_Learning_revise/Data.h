#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cmath>
#include <iostream>
#include <thread>
struct data_coordinate {
    double features[6];
};

struct datapoint
{
    double features[6];
    double distance;
    bool binary_class;
};

class dataset
{
public:
    dataset();
    dataset(int a, int b, int thr);
    ~dataset();

    void file_read();
    void file_read_parallel();

    void Execute_knn(int k);
    void Execute_knn_parallel(int k);
    double find_knn( int k, datapoint* testpoint, datapoint* trainpoint, int numpoints);
    double find_knn_p( int k, datapoint* testpoint, datapoint* trainpoints, int numpoints, int exec, double* return_index);

    void Execute_perceptron(int features, int numtrain, double alpha, int epochs);
    void Execute_perceptron(int features, int threads);

    void set_data_indices();//Use num_datapoints and num_testpoints. with index arrays.

    void allocate_data(int size);

private:
    int num_datapoints;
    int num_testpoint;
    datapoint* Data;
    datapoint* testdata;

    void fill_data(char* buff);
    void fill_data_parallel(char* buff);

    int* testindexes;
    int* trainindexes;
    int num_threads;
};


char* fillbuff(char* read_file, char* data_buff);