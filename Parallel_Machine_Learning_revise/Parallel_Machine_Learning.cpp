// Parallel_Machine_Learning.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#pragma once

#include "Data.h"



int main()
{
    std::cout << "Hello World!\n";
    double test_equivalence = 0;
    double model_equivalence = 0;
    

    double train_accuracy = 0, test_accuracy = 0;
    clock_t t;

    dataset proj(127,1600,0),proj2(127,1600,1), proj3(127, 1600, 2), proj4(127, 1600, 3), proj5(127, 1600, 4);
    //test points and data points with thread count
    proj.allocate_data(1727);
    proj2.allocate_data(1727);
    proj3.allocate_data(1727);
    proj4.allocate_data(1727);
    
    t = clock();
    proj.file_read();//Maybe a huge buffer is alright.
    t = clock() - t;
    printf("Serial took me (%f seconds).\n", ((float)t) / CLOCKS_PER_SEC);
    proj.set_data_indices();

    t = clock();
    proj2.file_read_parallel();//Maybe a huge buffer is alright.
    t = clock() - t;
    printf("1 thread took me (%f seconds).\n", ((float)t) / CLOCKS_PER_SEC);
    proj2.set_data_indices();

    t = clock();
    proj3.file_read_parallel();//Maybe a huge buffer is alright.
    t = clock() - t;
    printf("2 thread took me (%f seconds).\n", ((float)t) / CLOCKS_PER_SEC);
    proj3.set_data_indices();

    t = clock();
    proj4.file_read_parallel();//Maybe a huge buffer is alright.
    t = clock() - t;
    printf("3 thread took me (%f seconds).\n", ((float)t) / CLOCKS_PER_SEC);
    proj4.set_data_indices();

    printf("\nNow executing learning\n");
  
    int k = 5;
    t = clock();
    proj.Execute_knn(k);
    t = clock() - t;
    printf("serial test took me (%f seconds).\n", ((float)t) / CLOCKS_PER_SEC);

    t = clock();
    proj2.Execute_knn_parallel(k);
    t = clock() - t;
    printf("1 thread test thread took me (%f seconds).\n", ((float)t) / CLOCKS_PER_SEC);

    t = clock();
    proj3.Execute_knn_parallel(k);
    t = clock() - t;
    printf("2 thread test thread took me (%f seconds).\n", ((float)t) / CLOCKS_PER_SEC);

    t = clock();
    proj4.Execute_knn_parallel(k);
    t = clock() - t;
    printf("3 thread test thread took me (%f seconds).\n", ((float)t) / CLOCKS_PER_SEC);
}
//The project will be implementing several machine learning models and seeing how feasible and beneficial doing so is.
//Feasibility is based on how trivial or difficult it is to implement. 
//Benefit is the time improvement of parallelization and the equivelance of the parallel model to its serial counterpart
//I was encouraged to start with knn and compare supervised to supervised and unsupervised to unsupervised.
//Keep in mind which models can train in parallel all models should be able to test in parralel.
//Classification will most likely be used for all models.

//The data is car data and classified as whether a car is acceptable or not. 
//The features are cost, maintenance cost, doors, capacity, luggage space, and saftey.
//only doors and capacity are numberic but they are discrete integers. So categorical and numerical values can easily be 
//transformed. Classification is binary too. 