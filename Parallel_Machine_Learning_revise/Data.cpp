#pragma once
#include "Data.h"

#pragma warning(disable : 4996)

dataset::dataset()
{
    num_threads = 0;
    num_testpoint = 0;
    num_datapoints = 0;
    Data = nullptr;
    testdata = nullptr;
    testindexes = nullptr;
    trainindexes = nullptr;

}

dataset::dataset(int a, int b, int thr)
{
    num_testpoint = a;
    num_datapoints = b;
    num_threads = thr;
    Data = nullptr;
    testdata = nullptr;
    testindexes = nullptr;
    trainindexes = nullptr;
}

dataset::~dataset()
{
    free(this->Data);
    free(this->testdata);
    free(this->testindexes);
    free(this->trainindexes);
}

char* fillbuff(char* read_file, char* data_buff)
{
    FILE* f = fopen("car.data", "r");
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    //data_buff = (char*)malloc(fsize + 1); //51,866 space
    fread(data_buff, 1, fsize, f);
    fclose(f);
    data_buff[fsize] = NULL;
    return data_buff;
}

void dataset::allocate_data(int size)
{
    this->Data = (datapoint*)malloc(size * sizeof(datapoint));
}

double feature_creation(char* data, datapoint* D, int s, int thresh) {
    if (s >= thresh) {
        return 0.0;
    }
    char* copy = data;
    char* feature;
    
    feature = strtok(copy, ",");
    if (!strcmp(feature, "low")) {
        D->features[0] = 1;
    }
    if (!strcmp(feature, "med")) {
        D->features[0] = 2;
    }
    if (!strcmp(feature, "high")) {
        D->features[0] = 3;
    }
    if (!strcmp(feature, "vhigh")) {
        D->features[0] = 4;
    }

    feature = strtok(NULL, ",");
    if (!strcmp(feature, "low")) {
        D->features[1] = 1;
    }
    if (!strcmp(feature, "med")) {
        D->features[1] = 2;
    }
    if (!strcmp(feature, "high")) {
        D->features[1] = 3;
    }
    if (!strcmp(feature, "vhigh")) {
        D->features[1] = 4;
    }

    feature = strtok(NULL, ",");
    if (!strcmp(feature, "2")) {
        D->features[2] = 2;
    }
    if (!strcmp(feature, "3")) {
        D->features[2] = 3;
    }
    if (!strcmp(feature, "4")) {
        D->features[2] = 4;
    }
    if (!strcmp(feature, "5more")) {
        D->features[2] = 5;
    }

    feature = strtok(NULL, ",");
    if (!strcmp(feature, "2")) {
        D->features[3] = 2;
    }
    if (!strcmp(feature, "4")) {
        D->features[3] = 4;
    }
    if (!strcmp(feature, "more")) {
        D->features[3] = 6;
    }
    
    feature = strtok(NULL, ",");
    if (!strcmp(feature, "small")) {
        D->features[4] = 2;
    }
    if (!strcmp(feature, "med")) {
        D->features[4] = 4;
    }
    if (!strcmp(feature, "big")) {
        D->features[4] = 6;
    }

    feature = strtok(NULL, ",");
    if (!strcmp(feature, "low")) {
        D->features[5] = 2;
    }
    if (!strcmp(feature, "med")) {
        D->features[5] = 4;
    }
    if (!strcmp(feature, "high")) {
        D->features[5] = 6;
    }

    feature = strtok(NULL, ",");
    if (!strcmp(feature, "unacc")) {
        D->binary_class = false;
    }
    if (!strcmp(feature, "med") || !strcmp(feature, "acc") || !strcmp(feature, "good") || !strcmp(feature, "vgood")) {
        D->binary_class = true;
    }
    return 0;
}

void dataset::fill_data(char* buff) //Good here
{
    printf("Filling data...");
    std::thread threads[8];
    int i = 0;
    char* line = NULL;
    char* sav = NULL;
    line = strtok_s(buff, "\n", &sav);
    while (line != NULL) {
            feature_creation(line, this->Data + i,i,this->num_datapoints);//Limit
            //this->num_datapoints++;
            i++;
            //printf("%d\n", i);
            line = strtok_s(NULL, "\n", &sav);
    }
    printf("Finished filling data\n");
}

void dataset::fill_data_parallel(char* buff) //
{
    std::thread threads[8];
    int i = 0;
    int executions = 0;
    char* line = NULL;
    char* sav = NULL;
    this->num_datapoints = 1600;
    line = strtok_s(buff, "\n", &sav);
    while (line != NULL) {
        for (executions = 0; executions < this->num_threads ;executions++) {
            threads[executions] = std::thread(feature_creation, line, this->Data + i, i, this->num_datapoints);//feature_creation(line, this->Data + i, i, this->num_datapoints); // Essentially shooting blanks if I go over
            
            i++;
            //printf("%d\n", i);
            if (sav != NULL) {
                line = strtok_s(NULL, "\n", &sav);
            }
        }
        for (executions = 0; executions < this->num_threads ;executions++) {
            threads[executions].join();
        }
    }
}

bool is_within(int* arr, int size, int num) {
    int i = 0;
    for (i=0;i<size;i++) {
        if (num == arr[i]) {
            return true;
        }
    }
    return false;
}

void dataset::set_data_indices() {
    //Use random numbers.
    this->testindexes = (int*)malloc(sizeof(int) * this->num_testpoint);
    this->trainindexes = (int*)malloc(sizeof(int) * (this->num_datapoints - this->num_testpoint));

    srand(time(NULL));
    int i = 0;
    int random = -1;
    while (i<this->num_testpoint)
    {
        random = rand() % this->num_datapoints;
        if(!is_within(this->testindexes,i,random)){
            testindexes[i] = random;
            i++;
        }
    }
    int j = 0;
    printf("\nRandomizing train indices...");
    while (j < this->num_datapoints - this->num_testpoint)
    {
        random = rand() % this->num_datapoints;
        if (!is_within(this->testindexes, i, random) && !is_within(this->trainindexes, j, random)) {
            trainindexes[j] = random;
            j++;
            //printf("\n%d with %d", j, random);

        }
    }
    printf("Finished Randomizing train indices\n");
}//Make sure to give test data some indices to fill.
void dataset::file_read() {
    char buff[51867];
    char file[9] = { 'c', 'a', 'r', '.', 'd', 'a', 't', 'a', '\0' };
    
    fillbuff(file, buff);

    this->fill_data(buff);
    //    free(buff); <---Causes a crash fix it later
}

void dataset::file_read_parallel() {
    char buff[51867];
    char file[9] = { 'c', 'a', 'r', '.', 'd', 'a', 't', 'a', '\0' };
    fillbuff(file, buff);

    this->fill_data_parallel(buff);
//    free(buff); <---Causes a crash fix it later
}