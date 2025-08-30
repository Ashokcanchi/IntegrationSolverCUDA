#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "support.h"

void verify(unsigned int c, unsigned int d, unsigned int num_elements, float result) {

  
  int a = c; int b = d;
  float correct_result = 0;
    for (size_t i = 0; i < num_elements; i++) {
    float x = (float)a + (float)(b - a) * i / num_elements;
    float dx = (float)(b - a) / num_elements;
    correct_result += 2*x * dx;
  }

  const float relativeTolerance = 1e-1;
  float error = (result - correct_result)/result;

  printf("CPU RESULT: %f\n", correct_result);
  printf("GPU RESULT: %.3f\n", result);
  if(error > relativeTolerance || error < -relativeTolerance)
  {
    printf("\nFAILED\n");
    //exit(1);
  }
  else
  {
    printf("\nPASSED\n");
  }
}

void initVector(float **vec_h, unsigned size)
{
    *vec_h = (float*)malloc(size*sizeof(float));

    if(*vec_h == NULL) {
        FATAL("Unable to allocate host");
    }
    srand(217);
    for (unsigned int i=0; i < size; i++) {
        (*vec_h)[i] = (rand()%100)/100.00;
    }

}


void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

