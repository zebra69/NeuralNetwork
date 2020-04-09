/*************************************************************/
/* C-program for delta-learning rule                         */
/* Learning rule of one neuron                               */
/*                                                           */
/* This program is produced by Qiangfu Zhao.                 */
/* You are free to use it for educational purpose            */
/*************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define I             3
#define n_sample      4
#define eta           0.5
#define lambda        1.0
#define desired_error 0.01
#define step(x)    0 > x ? -1 : 1;
#define frand()       (rand()%10000/10001.0)
#define randomize()   srand((unsigned int)time(NULL))

double x[n_sample][I]={
  { 0, 0, -1},
  { 0, 1, -1},
  { 1, 0, -1},
  { 1, 1, -1}
};

double w[I];
double d[n_sample]={-1, -1, -1, 1};
double o;

void Initialization(void);
void FindOutput(int);
void PrintResult(void);

main(){
  int    i,p,q=0;
  double LearningSignal,Error=DBL_MAX;

  Initialization();
  while(Error>desired_error){
    q++;
    Error=0;
    for(p=0; p<n_sample; p++){
      FindOutput(p);
      Error+=0.5*pow(d[p]-o,2.0);
      LearningSignal = eta*(d[p]-o);
      for(i=0;i<I;i++){
        w[i]+=LearningSignal*x[p][i];
      }
    }
    printf("Error in the %d-th learning cycle=%f\n",q,Error); 
  }
  PrintResult();
}
  
/*************************************************************/
/* Initialization of the connection weights                  */
/*************************************************************/
void Initialization(void){
  int i;

  randomize();
  for(i=0; i<I; i++) w[i]=frand();
}

/*************************************************************/
/* Find the actual outputs of the network                    */
/*************************************************************/
void FindOutput(int p){
  int    i;
  double temp=0;

  for(i=0;i<I;i++) temp += w[i]*x[p][i];
  o = step(temp);
}

/*************************************************************/
/* Print out the final result                                */
/*************************************************************/
void PrintResult(void){
  int i;

  printf("\n\n");
  printf("The connection weights of the neurons:\n");
  for(i=0;i<I;i++) printf("%5f ",w[i]);
  printf("\n\n");
}

