#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <unistd.h>
//define matrix size
#define N_SIZE 100
//define max double size for each element
#define MAX_DATA_SIZE 10
#define NUM_NODES 6   

//generates doubles
double generate(double range){
  double b = RAND_MAX / range;
  return rand()/b;
}

//prints an array
void printa(double* array){
  int i=0;
  for (i=0; i< (N_SIZE *N_SIZE); i++){
    if( i % N_SIZE ==0){
      printf("\n");
    }
    printf("%f\t", array[i]);
  }
  printf("\n");
}

//populate with random doubles bounded by MAX_DATA_SIZE.
void populate(double* array){
  int i=0;
  for (i=0; i< (N_SIZE *N_SIZE); i++){
    array[i] = generate(MAX_DATA_SIZE);
  }
}

//compute the value of one cell in matrix 'c'
void compute(double* a, double* b, double* c, int element){
  int row = (element / N_SIZE) ;
  int col = (element % N_SIZE) ;

  //printf("(%d,%d)\n", row, col);

  double sum=0.0;
  int start_row = row*N_SIZE;
  int start_col = col;
  int i;
  for (i=0; i< N_SIZE; i++){
    //printf("(Element %d*%d)\n", start_row+i, start_col+(i*N_SIZE));
    sum += a[start_row+i]*b[start_col+(i*N_SIZE)];

  }
  c[element] = sum;

}

int main(int argc, char** argv) {
  // Initialize the MPI environment. The two arguments to MPI Init are not
  // currently used by MPI implementations, but are there in case future
  // implementations might need the arguments.
  MPI_Init(NULL, NULL);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  struct timeval start;
  struct timeval end;
  double *a;
  double *b;
  double *c;

  //malloc buffers to doubles of value 0.00...
  a = (double*) malloc(N_SIZE*N_SIZE*sizeof(double));
  b = (double*) malloc(N_SIZE*N_SIZE*sizeof(double));
  c = (double*) malloc(N_SIZE*N_SIZE*sizeof(double));

  //populate arrays
  populate(a);
  populate(b);

  //print out the matrix
  //printa(a);
  printf("\n");
  //printa(b);
  printf("\n");


  //start time

  //matrix multiplication
  int calculations =( N_SIZE*N_SIZE ) / NUM_NODES;
  int remainingCalculations = (N_SIZE*N_SIZE)%NUM_NODES;
  int x;
  int counter=0;
  for(x=0; x<=NUM_NODES; x++){
    if(x==NUM_NODES){
      if(world_rank ==x){
        int i;
        for(i=0; i< remainingCalculations; i++){
          compute(a,b,c,(x*calculations)+i);
        }
      }

    }
    else{
      if(world_rank == x){
        int i;
        for(i=0; i< calculations; i++){
          compute(a,b,c,(x*calculations)+i);
        }
      }

    }
  }

 // printa(c);
  int j;
  for(j=0; j<100; j++){
    printf("%f\t", c[j]);
  }
  printf("\n");
  //stop time
  // Finalize the MPI environment. No more MPI calls can be made after this
  MPI_Finalize();

  //check output

  //print results
  free(a);
  free(b);
  free(c);
}
