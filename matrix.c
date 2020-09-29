#include <mpi.h>
#include <float.h>
#include <stdio.h>
#include <stdbool.h>
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
#define N_SIZE 4
//define max double size for each element
#define MAX_DATA_SIZE 10
//specify the number of nodes to use for computation @ the full sizeof
//an additional node is needed to compute the remaining Calculations
//another additional node is needed to receive the resulting segments from the computation nodes and write into 1 results matrix
#define NUM_NODES 2
#define BLOCK_SIZE 2


//generates doubles
double generate(double range){
  double b = RAND_MAX / range;
  return rand()/b;
}

//prints an array
void printa(double* array, int size){
  int i=0;
  for (i=0; i< (size *size); i++){
    if( i % size ==0){
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

void load_block(double* input, double* output, int block_no){
  // int block_i;
  // int NUM_BLOCKS = (N_SIZE/BLOCK_SIZE)*(N_SIZE/BLOCK_SIZE);
  // for(block_i=0; i<NUM_BLOCKS; block_i++){
  //
  // }

  int start_index = (block_no*BLOCK_SIZE*BLOCK_SIZE);
  int index=0;
  int i;
  for (i=0; i<BLOCK_SIZE; i++){
    int j;
    for(j=0; j<BLOCK_SIZE; j++) {
      output[c] = input[(starting_index*i)+j];
    }
  }
}

//compute the value of one cell in matrix 'c'
void compute(double* a, double* b, double* c, int element, int size){
  int row = (element / size) ;
  int col = (element % size) ;

  //printf("(%d,%d)\n", row, col);

  double sum=0.0;
  int start_row = row*size;
  int start_col = col;
  int i;
  for (i=0; i< size; i++){
    //printf("(Element %d*%d)\n", start_row+i, start_col+(i*N_SIZE));
    sum += a[start_row+i]*b[start_col+(i*size)];
  }
  c[element] = sum;
}

bool doubles_equal(double a, double b){
  return (fabs(a - b) < 0.001);
}

void vector_product(double *matrix, double *vector, double *output){
  int i;
  int j;

  for (int i = 0; i < N_SIZE; i++){
    output[i] = 0.0;
  }

  for (int i = 0; i < N_SIZE; i++){
    for (int j = 0; j < N_SIZE; j++){
      output[i] += (*(matrix + (i * N_SIZE) + j)) * vector[j];
    }
  }
}

void check_output(double *A, double *B, double *C){
  // To check for correctness, verify vector product ABx = Cx

  // Define array x
  double *x = (double*) malloc(N_SIZE * sizeof(double));
  double *y = (double*) malloc(N_SIZE * sizeof(double));
  double *ABx = (double*) malloc(N_SIZE * sizeof(double));
  double *Cx = (double*) malloc(N_SIZE * sizeof(double));

  // Populate x with numbers between 0-1
  int i;
  for (int i = 0; i < N_SIZE; i++){
    x[i] = rand() / (double) RAND_MAX;
  }

  // Calculate vector products ABx and Cx
  vector_product(B, x, y);
  vector_product(A, y, ABx);
  vector_product(C, x, Cx);

  // Check if ABx = Cx
  for (int i = 0; i < N_SIZE; i++){
    if (!doubles_equal(ABx[i], Cx[i])){
      printf("ERROR: INCORRECT OUTPUT\n");
      printf("%f != %f\n", ABx[i], Cx[i]);
      free(x);
      free(y);
      free(ABx);
      free(Cx);
      return;
    }
  }

  printf("CONGRATULATIONS, OUTPUT SUCCESSFUL!");

  free(x);
  free(y);
  free(ABx);
  free(Cx);
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

  double *block_a;
  double *block_b;

  //malloc buffers to doubles of value 0.00...
  a = (double*) malloc(N_SIZE*N_SIZE*sizeof(double));
  b = (double*) malloc(N_SIZE*N_SIZE*sizeof(double));
  c = (double*) malloc(N_SIZE*N_SIZE*sizeof(double));

  //populate arrays
  populate(a);
  populate(b);

  block_a = (double*) malloc(BLOCK_SIZE*BLOCK_SIZE*size(double));
  block_b = (double*) malloc(BLOCK_SIZE*BLOCK_SIZE*size(double));

  load_block(a, block_a, 3);
  printa(block_a, BLOCK_SIZE);



  //start time

  //matrix multiplication
//   int calculations =( N_SIZE*N_SIZE ) / NUM_NODES;
//   int remainingCalculations = (N_SIZE*N_SIZE)%NUM_NODES;
//   int x;
//   int counter=0;
//   int done =0;
//   for(x=0; x<=NUM_NODES; x++){
//     if(x==NUM_NODES){
//       if(world_rank ==x){
//         int i;
//         double* c_temp;
//         c_temp = (double*) malloc(remainingCalculations*sizeof(double));
//         for(i=0; i< remainingCalculations; i++){
//           compute(a,b,c,(x*calculations)+i,N_SIZE);
//           c_temp[i] = c[(x*calculations)+i];
//         }
//         MPI_Send(c_temp, remainingCalculations, MPI_DOUBLE, NUM_NODES+1, 0, MPI_COMM_WORLD);
//       }
//
//     }
//     else{
//       if(world_rank == x){
//         int i;
//         double* c_temp;
//         c_temp = (double*) malloc(calculations*sizeof(double));
//         for(i=0; i< calculations; i++){
//           compute(a,b,c,(x*calculations)+i,N_SIZE);
//           c_temp[i]= c[(x*calculations)+i];
//         }
//         MPI_Send(c_temp, calculations, MPI_DOUBLE, NUM_NODES+1, 0, MPI_COMM_WORLD);
//
//       }
//
//     }
//   }
//
//   //NODE receives results from computation nodes, and writes into a unified c matrix.
//   if(world_rank == NUM_NODES+1){
//     int i;
//     //printa(a,N_SIZE);
//     //printa(b, N_SIZE);
//     for(i=0; i<=NUM_NODES; i++){
//       if(i==NUM_NODES){
//         MPI_Recv(&c[NUM_NODES*calculations], remainingCalculations, MPI_DOUBLE, NUM_NODES, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//       }
//       else{
//         MPI_Recv(&c[i*calculations], calculations, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//       }
//     }
//     //stop time
//     //check output
//
//     //print results
//     //printa(c,N_SIZE);
//
//     check_output(a, b, c);
//
//     int j;
//     int marker = (NUM_NODES-1)*calculations;
//     for(j=marker; j< marker+ calculations; j++){
// //	    printf("%f\t", c[j]);
//     }
//
//  }

  // Finalize the MPI environment. No more MPI calls can be made after this
  MPI_Finalize();

  free(a);
  free(b);
  free(c);
}
