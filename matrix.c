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
//define largest matrix size
#define N_SIZE 14000
//define max double size for each element
#define MAX_DATA_SIZE 10
//define the size of a subsequent square block matrix. Will contain BLOCK_SIZE by BLOCK_SIZE elements
#define BLOCK_SIZE 3500


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

//loads a block into of data into an indepenent array BLOCK_SIZExBLOCK_SZIE matrix
void load_block(double* input, double* output, int block_no){
  int NUM_BLOCKS = (N_SIZE/BLOCK_SIZE)*(N_SIZE/BLOCK_SIZE);
  int start_index;
  if((block_no%(N_SIZE/BLOCK_SIZE) ==0)) start_index = block_no*BLOCK_SIZE*BLOCK_SIZE;
  else{
    start_index = ((block_no-(block_no%(N_SIZE/BLOCK_SIZE)))*BLOCK_SIZE*BLOCK_SIZE)+((block_no%(N_SIZE/BLOCK_SIZE))*BLOCK_SIZE);
  }
  int index=0;
  int i;
  for (i=0; i<BLOCK_SIZE; i++){
    int j;
    for(j=0; j<BLOCK_SIZE; j++) {
      output[index] = input[(start_index+(i*N_SIZE)+j)];
      index+=1;
    }
  }
}

//write a 1D array into a block_no within a block_size x block_size matrix
void write_block(double* input, double* output, int block_no){
  int NUM_BLOCKS = (N_SIZE/BLOCK_SIZE)*(N_SIZE/BLOCK_SIZE);
  int start_index;
  if((block_no%(N_SIZE/BLOCK_SIZE) ==0)) start_index = block_no*BLOCK_SIZE*BLOCK_SIZE;
  else{
    start_index = ((block_no-(block_no%(N_SIZE/BLOCK_SIZE)))*BLOCK_SIZE*BLOCK_SIZE)+((block_no%(N_SIZE/BLOCK_SIZE))*BLOCK_SIZE);
  }
  int index=0;
  int i;
  for (i=0; i<BLOCK_SIZE; i++){
    int j;
    for(j=0; j<BLOCK_SIZE; j++) {
      output[(start_index+(i*N_SIZE)+j)] = input[index];
      index+=1;
    }
  }
}
void col_load(double*input, double*output, int col){
  double* temp = (double*) malloc(BLOCK_SIZE*BLOCK_SIZE*sizeof(double));
  int h;
  for(h=0; h<N_SIZE/BLOCK_SIZE; h++){
    load_block(input, temp, col+((N_SIZE/BLOCK_SIZE)*h));
    write_block(temp, output, (N_SIZE/BLOCK_SIZE)+h);
  }
  free(temp);
}
void row_load(double* input, double* output, int row){
  int i;
  for(i=0; i<N_SIZE*BLOCK_SIZE; i++){
    output[i]= input[i+(BLOCK_SIZE*N_SIZE*row)];
  }
}
//compute one element within a matrix of index element
void compute(double* a, double* b, double* c, int element, int size){
  int row = (element / size) ;
  int col = (element % size) ;
  double sum=0.0;
  int start_row = row*size;
  int start_col = col;
  int i;
  for (i=0; i< size; i++){
    sum += a[start_row+i]*b[start_col+(i*size)];
  }
  c[element] = sum;
}

//Compute the block matrix multiplication. Size is the size of the block matrix
void compute_matrix(double* input, double* output, int element, int size){
  int num_blocks = (N_SIZE/BLOCK_SIZE);
  int row = (element / (N_SIZE/BLOCK_SIZE)) ;
  int col = (element % (N_SIZE/BLOCK_SIZE)) ;

  double* block_a = (double*) malloc(BLOCK_SIZE*BLOCK_SIZE*sizeof(double));
  double* block_b = (double*) malloc(BLOCK_SIZE*BLOCK_SIZE*sizeof(double));
  double* result= (double*) malloc(BLOCK_SIZE*BLOCK_SIZE*sizeof(double));
  double* sum= (double*) malloc(BLOCK_SIZE*BLOCK_SIZE*sizeof(double));
  int start_row = row*size;
  int start_col = col;
  int i;

  //Calculate block multiplication.
  for (i=0; i< num_blocks; i++){
    //load two matrices for multiplication
    load_block(input, block_a, i );
    load_block(input,block_b, i+num_blocks);
    int j;
    for(j=0; j< BLOCK_SIZE*BLOCK_SIZE; j++){
      compute(block_a,block_b, result, j,BLOCK_SIZE );
    }

    //Add matrix multiplication result into the sum
    int k;
    for(k=0; k< BLOCK_SIZE*BLOCK_SIZE; k++){
      output[k] += result[k];
    }

  }
  free(block_a);
  free(block_b);
  free(sum);
  free(result);
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

  printf("CONGRATULATIONS, OUTPUT SUCCESSFUL!\n");

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


  if(world_rank==0){
    struct timeval start;
    struct timeval end;
    double *a;
    double *b;
    double *c;
    double *d;
    double*c_block;
    double *block_a;
    double* rcv;

    //malloc buffers to doubles of value 0.00...
    a = (double*) malloc(N_SIZE*N_SIZE*sizeof(double));
    b = (double*) malloc(N_SIZE*N_SIZE*sizeof(double));
    c = (double*) malloc(N_SIZE*N_SIZE*sizeof(double)); //populate arrays
    c_block = (double*) malloc(BLOCK_SIZE*BLOCK_SIZE*sizeof(double));
    populate(a);
    populate(b);
    int NUM_BLOCKS = (N_SIZE/BLOCK_SIZE)*(N_SIZE/BLOCK_SIZE);
    int blocks_length = (N_SIZE/BLOCK_SIZE);
    int x;


    //matrix multiplication
    //Send each block to be done by another rank.
    int j;
    d = (double*) malloc(N_SIZE*N_SIZE*sizeof(double));
    rcv = (double*) malloc(BLOCK_SIZE*BLOCK_SIZE*sizeof(double));
    double* temp = (double*) malloc(BLOCK_SIZE*BLOCK_SIZE*sizeof(double));
    double start_time;
    double* buffer = (double*) malloc(2*blocks_length*BLOCK_SIZE*BLOCK_SIZE*sizeof(double));

    start_time = MPI_Wtime();
    //compute the block for node 0
    row_load(a, buffer, 0);
    col_load(b, buffer, 0);


    compute_matrix(buffer, c_block, 0, blocks_length);
    write_block(c_block, d, 0);


    //send data to subseuquent nodes for calculations
    int f;
    for(f=1; f< NUM_BLOCKS; f++){
      int row = (f / blocks_length) ;
      int col = (f % blocks_length) ;
      row_load(a,buffer, row);
      col_load(b,buffer, col);
      MPI_Send(buffer, 2*blocks_length*BLOCK_SIZE*BLOCK_SIZE, MPI_DOUBLE, f, 0, MPI_COMM_WORLD);


    }
    //receive block data
    for(j=1; j<NUM_BLOCKS; j++){
      MPI_Recv(temp, BLOCK_SIZE*BLOCK_SIZE, MPI_DOUBLE, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      write_block(temp, d,j);
    }

    double end_time = MPI_Wtime();

    check_output(a, b, d);
    double duration = end_time - start_time;
    printf("TIME START: %f\nTIME END: %f\nDURATION: %f\n",start_time, end_time, end_time - start_time);
    free(a);
    free(b);
    free(c);
    free(c_block);
    free(rcv);
    free(d);
    free(temp);
    free(buffer);
  }

  //receive blocks from node 0 and compute. return to node 0 when computation is finished
  int i;
  for(i=1; i<(N_SIZE/BLOCK_SIZE)*(N_SIZE/BLOCK_SIZE); i++){
    if(world_rank==i){
      double* buffer = (double*)malloc(2*N_SIZE*BLOCK_SIZE*sizeof(double));
      double* c_block = (double* ) malloc(BLOCK_SIZE*BLOCK_SIZE*sizeof(double));
      MPI_Recv(buffer, 2*N_SIZE*BLOCK_SIZE, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      compute_matrix(buffer, c_block, i, N_SIZE/BLOCK_SIZE);
      MPI_Send(c_block, BLOCK_SIZE*BLOCK_SIZE, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
      free(buffer);
      free(c_block);
    }
  }
  MPI_Finalize();
}
