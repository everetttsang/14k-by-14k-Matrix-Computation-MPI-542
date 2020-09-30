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
#define N_SIZE 4
//define max double size for each element
#define MAX_DATA_SIZE 10
//specify the number of nodes to use for computation @ the full sizeof
//an additional node is needed to compute the remaining Calculations
//another additional node is needed to receive the resulting segments from the computation nodes and write into 1 results matrix
//#define NUM_NODES 2  - NUM NODES ONLY SPECIFIED FOR LINEAR MATRIX MULTIPLICATION

//define the size of a subsequent square block matrix. Will contain BLOCK_SIZE by BLOCK_SIZE elements
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

//loads a block into of data into an indepenent array BLOCK_SIZExBLOCK_SZIE matrix
void load_block(double* input, double* output, int block_no){
  // int block_i;
  int NUM_BLOCKS = (N_SIZE/BLOCK_SIZE)*(N_SIZE/BLOCK_SIZE);
  // for(block_i=0; i<NUM_BLOCKS; block_i++){
  //
  // }
  int start_index;
  if((block_no%(N_SIZE/BLOCK_SIZE) ==0)) start_index = block_no*BLOCK_SIZE*BLOCK_SIZE;
  else{
    start_index = ((block_no-(block_no%(N_SIZE/BLOCK_SIZE)))*BLOCK_SIZE*BLOCK_SIZE)+((block_no%(N_SIZE/BLOCK_SIZE))*BLOCK_SIZE);
  }
  //printf("Load block %d. Starting index: %d\n", block_no, start_index);
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
void buf_zero(double* input){
  int i;
  for(i=0; i< sizeof(input); i++){
    input[i]= 0.0;
  }
}


//write a 1D array into a block
void write_block(double* input, double* output, int block_no){
  int NUM_BLOCKS = (N_SIZE/BLOCK_SIZE)*(N_SIZE/BLOCK_SIZE);
  int start_index;
  if((block_no%(N_SIZE/BLOCK_SIZE) ==0)) start_index = block_no*BLOCK_SIZE*BLOCK_SIZE;
  else{
    start_index = ((block_no-(block_no%(N_SIZE/BLOCK_SIZE)))*BLOCK_SIZE*BLOCK_SIZE)+((block_no%(N_SIZE/BLOCK_SIZE))*BLOCK_SIZE);
  }
  //printf("Load block %d. Starting index: %d\n", block_no, start_index);
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
    //printf("Loaded block a:%d, and block b:%d for computation of block %d Row%d Col%d\n", (row*num_blocks)+i, col+(num_blocks*i),element,row,col);
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

  double start_time = MPI_Wtime();
  printf("Start %f\n", start_time);
  printa(a, N_SIZE);
  printa(b, N_SIZE);
  printa(d, N_SIZE);

  double* buffer = (double*) malloc(2*blocks_length*BLOCK_SIZE*BLOCK_SIZE*sizeof(double));



  //compute_matrix(a,b,c_block,0,BLOCK_SIZE);
  //write_block(c_block, d,0);
  //start sending block data
  // int i;
  // for(i=0; i<N_SIZE*BLOCK_SIZE; i++){
  //      buffer[i]= a[i];
  // }
  row_load(a, buffer, 0);

  // int h;
  // for(h=0; h<blocks_length; h++){
  //      load_block(b, temp, 0+(blocks_length*h));
  //      write_block(temp, buffer, blocks_length+h);
  // }
  col_load(b, buffer, 0);
  int printi;
  for(printi=0; printi< sizeof(buffer); printi++){
       printf("%f\t", buffer[printi]);
       if (printi%N_SIZE==0) printf("\n");
  }
  compute_matrix(buffer, c_block, 0, blocks_length);
  write_block(c_block, d, 0);
  printa(c_block,BLOCK_SIZE);
  buf_zero(c_block);
  //printa(d, N_SIZE);

  int f;
  for(f=1; f< NUM_BLOCKS; f++){
    int row = (f / blocks_length) ;
    int col = (f % blocks_length) ;
    row_load(a,buffer, row);
    col_load(b,buffer, col);
    compute_matrix(buffer, c_block, f, blocks_length);
    printa(c_block, BLOCK_SIZE);
    buf_zero(c_block);
  }
  //receive block data
  for(j=0; j<NUM_BLOCKS; j++){
      //MPI_Recv(&c[j*(BLOCK_SIZE*BLOCK_SIZE)], BLOCK_SIZE*BLOCK_SIZE, MPI_DOUBLE, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      //write_block(&c[j*(BLOCK_SIZE*BLOCK_SIZE)], d,j);
  }
  //stop timeval
  gettimeofday(&end, NULL);
  long int duration = (end.tv_sec*1e6 + end.tv_usec) - (start.tv_sec*1e6 + start.tv_usec);
  //printf("Calculation Run Time: %d microseconds\n");
  double end_time = MPI_Wtime();
  //printf("End %f\n", end_time);
  //print a , b input matrices and the resulting matrix d
  //printa(a,N_SIZE);
  //printa(b,N_SIZE);
  //printa(d,N_SIZE);
  //check_output(a, b, d);
  //printf("End of checktime %f\n", MPI_Wtime());
  free(a);
  free(b);
  free(c);
  free(d);
}
// int i;
// for(i=1; i<NUM_BLOCKS; i++){
//   if(world_rank==i){
//     double* c_block = (double* ) malloc(BLOCK_SIZE*BLOCK_SIZE*sizeof(double));
//
//     compute_matrix(a,b,c_block,i,BLOCK_SIZE);
//     //printa(c_block, BLOCK_SIZE);
//     MPI_Send(c_block, BLOCK_SIZE*BLOCK_SIZE, MPI_DOUBLE, NUM_BLOCKS, 0, MPI_COMM_WORLD);
//     free(c_block);
//   }
// }


  //start time

  // LINEAR MATRIX MULTIPLICATION CODE BELOW-----------------------
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

//  free(a);
  //free(b);
  //free(c);
  //free(d);
}
