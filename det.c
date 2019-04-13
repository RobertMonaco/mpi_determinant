//Robert Monaco
//Daniel Schon
//Cody Degner
//Cody Poteet
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Allocate a 2-dimensional array of doubles in contiguous memory
// This makes it much easier to send and receive with MPI
double** alloc_contiguous(int rows, int cols)
{
  // Allocate one contiguous memory block for data
  double* data = (double*)malloc(rows*cols*sizeof(double));
  // Allocate our matrix
  double** matrix = (double**)malloc(rows*sizeof(double*));
  // Point matrix cells to contiguous memory locations
  for (int i = 0; i < rows; i++)
    matrix[i] = &(data[cols*i]);

  return matrix;
}

// Get a pointer to a matrix so we can send or receive it
double* get_ptr(double** matrix)
{
  return &(matrix[0][0]);
}

// Free memory used by a contiguous matrix
void free_contiguous(double** matrix)
{
  // Free the data array and the nested array itself
  free(matrix[0]);
  free(matrix);
}

// Swap values of double a and b
void swap_double(double* a, double* b)
{
  double temp = *a;
  *a = *b;
  *b = temp;
}

void print_matrix(double** matrix, int rows, int cols)
{
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      printf("%f, ", matrix[i][j]);
    }
    printf("\n");
  }
}

/** Calculate the log determinant of a square matrix 
 *  N: Size of the matrix
 *  n: Number of processes
 *  a: the matrix
 *  my_rank: current process number
 *  comm: the MPI communicator
 */
double logdet(int local_Nrow, int local_Ncol, double** local_A, int my_rank, int n, MPI_Comm comm){
  double local_logdet = 0.0;
  double pivot_val = 0.0; 
  int pivot;
  double * pivot_row;
  double * pivot_col;
  int row_shift;
  
  pivot_row = (double*)malloc(sizeof(double)*local_Ncol);

  // Start the algorithm
  for(int row = 0; row < local_Nrow - 1; row++){
    for(int p = 0; p < n; p++){
      
      if(my_rank == p){

        for (int col = 0; col < local_Ncol; col++)
          pivot_row[col] = local_A[row][col];

        pivot_val = -1;
        pivot = -1;

        //get max absolute value of pivot row
        for(int col=0; col < local_Ncol; col++){
          if(fabs(pivot_row[col]) >= pivot_val){
            pivot_val = fabs(pivot_row[col]);
            pivot = col;
          }
        }

        // divide pivot row by pivot value
        for (int col=0; col<local_Ncol; col++){
          pivot_row[col] = pivot_row[col]/pivot_val;
        }
        
        // Swap pivot row
        pivot_row[pivot] = pivot_row[local_Ncol - 1];

        if(pivot_val != 0){
          local_logdet += log10(pivot_val);
        }
        row_shift = 1;

      } else {
        row_shift = 0;
      }

      // Broadcast pivot_row and j from proc p to all other procs
      MPI_Bcast(pivot_row, local_Ncol, MPI_DOUBLE, p, comm);
      MPI_Bcast(&pivot, 1, MPI_INT, p, comm);
      
      pivot_col = (double*)malloc(sizeof(double)*local_Nrow);

      // Make the pivot column
      for(int i = row+row_shift; i < local_Nrow; i++){
        pivot_col[i] = local_A[i][pivot];
      }

      // 
      for( int i = row+row_shift; i < local_Nrow ; i++){
        local_A[i][pivot] = local_A[i][local_Ncol - 1];
      }

      local_Ncol--;

      for (int i = row+row_shift; i < local_Nrow ; i++){
        for (int k = 0; k < local_Ncol; k++){
          local_A[i][k] -= pivot_col[i]*pivot_row[k];
        }
      }
    }
  }

 return local_logdet;
}

void gauss_elim(double ** a, int N){
  for(int j = 0; j < N; j++) /* loop for the generation of upper triangular matrix*/
  {
    for(int i = 0; i < N; i++)
    {
        if(i > j)
        {
            double c = a[i][j] / a[j][j];
            for(int k = 0; k < N+1; k++)
            {
              a[i][k]= a[i][k] - c * a[j][k];
            }
        }
    }
  }
}

double serial_logdet(double ** a, int N){
  double logdet = 0.0;

  for(int i = 0; i < N; i++){

    if(a[i][i] != 0){
      logdet += log10(fabs(a[i][i]));
    }
  }

  return logdet;
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  //MPI vars
  int comm_sz, my_rank;
  MPI_Comm comm;
  comm = MPI_COMM_WORLD;
  MPI_Comm_size(comm, &comm_sz);
  MPI_Comm_rank(comm, &my_rank);

  //Needed for function call
  int N;
  double ** a, ** a_serial;
  double log_det;
  double local_logdet = 0.0;

  if(my_rank == 0){
    N = atoi(argv[1]);
    char f_name[50];
    int i,j;

    a = alloc_contiguous(N,N);
    a_serial = alloc_contiguous(N,N);

    //Create filename
    
    sprintf(f_name,"m0016x0016.bin");
    
    //Open matrix binary file
    FILE *datafile = fopen(f_name,"rb");
    
    //Read elements into matrix a from binary
    for (i = 0; i < N; i++)
    {
      for (j = 0; j < N; j++)
      {
          fread(&(a[i][j]),sizeof(double),1,datafile);
          a_serial[i][j] = a[i][j];
      }
    }

    print_matrix(a, N, N);
  }

  //broadcast value of N from process 0
  MPI_Bcast(&N, 1, MPI_INT, 0, comm);

  // Allocate a for every other process
  if (my_rank != 0)
    a = alloc_contiguous(N,N);
  
  //allocate local Nrow for each process
  int local_Nrow = N / comm_sz;
  //allocate room for local matrix
  double** local_A = alloc_contiguous(local_Nrow, N);

  // Scatter A to all other processes
  MPI_Scatter(get_ptr(a), 
    local_Nrow*N, 
    MPI_DOUBLE, 
    get_ptr(local_A),
    local_Nrow*N,
    MPI_DOUBLE,
    0,
    comm);

  //get log det for each local matrix
  local_logdet = logdet(local_Nrow, N, local_A, my_rank, comm_sz, comm);
  
  MPI_Reduce(&local_logdet,
    &log_det,
    1,
    MPI_DOUBLE,
    MPI_SUM,
    0,
    comm);
  double ** global_a = alloc_contiguous(comm_sz, comm_sz);

  // Gather the local A matrices into proc 0
  MPI_Gather(&(local_A[local_Nrow-1][0]),
    comm_sz,
    MPI_DOUBLE,
    get_ptr(global_a),
    comm_sz,
    MPI_DOUBLE,
    0,
    comm);
  
  if(my_rank == 0){

    gauss_elim(global_a, comm_sz);
    printf("global_a after elimination: \n");
    print_matrix(global_a, comm_sz, comm_sz);
    double serial2 = serial_logdet(global_a, comm_sz);

    gauss_elim(a_serial, N);
    double serial = serial_logdet(a_serial, N);

    printf("Serial result: %f\n", serial);
    printf("Parallel result: %f\n", log_det);
    printf("Serial result of post-algorithm matrix: %f\n", serial2);
    printf("Parallel result plus serial: %f\n", log_det + serial2);
  }

  MPI_Finalize();
  return 0;
}
