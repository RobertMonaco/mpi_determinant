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
  *b = temp;
  *a = *b;
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
  double logdet = 0.0;
  double pivot_val = 0.0; 
  int pivot;
  double * pivot_row;
  double * pivot_col;
  int row_shift;
  int j;

  printf("%d: 1\n", my_rank);
  // Start the algorithm
  for(int row = 0; row < local_Nrow - 1; row++){
    for(int p = 0; p < n; p++){
      
      if(my_rank == p){
        pivot_row = local_A[row];
        pivot_val = -1;
        j = -1;

        //get max absolute value of pivot row
        for(int col=0; col < local_Ncol; col++){
          if(abs(pivot_row[col]) > pivot_val){
            pivot_val = abs(pivot_row[col]);
            j = col;
          }
        }

        // divide pivot row by pivot value
        for (int col=0; col<local_Ncol; col++){
          pivot_row[col] = pivot_row[col]/pivot_val;
        }
        
        // Swap pivot row
        swap_double(&(pivot_row[j]), &(pivot_row[local_Ncol - 1]));
        if(pivot_val == 0){
          local_logdet += log2(abs(pivot_val));
        }
        row_shift = 1;

      } else {
        row_shift = 0;
      }

      printf("%d: 2\n", my_rank);

      // Broadcast pivot_row and j from proc p to all other procs
      MPI_Bcast(&pivot_row, N, MPI_DOUBLE, p, comm);
      MPI_Bcast(&pivot, 1, MPI_INT, p, comm);
      
      printf("%d: 3\n", my_rank);

      pivot_col = (double*)malloc(sizeof(double)*local_Nrow);
      for(int i = row+row_shift; i < local_Nrow; i++){
        pivot_col[i] = local_A[i][pivot];
      }

      printf("%d: 4\n", my_rank);

      for( int i = row+row_shift; i < local_Nrow; i++){
        swap_double(&(local_A[i][pivot]), &(local_A[i][local_Ncol - 1]));
      }

      local_Ncol--;

      printf("%d: 5\n", my_rank);
      
      for (int i = row+row_shift; i < local_Nrow; i++){
        for (int k = 0; k < local_Ncol; k++){
          local_A[i][k] -= (pivot_col[i]*pivot_col[k]);
        }
      }
    }
  }
  printf("%d: 6\n", my_rank);
  printf("3\n");
  MPI_Reduce(&local_logdet,
    &logdet,
    1,
    MPI_DOUBLE,
    MPI_SUM,
    0,
    comm);
  printf("4\n");
  return logdet;
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
    if(a[i][i] == 0){
      logdet += log2(abs(a[i][i]));
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
  double ** a;
  double log_det;

  printf("%d\n",my_rank);
  if(my_rank == 0){
    N = atoi(argv[1]);
    a = malloc(N*N*sizeof(double));
    char f_name[50];
    int i,j;

    //Create filename
    sprintf(f_name,"m0016x0016.bin");
    printf("Reading array file %s of size %dx%d\n",f_name,N,N);
    
    //Open matrix binary file
    FILE *datafile = fopen(f_name,"rb");
    //Read elements into matrix a from binary
    for (i = 0; i < N; i++)
      for (j = 0; j < N; j++)
      {
          fread(&(a[i][j]),sizeof(double),1,datafile);
          printf("a[%d][%d]=%f\n",i,j,a[i][j]);
      }
    printf("Matrix has been read.\n");
  }

  //broadcast value of N from process 0
  MPI_Bcast(&N, 1, MPI_INT, 0, comm);
  
  //allocate local Nrow for each process
  int local_Nrow = N / comm_sz;
  //allocate room for local matrix
  double** local_A = malloc(local_Nrow*N*sizeof(double));

  // Scatter A to all other processes
  MPI_Scatter(get_ptr(a), 
    local_Nrow*N, 
    MPI_DOUBLE, 
    get_ptr(local_A),
    local_Nrow*N,
    MPI_DOUBLE,
    0,
    comm);

  printf("%d: Scatter finished\n",my_rank);

  //get log det for each local matrix
  log_det = logdet(local_Nrow, N, local_A, my_rank, comm_sz, comm);
  
  // Gather the local A matrices into proc 0
  MPI_Gather(get_ptr(local_A),
    local_Nrow*N,
    MPI_DOUBLE,
    get_ptr(a),
    local_Nrow*N,
    MPI_DOUBLE,
    0,
    comm);
  
  if(my_rank == 0){
    gauss_elim(a, N);
    log_det += serial_logdet(a, N);
  }

  MPI_Finalize();
  return 0;
}
