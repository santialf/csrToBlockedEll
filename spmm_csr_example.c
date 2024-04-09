#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <zlib.h>

#include <string.h>
#include <time.h>

#include "mmio.h"

// Node structure for the linked list
typedef struct Node {
    int data;
    struct Node* next;
} Node;

// Function to insert a new node into the linked list in ascending order
void insertNode(Node** head, int value) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    if (newNode == NULL) {
        fprintf(stderr, "Memory allocation failed for a new node\n");
        exit(1);
    }
    newNode->data = value;
    newNode->next = NULL;

    Node* current = *head;
    Node* prev = NULL;

    while (current != NULL && current->data < value) {
        prev = current;
        current = current->next;
    }

    if (prev == NULL) {
        // Insert at the beginning
        newNode->next = *head;
        *head = newNode;
    } else {
        // Insert in the middle or at the end
        prev->next = newNode;
        newNode->next = current;
    }
}

float* createRandomArray(int n) {
    float* array = malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) { 
        array[i] = 1;
    }

    return array;
}

/* Finds the possible amount of column blocks the matrix can have */
int findMaxNnz(int *rowPtr, int *colIndex, int num_rows, int block_size) {

    int max = 0;
    int num_blocks = num_rows / block_size;
    if (num_rows % block_size != 0)
        num_blocks++;

    for(int i=0; i < num_rows; i++) {
        int flag=0;
		int number_of_cols = 0;
		for(int j=rowPtr[i]; j<rowPtr[i+1]; j++) {
            if (flag <= colIndex[j]) {
                flag = (colIndex[j]/block_size) * block_size + block_size;
                number_of_cols++;
            }
        }
        if (number_of_cols > max)
            max = number_of_cols;
	}

    return max*block_size;
}

int *createBlockIndex(int *rowPtr, int *colIndex, int num_rows, int block_size, int ell_cols) {

    long int mb = num_rows/block_size, nb = ell_cols/block_size;
    if (num_rows % block_size != 0)
        mb++;
    printf("%ld %ld\n%ld\n", mb, nb, nb*mb);
    int *hA_columns = (int *)calloc(nb * mb, sizeof(int));
    int ctr = 0;

    memset(hA_columns, -1, nb * mb * sizeof(int));

    for(int i=0; i<mb; i++) {

        int *flag = (int *)calloc(mb, sizeof(int));
        Node* block_list = NULL;

        for (int j = 0; j < block_size; j++) {
            int id = block_size*i + j;
            int index = 0;
            if (id >= num_rows)
                break;

            for(int k=rowPtr[id]; k<rowPtr[id+1]; k++) {    
                index = (colIndex[k]/block_size);
                if (flag[index] == 0) {
                    insertNode(&block_list, index);
                    flag[index] = 1;
                }
            }
        }
        
        while (block_list != NULL) {
            Node *temp = block_list;
            hA_columns[ctr++] = block_list->data;
            block_list = block_list->next;
            free(temp);
        }
        ctr = i*nb+nb;
        free(flag);
    }

    return hA_columns; 
}

float *createValueIndex(int *rowPtr, int *colIndex, float *values, int *hA_columns, int num_rows, int block_size, int ell_cols) {

    float *hA_values = (float *)calloc(num_rows * ell_cols, sizeof(int));
    long int mb = num_rows/block_size;
    if (num_rows % block_size != 0)
        mb++;

    memset(hA_columns, 0, num_rows * ell_cols * sizeof(int));

    for (int i=0; i<mb;i++){
        for (int j = 0; j < block_size; j++) {
            int id = block_size*i + j;
            int flag = 0;
            int ctr = 0;
            if (id >= num_rows)
                break;

            for(int k=rowPtr[id]; k<rowPtr[id+1]; k++) {  
                hA_values[(colIndex[k] - (colIndex[k]/block_size) * block_size) + block_size*ctr] = values[k];
                if (flag <= colIndex[k]) {
                    flag = (colIndex[k]/block_size) * block_size + block_size;
                    ctr++;
                }
            }
        }
    }
    
    return hA_values;
}

int main(int argc, char *argv[]) {

    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int A_num_rows, A_num_cols, nz, A_nnz;
    int i = 0, *I_complete, *J_complete;
    float *V_complete;

    /*******************************************************************/
    if ((f = fopen(argv[1], "r")) == NULL)
    {
        printf("Could not locate the matrix file. Please make sure the pathname is valid.\n");
        exit(1);
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }
    matcode[4] = '\0';
    
    if ((ret_code = mm_read_mtx_crd_size(f, &A_num_rows, &A_num_cols, &nz)) != 0)
    {
        printf("Could not read matrix dimensions.\n");
        exit(1);
    }
    
    if ((strcmp(matcode, "MCRG") == 0) || (strcmp(matcode, "MCIG") == 0) || (strcmp(matcode, "MCPG") == 0) || (strcmp(matcode, "MCCG") == 0))
    {

        I_complete = (int *)calloc(nz, sizeof(int));
        J_complete = (int *)calloc(nz, sizeof(int));
        V_complete = (float *)calloc(nz, sizeof(float));

        for (i = 0; i < nz; i++)
        {
            if (matcode[2] == 'P') {
                fscanf(f, "%d %d", &I_complete[i], &J_complete[i]);
                V_complete[i] = 1;
            }  
            else {
                fscanf(f, "%d %d %f", &I_complete[i], &J_complete[i], &V_complete[i]);
            } 
            fscanf(f, "%*[^\n]\n");
            /* adjust from 1-based to 0-based */
            I_complete[i]--;
            J_complete[i]--;
        }
    }

    /* If the matrix is symmetric, we need to construct the other half */

    else if ((strcmp(matcode, "MCRS") == 0) || (strcmp(matcode, "MCIS") == 0) || (strcmp(matcode, "MCPS") == 0) || (strcmp(matcode, "MCCS") == 0) || (strcmp(matcode, "MCCH") == 0) || (strcmp(matcode, "MCRK") == 0) || (matcode[0] == 'M' && matcode[1] == 'C' && matcode[2] == 'P' && matcode[3] == 'S'))
    {

        I_complete = (int *)calloc(2 * nz, sizeof(int));
        J_complete = (int *)calloc(2 * nz, sizeof(int));
        V_complete = (float *)calloc(2 * nz, sizeof(float));

        int i_index = 0;

        for (i = 0; i < nz; i++)
        {
            if (matcode[2] == 'P') {
                fscanf(f, "%d %d", &I_complete[i], &J_complete[i]);
                V_complete[i] = 1;
            }
            else {
                fscanf(f, "%d %d %f", &I_complete[i], &J_complete[i], &V_complete[i]);
            }
                
            fscanf(f, "%*[^\n]\n");

            if (I_complete[i] == J_complete[i])
            {
                /* adjust from 1-based to 0-based */
                I_complete[i]--;
                J_complete[i]--;
            }
            else
            {
                /* adjust from 1-based to 0-based */
                I_complete[i]--;
                J_complete[i]--;
                J_complete[nz + i_index] = I_complete[i];
                I_complete[nz + i_index] = J_complete[i];
                V_complete[nz + i_index] = V_complete[i];
                i_index++;
            }
        }
        nz += i_index;
    }
    else
    {
        printf("This matrix type is not supported: %s \n", matcode);
        exit(1);
    }

    /* sort COO array by the rows */
    if (!isSorted(J_complete, I_complete, nz)) {
        quicksort(J_complete, I_complete, V_complete, nz);
    }
    
    /* Convert from COO to CSR */
    int *rowPtr = (int *)calloc(A_num_rows + 1, sizeof(int));
    int *colIndex = (int *)calloc(nz, sizeof(int));
    float *values = (float *)calloc(nz, sizeof(float));

    for (i = 0; i < nz; i++) {
        colIndex[i] = J_complete[i];
        values[i] = V_complete[i];
        rowPtr[I_complete[i] + 1]++;
    }
    for (i = 0; i < A_num_rows + 1; i++) {
        rowPtr[i + 1] += rowPtr[i];
    }
    A_nnz = nz;
    /*******************************************************************/

    /* biggest number of nnzs in a row */
    int A_ell_blocksize = 2;
    int A_ell_cols = findMaxNnz(rowPtr, colIndex, A_num_rows, A_ell_blocksize);
    int A_num_blocks = A_ell_cols * A_num_rows / (A_ell_blocksize * A_ell_blocksize);
    int *hA_columns = createBlockIndex(rowPtr, colIndex, A_num_rows, A_ell_blocksize, A_ell_cols);
    float *hA_values = createValueIndex(rowPtr, colIndex, values, hA_columns, A_num_rows, A_ell_blocksize, A_ell_cols);

    for(int i= 0; i < A_num_rows;i++){
        printf("%f\n", hA_values[i]);
    }
    printf("max = %d\n", A_ell_cols);
    
    return 0;
}