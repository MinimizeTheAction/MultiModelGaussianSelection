/* 
 *----------------------------------------------------------------------------------------
 * Function:
 *     This program calculates the unbiased sample autocorrelation for a time series
 *     x with N elements for each parameter of a chain file.
 *     
 *     rho(l) = gamma(l)/gamma(0)
 * 	   where rho is the autocorrelation, l is the lag and gamma is defined by
 *     
 *     gamma(l) = 1/(N-l)SUM(i = 1 to N-l)[(x_(i+l) - x_avg)(x_i-x_avg)]
 *
 *     gamma(0) is just the variance. x_avg is the average of the time series
 *     The autocorrelation length is return and defined where rho(l) drops
 *     to about 0.01.
 * ---------------------------------------------------------------------------------------
 * Notes:
 *    1) Greatly inspired by: http://www.csee.usf.edu/~kchriste/tools/autoc.c
 *    2) Refer to 'test_data.dat' to see the file format that one should have.
 *			i.e. no string/characters at top of relevant columns, format of 
 *			columns before the first_column are irrelevant.
 *       
 * ---------------------------------------------------------------------------------------
 * Example Input: ./auto_corr 'chain.dat' 4 2
 				  ./auto_corr [chain file] [no parameters] [first column]
 * ---------------------------------------------------------------------------------------
 * Example Output:
 *  			 -------------------------------------------------------------------------
 *
 *				 Number of samples: 5000
 *
 *				 Created file: autocorrelation_data_file_1.txt
 *
 *				 -------------------------------------------------------------------------
 * ---------------------------------------------------------------------------------------
 * Build: gcc auto_corr.c -lgsl -lm -o auto_corr
 * ---------------------------------------------------------------------------------------
 * Author: Travis Robson (TR)
 * ---------------------------------------------------------------------------------------
 * History:
 *     TR (12/9/16)  - Genesis
 *     TR (12/11/16) - Generates autocorrelation to an output file though with unwanted 
 *						format. Still no autocorrelation length implementation. 
 * ---------------------------------------------------------------------------------------
 * Test:
 *     1) Need to see what a max NUM_LAG should be.
 *     2) Which estimation of autocorrelation length is better?
 * ---------------------------------------------------------------------------------------
 * Development:
 *     1) compute auto_correlation_length function
 *         a) first attempt just where it falls to 0.01
 *         b) do an exponential fit and find scale factor 
 *     2) Make sure you make test data for this to correspond to note 2.
 *     3) Some rigorous testing is needed. i.e. compare output against my
 *			python function which does the same.
 * ---------------------------------------------------------------------------------------
 */
 
/*------------ Include Files -----------------------------------------------------------*/
#include <stdio.h>              // Needed for printf() and feof()
#include <math.h>               // Needed for pow()
#include <stdlib.h>             // Needed for exit() and atof()
#include <string.h>             // Needed for strcmp()
#include <gsl/gsl_matrix.h>     // Needed for matrix manipulations
#include <gsl/gsl_vector.h>     // Needed for vector manipulations
#include <gsl/gsl_statistics.h> // Needed for mean and variance

/*------------ Defines -----------------------------------------------------------------*/
#define MAX_SIZE  10000000   // Maximum size of the time series array
#define NUM_LAG   1000       // Number of lags to calculate for

/*------------ Function Prototypes -----------------------------------------------------*/
double compute_autocorrelation(int lag,gsl_vector * vec); // compute the autocorrelation for lag 
int compute_autocorr_length(gsl_vector * vec);

/*------------ Globals -----------------------------------------------------------------*/
long int N,M;
double Mean;
double Variance;

/*
 *----------------------------------------------------------------------------------------
 * Main program
 *----------------------------------------------------------------------------------------
 */
int main(int argc, char* argv[]) {

	//--------Initialization--------------------------------------------------------------
	double ac_value;      // computed autocorrelation value
	int i,j;              // Loop counter
	int col;              // current column vector

	gsl_matrix * chains, * autocorr_matrix;
	gsl_vector_view chain, chain_autocorr ;
	
	
	char filename[100];
	FILE* in_file;	    // input file
	FILE* out_file;     // output file
	
	int no_params;       // number of parameters to calculate autocorrelation for
	int first_column;    // Which column first corresponds to a chain
	int ch;	             // to determine number of samples in file
	int autocorr_length; // number of iterations needed to drop off to 0.01 ish.
	
	printf("-------------------------------------------------------------------------\n");
	//--------Check that there are the correct number of arguments passed-----------------
	if(argc != 4) { 
		printf("usage: ./auto_corr chainfile no_params first_column \n");
		exit(1); // 0 means success typically, non-zero indicates an error
	}
	
	//--------Extract arguments-----------------------------------------------------------
	sprintf(filename,"%s",argv[1]); // convert input file to string
	in_file = fopen(filename,"rb");  // open input file for reading
	
	no_params = atoi(argv[2]);	    
	first_column = atoi(argv[3]);
	
	//--------What is the number of samples in chain file?--------------------------------
	N = 0; // Initialize count
	M = 0;
	while(!feof(in_file)) {
		ch = fgetc(in_file);
		if(ch == '\n'){
			N++;
		}
		if(ch == ' '){
			M++; // to be used to determine the number of columns
		}
	}
	
	M = M/N; // convert the M could to number of columns
	
	printf("\nNumber of samples: %li\n", N); // print number of samples
	printf("Number of columns: %li\n\n", M); // print out number of columns
	if (N > MAX_SIZE) { // throw error if there are too many samples
		printf("ERROR - Too many samples! MAX_SIZE = %i", MAX_SIZE);
		exit(2);
	}
	rewind(in_file);
	
	//--------Generate a gsl matrix from the chains---------------------------------------
	chains = gsl_matrix_alloc(N, M);                   // allocate memory for gsl_matrix(rows,cols)
	autocorr_matrix = gsl_matrix_alloc(N,no_params);   // allocate memory for gsl_matrix(rows,cols)
	gsl_matrix_fscanf(in_file, chains);        // read in chains to the gsl_matrix
	// gsl_matrix_fprintf(stdout,chains,"%f");
	fclose(in_file);
	
	// create output file
	sprintf(filename,"autocorrelation_%s",argv[1]);
	out_file = fopen(filename,"wb");
	printf("Created file: %s \n\n",filename);
	
	//--------Perform operations on each series-------------------------------------------
	
	for (i=0;i<no_params;i++) {
		col = i + first_column; // current column vector
		chain = gsl_matrix_column(chains, col); // extract a column vector, i.e. chain
		// gsl_vector_fprintf(stdout,&chain.vector,"%f"); // for testing
		
		// calculate the mean and variance
		Mean = gsl_stats_mean(chain.vector.data,chain.vector.stride,chain.vector.size);
		Variance = gsl_stats_variance(chain.vector.data,chain.vector.stride,chain.vector.size);
		
		gsl_matrix_set(autocorr_matrix,0,i,1.0); // store lag 0, 1.0 = Variance/Variance
		
		for (j=1;j<N;j++) {
			ac_value = compute_autocorrelation(j,&chain.vector);
			gsl_matrix_set(autocorr_matrix,j,i,ac_value); // store in autocorrelation value in matrix
		}
		chain_autocorr = gsl_matrix_column(autocorr_matrix, col);
		autocorr_length = compute_autocorr_length(&chain_autocorr.vector);
		printf("Autocorrelation length %i: %i\n",i+1,autocorr_length);
		fprintf(out_file,"%i ",autocorr_length);
	}
	fprintf(out_file,"\n");
	
	// print results to out_file
	for (j=1;j<N;j++) {
		for (i=0;i<no_params;i++) {
			fprintf(out_file,"%f ",gsl_matrix_get(autocorr_matrix,j,i));
		}
		fprintf(out_file,"\n");
	}
	
	
	gsl_matrix_free(chains);           // free the chain gsl_matrix
	gsl_matrix_free(autocorr_matrix);
	fclose(out_file);
	printf("\n\nauto_corr completed.\n");
	printf("-------------------------------------------------------------------------\n");
	return 0;
}

double compute_autocorrelation(int lag, gsl_vector * chain) {
	// Refer to formula at top of program
	double autocv;
	int k;
	
	// loop to compute autocorrelation
	autocv = 0.0;
	for (k=0;k<(N-lag);k++) {
		autocv += ( gsl_vector_get(chain,k+lag)-Mean )*( gsl_vector_get(chain,k)-Mean );
	}
	
	return autocv/(N-lag)/Variance;
}

int compute_autocorr_length(gsl_vector * chain) {
	int i = 0;
	float drop_off = 0.001;
	
	while (gsl_vector_get(chain,i)>drop_off && i < N) {i++;}

	return i;
}


	
	
















 *----------------------------------------------------------------------------------------
 * Function:
 *     This program calculates the unbiased sample autocorrelation for a time series
 *     x with N elements for each parameter of a chain file.
 *     
 *     rho(l) = gamma(l)/gamma(0)
 * 	   where rho is the autocorrelation, l is the lag and gamma is defined by
 *     
 *     gamma(l) = 1/(N-l)SUM(i = 1 to N-l)[(x_(i+l) - x_avg)(x_i-x_avg)]
 *
 *     gamma(0) is just the variance. x_avg is the average of the time series
 *     The autocorrelation length is return and defined where rho(l) drops
 *     to about 0.01.
 * ---------------------------------------------------------------------------------------
 * Notes:
 *    1) Greatly inspired by: http://www.csee.usf.edu/~kchriste/tools/autoc.c
 *    2) Refer to 'test_data.dat' to see the file format that one should have.
 *			i.e. no string/characters at top of relevant columns, format of 
 *			columns before the first_column are irrelevant.
 *       
 * ---------------------------------------------------------------------------------------
 * Example Input: ./auto_corr 'chain.dat' 4 2
 				  ./auto_corr [chain file] [no parameters] [first column]
 * ---------------------------------------------------------------------------------------
 * Example Output:
 *  			 -------------------------------------------------------------------------
 *
 *				 Number of samples: 5000
 *
 *				 Created file: autocorrelation_data_file_1.txt
 *
 *				 -------------------------------------------------------------------------
 * ---------------------------------------------------------------------------------------
 * Build: gcc auto_corr.c -lgsl -lm -o auto_corr
 * ---------------------------------------------------------------------------------------
 * Author: Travis Robson (TR)
 * ---------------------------------------------------------------------------------------
 * History:
 *     TR (12/9/16)  - Genesis
 *     TR (12/11/16) - Generates autocorrelation to an output file though with unwanted 
 *						format. Still no autocorrelation length implementation. 
 * ---------------------------------------------------------------------------------------
 * Test:
 *     1) Need to see what a max NUM_LAG should be.
 *     2) Which estimation of autocorrelation length is better?
 * ---------------------------------------------------------------------------------------
 * Development:
 *     1) compute auto_correlation_length function
 *         a) first attempt just where it falls to 0.01
 *         b) do an exponential fit and find scale factor 
 *     2) Make sure you make test data for this to correspond to note 2.
 *     3) Some rigorous testing is needed. i.e. compare output against my
 *			python function which does the same.
 * ---------------------------------------------------------------------------------------
 */
 
/*------------ Include Files -----------------------------------------------------------*/
#include <stdio.h>              // Needed for printf() and feof()
#include <math.h>               // Needed for pow()
#include <stdlib.h>             // Needed for exit() and atof()
#include <string.h>             // Needed for strcmp()
#include <gsl/gsl_matrix.h>     // Needed for matrix manipulations
#include <gsl/gsl_vector.h>     // Needed for vector manipulations
#include <gsl/gsl_statistics.h> // Needed for mean and variance

/*------------ Defines -----------------------------------------------------------------*/
#define MAX_SIZE  10000000   // Maximum size of the time series array
#define NUM_LAG   1000       // Number of lags to calculate for

/*------------ Function Prototypes -----------------------------------------------------*/
double compute_autocorrelation(int lag,gsl_vector * vec); // compute the autocorrelation for lag 
int compute_autocorr_length(gsl_vector * vec);

/*------------ Globals -----------------------------------------------------------------*/
long int N,M;
double Mean;
double Variance;

/*
 *----------------------------------------------------------------------------------------
 * Main program
 *----------------------------------------------------------------------------------------
 */
int main(int argc, char* argv[]) {

	//--------Initialization--------------------------------------------------------------
	double ac_value;      // computed autocorrelation value
	int i,j;              // Loop counter
	int col;              // current column vector

	gsl_matrix * chains, * autocorr_matrix;
	gsl_vector_view chain, chain_autocorr ;
	
	
	char filename[100];
	FILE* in_file;	    // input file
	FILE* out_file;     // output file
	
	int no_params;       // number of parameters to calculate autocorrelation for
	int first_column;    // Which column first corresponds to a chain
	int ch;	             // to determine number of samples in file
	int autocorr_length; // number of iterations needed to drop off to 0.01 ish.
	
	printf("-------------------------------------------------------------------------\n");
	//--------Check that there are the correct number of arguments passed-----------------
	if(argc != 4) { 
		printf("usage: ./auto_corr chainfile no_params first_column \n");
		exit(1); // 0 means success typically, non-zero indicates an error
	}
	
	//--------Extract arguments-----------------------------------------------------------
	sprintf(filename,"%s",argv[1]); // convert input file to string
	in_file = fopen(filename,"rb");  // open input file for reading
	
	no_params = atoi(argv[2]);	    
	first_column = atoi(argv[3]);
	
	//--------What is the number of samples in chain file?--------------------------------
	N = 0; // Initialize count
	M = 0;
	while(!feof(in_file)) {
		ch = fgetc(in_file);
		if(ch == '\n'){
			N++;
		}
		if(ch == ' '){
			M++; // to be used to determine the number of columns
		}
	}
	
	M = M/N; // convert the M could to number of columns
	
	printf("\nNumber of samples: %li\n", N); // print number of samples
	printf("Number of columns: %li\n\n", M); // print out number of columns
	if (N > MAX_SIZE) { // throw error if there are too many samples
		printf("ERROR - Too many samples! MAX_SIZE = %i", MAX_SIZE);
		exit(2);
	}
	rewind(in_file);
	
	//--------Generate a gsl matrix from the chains---------------------------------------
	chains = gsl_matrix_alloc(N, M);                   // allocate memory for gsl_matrix(rows,cols)
	autocorr_matrix = gsl_matrix_alloc(N,no_params);   // allocate memory for gsl_matrix(rows,cols)
	gsl_matrix_fscanf(in_file, chains);        // read in chains to the gsl_matrix
	// gsl_matrix_fprintf(stdout,chains,"%f");
	fclose(in_file);
	
	// create output file
	sprintf(filename,"autocorrelation_%s",argv[1]);
	out_file = fopen(filename,"wb");
	printf("Created file: %s \n\n",filename);
	
	//--------Perform operations on each series-------------------------------------------
	
	for (i=0;i<no_params;i++) {
		col = i + first_column; // current column vector
		chain = gsl_matrix_column(chains, col); // extract a column vector, i.e. chain
		// gsl_vector_fprintf(stdout,&chain.vector,"%f"); // for testing
		
		// calculate the mean and variance
		Mean = gsl_stats_mean(chain.vector.data,chain.vector.stride,chain.vector.size);
		Variance = gsl_stats_variance(chain.vector.data,chain.vector.stride,chain.vector.size);
		
		gsl_matrix_set(autocorr_matrix,0,i,1.0); // store lag 0, 1.0 = Variance/Variance
		
		for (j=1;j<N;j++) {
			ac_value = compute_autocorrelation(j,&chain.vector);
			gsl_matrix_set(autocorr_matrix,j,i,ac_value); // store in autocorrelation value in matrix
		}
		chain_autocorr = gsl_matrix_column(autocorr_matrix, col);
		autocorr_length = compute_autocorr_length(&chain_autocorr.vector);
		printf("Autocorrelation length %i: %i\n",i+1,autocorr_length);
		fprintf(out_file,"%i ",autocorr_length);
	}
	fprintf(out_file,"\n");
	
	// print results to out_file
	for (j=1;j<N;j++) {
		for (i=0;i<no_params;i++) {
			fprintf(out_file,"%f ",gsl_matrix_get(autocorr_matrix,j,i));
		}
		fprintf(out_file,"\n");
	}
	
	
	gsl_matrix_free(chains);           // free the chain gsl_matrix
	gsl_matrix_free(autocorr_matrix);
	fclose(out_file);
	printf("\n\nauto_corr completed.\n");
	printf("-------------------------------------------------------------------------\n");
	return 0;
}

double compute_autocorrelation(int lag, gsl_vector * chain) {
	// Refer to formula at top of program
	double autocv;
	int k;
	
	// loop to compute autocorrelation
	autocv = 0.0;
	for (k=0;k<(N-lag);k++) {
		autocv += ( gsl_vector_get(chain,k+lag)-Mean )*( gsl_vector_get(chain,k)-Mean );
	}
	
	return autocv/(N-lag)/Variance;
}

int compute_autocorr_length(gsl_vector * chain) {
	int i = 0;
	float drop_off = 0.001;
	
	while (gsl_vector_get(chain,i)>drop_off && i < N) {i++;}

	return i;
}


	
	















