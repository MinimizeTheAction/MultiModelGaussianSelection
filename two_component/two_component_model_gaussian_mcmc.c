/* 
 *----------------------------------------------------------------------------------------
 * Function: This uses a standard Metropolis Hastings algorithm implementation to generate
 *				a sample from the posterior of a chi squared likelihood, i.e.
 *					chi_sq = SUM(k=1 to N) [(data_k - z_k)^2]/(2*variance_k)
 * 				where the model z(t) = Amp*exp(-(x-x_avg)^2 / (2*sigma^2))
 *
 *				This outputs a data file with samples corresponding to Amp, x_avg, and sigma
 *     
 *----------------------------------------------------------------------------------------
 * Notes:
 *    1) I have set 2*variance_k == 1
 *	  2) acceptance IDs: 
 *				0 - rejected
 *				1 - accepted
 *    3) output file formated in the following manner:
 *				[iteration] [log likelihood] [amplitude] [mean] [sigma] [acceptance_id]
 *	  4) I am using a symmetric proposal density so it will always cancel in the
 *			the MH ratio.
 *	  5) Priors are as follows:
 *			amplitude ~ U[0.0,10.0]
 *			t_avg     ~ U[-10.0,10.0]
 *			sigma     ~ U[0.0,10.0]
 *
 *	 
 *----------------------------------------------------------------------------------------
 * Example Input: ./single_component_standard_mcmc data_file.txt 1000 
 *				  ./single_component_standard_mcmc [data file] [iterations]
 *	
 *----------------------------------------------------------------------------------------
 * Example Output:
 *	
 *----------------------------------------------------------------------------------------
 * Build: gcc single_component_standard_mcmc.c -lgsl -lm -o single_component_standard_mcmc
 *----------------------------------------------------------------------------------------
 * Author: Travis Robson (TR)
 *----------------------------------------------------------------------------------------
 * History:
 *     TR (12/12/16)  - Genesis
 *	
 *----------------------------------------------------------------------------------------
 * Test:
 *     1) 
 *
 *----------------------------------------------------------------------------------------
 * Development:
 *		1) 
 *     
 *	
 *----------------------------------------------------------------------------------------
 */
 
 
/*--------Include Files-----------------------------------------------------------------*/
#include <stdio.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_math.h>

/*--------Defines-----------------------------------------------------------------------*/
#define NO_PARAMS 3        // This is the number of parameters in the model
#define PROPOSAL_SIGMA 0.1 // proposal standard deviation

#define amp_lower 0.0
#define amp_upper 10.0
#define avg_lower -10.0
#define avg_upper 10.0
#define sigma_lower 0.0
#define sigma_upper 10.0

/*--------Globals-----------------------------------------------------------------------*/
int no_rows, no_cols;	  // number of rows and columns in data file

/*--------Function Prototypes-----------------------------------------------------------*/
double log_lklhood(gsl_vector *x,gsl_matrix *m);       // Calculates the log likelihood
													   //   given the current and proposed
													   //   parameters.
													
double log_prior_pdf(gsl_vector *x);                   // Calculates the log of the 
													   //   prior probability density 
													   //   function (pdf)
													   
void propose_parameters(gsl_vector *x, gsl_vector *y, gsl_rng *r);    
													   // propose a random variate (rvs)
													   // from the proposal density													   
													   
void initialize_parameters(gsl_vector *x, gsl_rng *r); // Initialize parameters	

double evaluate_model(gsl_vector *x, double t);       // evaluate the model




/*----------------------------------------------------------------------------------------
 *----------------------------------------------------------------------------------------
 * 										Main Program
 *----------------------------------------------------------------------------------------
 *----------------------------------------------------------------------------------------
 */
 
 int main(int argc, char* argv[]) {
 	printf("---------------------------------------------------------------------------");
 	printf("\n\n");
 	/*--------Initialization------------------------------------------------------------*/
 	int i,itr;				  // MCMC iteration counter
 	char ch;				  // for counting of rows and columns lated of input file
 	char filename[100];       // for in_file and out_file specification
 	FILE *in_file;			  // input data file
 	FILE *out_file;           // output sample file
 	double amplitude,t_avg,sigma;
 	
 	gsl_vector *x, *y; 	      // current and proposed parameter vectors
 	gsl_vector_view t, d;     // vectors to store time and data 
 	gsl_matrix *data_matrix;  // matrix to read in stored data
 	
 	double accept_prob;       // acceptance probability for proposed parameters
 	double u;				  // uniformly selected value element of [0,1], standard MH
 	//double proposal_y_x;      // proposal pdf evaluated for x->y
 	//double proposal_x_y;      // proposal pdf evaluated for y->x
 	double lg_lklhood_y;      // log likelihood evaluated for proposed parameters
 	double lg_lklhood_x;	  // log likelihood evaluated for current parameters
 	double lg_prior_x;		  // log prior evaluated at current parameters
 	double lg_prior_y;        // log prior evaluated at proposed parameters
 	int acceptance_id;		  // identifier for acceptance or rejections.
 	double MH_ratio_num; 	  // metropolis hastings ratio numerator
 	double MH_ratio_denom;
 	
 	// memory allocation of parameter vectors
 	x = gsl_vector_alloc(3);
 	y = gsl_vector_alloc(3);
 	
 	// Initialize random generators
 	const gsl_rng_type * T;
	gsl_rng * r;
	
	// Allocate generate and specify type
	r = gsl_rng_alloc(gsl_rng_mt19937); 
	gsl_rng_set(r,1);		// seed is set to 1	
	
	
	
	//---open input file and generate data matrix---
	sprintf(filename,"%s",argv[1]); // convert input file to string
	in_file = fopen(filename,"rb");  // open input file for reading
	
	// how many rows and columns
	no_rows = 0;
	no_cols = 0;
	
	while(!feof(in_file)) { // count rows
		ch = fgetc(in_file);
		if(ch == '\n'){
			no_rows++;
		}
	}
	rewind(in_file);
	ch = 'a';
	while(ch != '\n') { // count columns
		ch = fgetc(in_file);
		if (ch == ' ') {
			no_cols++;
		}
	}
	
	no_cols +=1;			   // convert the M could to number of columns
	rewind(in_file);           // so that it can be scanned to generate a matrix
	printf("Data File:\n     no_rows: %i\n     no_cols: %i\n\n",no_rows,no_cols);
	
	data_matrix = gsl_matrix_alloc(no_rows,no_cols); // allocate memory for data matrix
	gsl_matrix_fscanf(in_file,data_matrix);			 // scan in file to matrix
	fclose(in_file);
	
	// create output file
	sprintf(filename,"%s",argv[3]); // convert input file to string
	out_file = fopen(filename,"wb");  // open input file for reading
	printf("File created: %s",filename);
	
	// set number of iterations from input
	itr = atoi(argv[2]);
	
	
 	// Initialize current parameters
 	initialize_parameters(x,r);
 	
 	/*--------Initialization COMPLETED--------------------------------------------------*/
 	
 	
 	/*--------MCMC Loop-----------------------------------------------------------------*/
 	
 	lg_lklhood_x = log_lklhood(x,data_matrix); // calculate the log likelihood
 	lg_prior_x = log_prior_pdf(x); // calculate the log prior
 	for (i=0;i<itr;i++) {
 		// propose a new set of parameters
 		propose_parameters(x,y,r);
 		
 		// calculate the log likelihood of proposed parameters
 		lg_lklhood_y = log_lklhood(y,data_matrix);
 		
 		// calculate priors on the proposed parameters
 		lg_prior_y = log_prior_pdf(y);
 		
 		MH_ratio_num = lg_prior_y + lg_lklhood_y;
 		MH_ratio_denom = lg_prior_x + lg_lklhood_x;
 		
 		accept_prob = GSL_MIN(1.0, exp(MH_ratio_num-MH_ratio_denom));
 		
 		// Decide whether to accept or not
 		u = gsl_ran_flat(r,0.0,1.0);
 		
 		if (u < accept_prob) {
 			lg_lklhood_x = lg_lklhood_y;
 			lg_prior_x = lg_prior_y;
 			gsl_vector_memcpy(x,y); // set x vector as copy of y vector
 			acceptance_id = 1;
 			//  printf("%i\n",acceptance_id);
 		} else {
 			acceptance_id = 0;
 		}
 		
 		amplitude = gsl_vector_get(x,0);
		t_avg = gsl_vector_get(x,1);
		sigma = gsl_vector_get(x,2);
 		
 		// print to output file
 		fprintf(out_file,"%i %f %f %f %f %i\n",itr,lg_lklhood_x,amplitude,t_avg,sigma,acceptance_id);
 		

 	}
 	
	// close output file
 	fclose(out_file);
 	
 	// Free memory
 	gsl_vector_free(x);
 	gsl_vector_free(y);
 	gsl_rng_free(r);
 	
 	printf("\n\n");
 	printf("---------------------------------------------------------------------------");
 	printf("\n");
 	
 	return 0;
}
 
void initialize_parameters(gsl_vector *x, gsl_rng *r) {
	int i; // loop iterator
	
	for (i=0;i<NO_PARAMS;i++) {
		gsl_vector_set(x,i,gsl_ran_flat(r,0.0,5.0));
	}
	
 	return;
}

void propose_parameters(gsl_vector *x, gsl_vector *y, gsl_rng *r) {
	int i; 		// loop iterator
	double val; // value to set 
	
	for (i=0;i<NO_PARAMS;i++) { // choose rvs from gaussian
		val = gsl_ran_gaussian(r,PROPOSAL_SIGMA)+gsl_vector_get(x,i);
		gsl_vector_set(y,i,val);
	}
	
	return;
}

double log_lklhood(gsl_vector *x,gsl_matrix *data_matrix) {
	int i,j;
	double t,z_t;
	double chi_square;
	
	chi_square = 0.0;
	
	for (i=0;i<no_rows;i++) {
		t = gsl_matrix_get(data_matrix,i,0);
		z_t = evaluate_model(x,t);
		
		chi_square += pow((gsl_matrix_get(data_matrix,i,1) - z_t),2.0);
	}
	
	
	return -chi_square;
}

double evaluate_model(gsl_vector *x, double t) {
	double arg; // argument of exponential for gaussian model
	double amplitude,t_avg,sigma;
	
	// set the parameters
	amplitude = gsl_vector_get(x,0);
	t_avg = gsl_vector_get(x,1);
	sigma = gsl_vector_get(x,2);
	
	arg = -pow((t-t_avg),2.0)/2.0/pow(sigma,2.0);
	return amplitude*exp(arg);
}

double log_prior_pdf(gsl_vector *x) {
	double amplitude,t_avg,sigma;
	
	// set the parameters
	amplitude = gsl_vector_get(x,0);
	t_avg = gsl_vector_get(x,1);
	sigma = gsl_vector_get(x,2);
	
	
	// return hopefully something around machine epsilon, otherwise return 1.0 (some constant)
	if (amplitude < amp_lower || amplitude > amp_upper) { 
		return -1.0e-16;
	} else if (t_avg < avg_lower || t_avg > avg_upper) {
		return -1.0e-16;
	} else if (sigma < sigma_lower || sigma > sigma_upper) {
		return -1.0e-16;
	} else {
		return 1.0;
	}
}



























