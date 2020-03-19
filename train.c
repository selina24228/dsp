#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "hmm.h"

#ifndef MAX_SEQ_NUM
#	define MAX_SEQ_NUM 10000
#endif

#ifndef MAX_SEQ_LEN
#	define MAX_SEQ_LEN 100
#endif

#define ERR_EXIT(a) { perror(a); exit(0); }

static void forward_algo( double alpha[][MAX_STATE], char* seq, HMM* hmm );
static void backward_algo( double beta[][MAX_STATE], char* seq, HMM* hmm );
static void count_gamma( double gamma[][MAX_STATE], double alpha[][MAX_STATE], double beta[][MAX_STATE],int seq_len, HMM* hmm );
static void count_epsilon( double epsilon[][MAX_STATE], int t, double alpha[][MAX_STATE], double beta[][MAX_STATE], char* seq, HMM* hmm );
static void update_hmm( double gamma_abstract[][MAX_STATE], double epsilon_sum[][MAX_STATE], int seq_num, HMM* hmm );

int main(int argc, char* argv[]){
	if( argc != 5){
		ERR_EXIT("input format error");
	}	
	int iter = atoi( argv[1] );
	char* model_init_path = argv[2];
	char* seq_path = argv[3];
	char* output_model_path = argv[4];
	
	HMM hmm;
	loadHMM( &hmm, model_init_path );

	FILE* fp = open_or_die( seq_path, "r" );
	char seq[MAX_SEQ_NUM][MAX_SEQ_LEN];
	int seq_num = 0;
	while( fgets( seq[seq_num], MAX_SEQ_LEN, fp ) != NULL){
		if(seq[seq_num][ strlen( seq[seq_num] ) - 1 ]=='\n'){
			seq[seq_num][ strlen( seq[seq_num] ) - 1 ] ='\0';
		}
		seq_num++;
	}
	fclose(fp);

	while( iter-- ){
		double gamma_abstract[MAX_OBSERV][MAX_STATE]={0};
		double epsilon_sum[MAX_STATE][MAX_STATE] = {0};
		for(int i = 0; i < seq_num; i++){	
			int len = strlen( seq[i] );

			double alpha[MAX_SEQ_LEN][MAX_STATE]={0};
			forward_algo( alpha, seq[i], &hmm );
		
			double beta[MAX_SEQ_LEN][MAX_STATE]={0};
			backward_algo( beta, seq[i], &hmm );
		
			double gamma[MAX_SEQ_LEN][MAX_STATE]={0};
			count_gamma( gamma, alpha, beta, len, &hmm );
			for(int j = 0; j < len - 1; j++){
				for(int k = 0; k < hmm.state_num; k++){
					gamma_abstract[seq[i][j] - 'A'][k] += gamma[j][k];
				}
			}

			double epsilon[MAX_STATE][MAX_STATE]={0};
			for(int t = 0; t < len - 1; t++){
				count_epsilon( epsilon, t, alpha, beta, seq[i], &hmm);
				for(int x = 0; x < hmm.state_num; x++){
					for(int y = 0; y < hmm.state_num; y++){
						epsilon_sum[x][y] += epsilon[x][y];
					}
				}
			}
		}
		update_hmm( gamma_abstract, epsilon_sum, seq_num, &hmm );
#ifdef DEBUG
		dumpHMM(stdout,&hmm);
#endif
	}

	fp = open_or_die(output_model_path,"w");
	dumpHMM( fp, &hmm );
	fclose(fp);
	
	return 0;
}

static void forward_algo( double alpha[][MAX_STATE], char* seq, HMM* hmm ){
	for(int i = 0; i < hmm->state_num; i++){
		alpha[0][i] = hmm->initial[i] * hmm->observation[seq[0] - 'A'][i];
	}
	int len = strlen(seq);
	for(int i = 1; i < len; i++){
		for(int j = 0; j < hmm->state_num; j++){
			alpha[i][j] = 0;
			for(int k = 0; k < hmm->state_num; k++){
				alpha[i][j] += alpha[i-1][k] * hmm->transition[j][k]; 
			}
			alpha[i][j] *= hmm->observation[seq[i] - 'A'][j];
		}
	}
	return;
}

static void backward_algo( double beta[][MAX_STATE], char* seq, HMM* hmm ){
	for(int i = 0; i < hmm->state_num; i++){
		beta[strlen(seq) - 1][i]=1;
	}
	int len = strlen(seq);
	for(int i = (len - 2); i >= 0; i--){
		for(int j = 0; j < hmm->state_num; j++){
			beta[i][j] = 0;
			for(int k = 0; k < hmm->state_num; k++){
				beta[i][j] += hmm->transition[j][k] * hmm->observation[seq[i+1] - 'A'][k] * beta[i+1][k];
			}
		}	

	}
	return;
}

static void count_gamma( double gamma[][MAX_STATE], double alpha[][MAX_STATE], double beta[][MAX_STATE],int seq_len, HMM* hmm ){
	for(int i = 0; i < seq_len; i++){
		double sum = 0;
		for(int j = 0; j < hmm->state_num; j++){
			gamma[i][j] = alpha[i][j] * beta[i][j];
			sum += gamma[i][j];
		}
		for(int j = 0; j < hmm->state_num;j++){
			gamma[i][j] /= sum;
		}
	}
	return;
}

static void count_epsilon( double epsilon[][MAX_STATE], int t, double alpha[][MAX_STATE], double beta[][MAX_STATE], char* seq, HMM* hmm ){
	double sum = 0;
	for(int i = 0; i < hmm->state_num; i++){
		for(int j = 0; j < hmm->state_num; j++){
			epsilon[i][j] = alpha[t][i] * hmm->transition[i][j] * hmm->observation[seq[t+1] - 'A'][j] * beta[t+1][j];
			sum += epsilon[i][j];
		}
	}
	for(int i = 0; i < hmm->state_num; i++){
		for(int j = 0; j < hmm->state_num; j++){
			epsilon[i][j] /= sum;
		}
	}
	return;
}

static void update_hmm( double gamma_abstract[][MAX_STATE], double epsilon_sum[][MAX_STATE],int seq_num, HMM* hmm ){
	double gamma_sum[MAX_STATE] = {0};

	for(int i = 0; i < hmm->state_num; i++){
		for(int j = 0; j < hmm->observ_num; j++){
			gamma_sum[i] += gamma_abstract[j][i];
		}
	}
	for(int i = 0; i < hmm->state_num; i++){
		hmm->initial[i] = gamma_sum[i] / seq_num;
	}	
	for(int i = 0; i < hmm->state_num; i++){
		for(int j = 0; j < hmm->state_num; j++){
			hmm->transition[i][j] = epsilon_sum[i][j] / gamma_sum[i];
		}
	}
	for(int i = 0; i < hmm->observ_num; i++){
		for(int j = 0; j < hmm->state_num; j++){
			hmm->observation[i][j] = gamma_abstract[i][j] / gamma_sum[j];
		}
	}
	return;
}
