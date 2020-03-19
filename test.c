#include "hmm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef MAX_MODEL_NUM
#	define MAX_MODEL_NUM 10
#endif

#ifndef MAX_SEQ_NUM
#	define MAX_SEQ_NUM 2501
#endif

#ifndef MAX_SEQ_LEN
#	define MAX_SEQ_LEN 100
#endif

#define ERR_EXIT(a) {perror(a); exit(0);}

static double viterbi( const char* seq, HMM *hmm );

int main(int argc,char* argv[]){
	if( argc != 4 ){
		ERR_EXIT("input format error");
	}
	HMM models[MAX_MODEL_NUM];
	char* models_list_path = argv[1]; 
	int total_models = load_models( models_list_path, models, MAX_MODEL_NUM);
	
	char* seq_path = argv[2];
	FILE* seq_fp = open_or_die( seq_path, "r" );

	char* output_result_path = argv[3];
	FILE* output_fp = open_or_die( output_result_path, "w" );
	
	char test_seq[MAX_SEQ_LEN];
	while(fgets(test_seq,MAX_SEQ_LEN,seq_fp) != NULL){
		if(test_seq[ strlen(test_seq) - 1] == '\n'){
			test_seq[ strlen(test_seq) - 1] = '\0';
		}
		double prob[MAX_MODEL_NUM];
		for(int i = 0; i < total_models; i++){
			prob[i] = viterbi( test_seq, &(models[i]) );
		}
		int max_prob = 0;
		for(int i = 1; i < total_models; i++){
			if(prob[max_prob] < prob[i]){
				max_prob = i;
			}
		}
		fprintf( output_fp, "%s %e\n", models[max_prob].model_name, prob[max_prob]);
	}
	fclose(seq_fp);
	fclose(output_fp);
	return 0;
}

static double viterbi( const char* seq, HMM *hmm ){
	double delta[MAX_SEQ_LEN][MAX_STATE];

	//initialization
	for( int i = 0 ; i < hmm->state_num ; i++ ){
		delta[0][i] = hmm->initial[i] * hmm->observation[seq[0] - 'A'][i];
	}

	//iteration
	int len=strlen(seq);
	for(int t = 1; t < len; t++){
		for( int i = 0 ; i < hmm->state_num ; i++ ){
			delta[t][i]= -1;
			for(int j = 0;j < hmm->state_num; j++){
				double buffer = delta[t-1][j] * hmm->transition[j][i];
				delta[t][i] =((delta[t][i] > buffer)? delta[t][i] : buffer);
			} 
			delta[t][i] *= hmm->observation[seq[t] - 'A'][i];
		}
	}

	// find the max of the 2-dim table
	double hmm_max_prob = -1;
	for(int i = 0; i < hmm->state_num; i++ ){
		hmm_max_prob = ((hmm_max_prob > delta[len-1][i])? hmm_max_prob : delta[len-1][i]);
	}
	return hmm_max_prob;
}

