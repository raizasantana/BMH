#include<stdio.h>
#include<stdlib.h>

int main() 
{
	char *nome_arquivo = "dna.txt";
	char dna[128];
	char vet[4] = {'A','C','G','T'};
	char padrao[2] = {'G','T'};
	int k = 0;
	FILE *arq;
	arq = fopen(nome_arquivo, "w");

	//Escrevendo sequencia
	for(k = 0; k < 128; k++)
	{
		dna[k] = vet[rand()%3];
		//printf("[%c] ",dna[k]);

		fprintf(arq,"%c", dna[k]);
	}

	//Escrevendo padrao
	nome_arquivo = "padrao_dna.txt";
	arq = fopen(nome_arquivo, "w");

	for(k = 0; k < 2; k++)
		fprintf(arq, "%c",padrao[k]);

}
