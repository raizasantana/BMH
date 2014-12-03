#include <stdio.h>
#include <string.h>
#include <time.h>


#define M 10
#define N 100000

void le_sequencia(char *nome_arquivo, char *seq, int tam)
{
  FILE *arq;
  arq = fopen(nome_arquivo, "r");
  fscanf(arq, "%s", seq);
}

int ord(char *padrao, char c)
{
	int i = M - 1;

	while(padrao[i] != c && i >= 0)
		i--;

	if(i >= 0)
		return i;
	else 
		return M - 1;
}


void bmh(char *texto, char *padrao, int *resultado)
{

	int d[M];
	int i = 0, k, j;
	int a = 1;
	
	//Pre-processamento
	for (j = 0; j < M; j++)	
		d[j] = M;

	for (j = 0; j < M - 1; j++){
		d[ord(padrao, padrao[j])] = M - a;
		a++;	
	}

	i = M;
	while (i <= N)
	{
		k = i - 1;
		j = M - 1;
			

		while ((j > 0) && (texto[k] == padrao[j]))
		{
			k -= 1;
			j -= 1;
			
		}

		if (j == 0 && (texto[k] == padrao[j]) )
		{
			//printf("Casamento do padrao no indice: %d \n",k);
			resultado[k] = 1;
		}

		 a = ord(padrao, texto[i-1]);
		i = i + d[a];
	}

}


int main()
{

	clock_t tInicio, tFim;
	double tDecorrido;

	time_t inicio, fim;

	char texto[N], padrao[M];

	int resultado[N], i;

	for (i = 0; i < N; i++)
		resultado[i] = 0;

	le_sequencia("dna.txt", texto, M);
  	le_sequencia("padrao_dna.txt", padrao, M);

  	tInicio = clock();
	
	bmh(texto, padrao, resultado);
	
	tFim = clock();

	tDecorrido = ((double)(tFim - tInicio) / CLOCKS_PER_SEC );

	printf("Tempo de Execucao do BMH %f\n",tDecorrido);
	

	
}
