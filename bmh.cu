    //nvcc -arch=sm_11 -m64 -O3 main.cu -o stream.bin


#include<iostream>
#include<cstdlib>
#include <cuda_runtime.h>
#include <cassert>
#include <ctime>
#include <curand.h>
#include <curand_kernel.h>
#include<cmath>
#include <string.h>


#define DOMINIO 128
#define SUBDOMINIO 64 // = DOMINIO / BLOCO
#define BLOCOS 2
#define M 3    //Tamanho do padrao
#define N 8     //Tamanho da linha
#define LINHAS 8 //Linhas por bloco = threads por bloco
#define TAMLINHA 8


#define CHECK_ERROR(call) do {                                                    \
   if( cudaSuccess != call) {                                                             \
      std::cerr << std::endl << "CUDA ERRO: " <<                             \
         cudaGetErrorString(call) <<  " in file: " << __FILE__                \
         << " in line: " << __LINE__ << std::endl;                               \
         exit(0);                                                                                 \
   } } while (0)


//Lendo sequencia e padrao a partir de um arquivo
__host__ void le_sequencia(char *nome_arquivo, char *seq, int tam)
{
  FILE *arq;
  arq = fopen(nome_arquivo, "r");
  fscanf(arq, "%s", seq);
}


//Calcula qual o avanço de acordo com a localizacao do caracter no padrao
__device__ int ord(char *padrao, char c)
{
  int i = M - 1;
  while(padrao[i] != c && i >= 0)
    i--;
  if(i >= 0)
    return i;
  else 
    return M - 1;
}


__global__ void kernel(char *texto, char *padrao, int tamLinha, int iBloco)
{
  int iThread  = blockDim.x * blockIdx.x + threadIdx.x;  //Alterando o indice da Thread de acordo com o bloco
  int d[M];
  int i = 0, k, j;
  int a = 1;
  
  //Pre-processamento
  for (j = 0; j < M; j++) 
    d[j] = M;

  for (j = 0; j < M - 1; j++)
  {
    d[ord(padrao, padrao[j])] = M - a;
    a++;  
  }

  i = (iThread * tamLinha) + M;

  //C e F sao o inicio e o fim de cada linha, pra evitar que uma thread acesse a linha da outra thread
  int c = iThread * tamLinha;
  int f =  (iThread * tamLinha) + tamLinha;

  while ((i <= f) && ( i > c))
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
      printf("Casamento no indice: %d\n",k + (iBloco * SUBDOMINIO), iThread, iBloco);
    }
    a = ord(padrao, texto[i-1]);
    i = i + d[a];
  }
}


using namespace std;
int main (int argc, char **argv)
{
   cudaEvent_t e_Start,
                      e_Stop;
   curandState       *mStates = NULL;
  float elapsedTime = 0.0f;

   //Criando os vetores
  char *d_Texto = NULL, *d_Padrao = NULL;
  char h_Texto[DOMINIO], h_Padrao[M];

  le_sequencia("dna.txt", h_Texto, DOMINIO);
  le_sequencia("padrao_dna.txt", h_Padrao, M);

  unsigned int qtdeDados = DOMINIO * sizeof(char);
   
  //Aloca memória GPU
  CHECK_ERROR(cudaMalloc((void**) &d_Texto, DOMINIO * sizeof(char)));
  CHECK_ERROR(cudaMalloc((void**) &d_Padrao, M * sizeof(char)));

  //Copiando o texto da CPU -> GPU 
  CHECK_ERROR(cudaMemcpy(d_Texto, h_Texto , DOMINIO * sizeof(char),  cudaMemcpyHostToDevice)); 
  CHECK_ERROR(cudaMemcpy(d_Padrao, h_Padrao, M * sizeof(char),  cudaMemcpyHostToDevice)); 

  cudaDeviceProp deviceProp;                   //Levantar a capacidade do device
  cudaGetDeviceProperties(&deviceProp, 0);

  cout << "\nAlgoritmo Boyer Moore Horspool\n";

   //Dados do Problema
   cout << "Tamanho do texto: " << DOMINIO << " caracteres" << endl;
   cout << "Blocos: " << BLOCOS << endl;
   cout << "Threads: " << LINHAS << endl;

     //Reset no device
   CHECK_ERROR(cudaDeviceReset());

     //Criando eventos
   CHECK_ERROR(cudaEventCreate(&e_Start));
   CHECK_ERROR(cudaEventCreate(&e_Stop));
   
   //alocando memória em GPU
   CHECK_ERROR(cudaMalloc(reinterpret_cast<void**> (&d_Texto), qtdeDados));
   CHECK_ERROR(cudaMalloc(reinterpret_cast<void**> (&d_Padrao), M * sizeof(char)));
   CHECK_ERROR(cudaMalloc(reinterpret_cast<void**> (&mStates), DOMINIO * sizeof(curandState)));

   CHECK_ERROR(cudaEventRecord(e_Start, cudaEventDefault));
 
  //Lançando o KERNEL
  for(int k = 0; k < BLOCOS; k++)
     kernel<<<1, LINHAS, 1>>>(d_Texto + (SUBDOMINIO * k), d_Padrao, TAMLINHA, k);
   
   CHECK_ERROR(cudaDeviceSynchronize());
   CHECK_ERROR(cudaEventRecord(e_Stop, cudaEventDefault));
   CHECK_ERROR(cudaEventSynchronize(e_Stop));
   CHECK_ERROR(cudaEventElapsedTime(&elapsedTime, e_Start, e_Stop));
   
   cout << "Tempo de execucao: " << elapsedTime / 1000.0f << " (s) \n";

   CHECK_ERROR( cudaFree(mStates) ); 
   CHECK_ERROR( cudaEventDestroy (e_Start)  );
   CHECK_ERROR( cudaEventDestroy (e_Stop)  );
   cout << "\nFIM\n";
   return EXIT_SUCCESS;
}
