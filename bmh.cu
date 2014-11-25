#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cassert>
#include <ctime>
#include <curand.h>
#include <curand_kernel.h>
#include <cmath>
#include <string.h>


#define DOMINIO 10000
#define SUBDOMINIO 5000 // = DOMINIO / BLOCO
#define BLOCOS 2
#define M 4    //Tamanho do padrao
#define N 100     //Tamanho da linha
#define LINHAS 100 //Linhas por bloco = threads por bloco
#define TAMLINHA 100


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


__global__ void kernel(char *texto, char *padrao, int tamLinha, int bloco, int *res)
{

  int thread  = blockDim.x * blockIdx.x + threadIdx.x; 

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

  i = (thread * tamLinha) + M;

  //C e F sao o inicio e o fim de cada linha, pra evitar que uma thread acesse a linha da outra thread
  int c = thread * tamLinha;
  int f =  (thread * tamLinha) + tamLinha;

  while ((i <= f) && ( i > c))
  {
    k = i - 1;
    j = M - 1;
  
    while ((j > 0) && (texto[k] == padrao[j]))
    {
      k -= 1;
      j -= 1;
    }
  
    if (j == 0 && (texto[k] == padrao[j]))
      res[k] = 1;
    
    a = ord(padrao, texto[i-1]);
    i = i + d[a];

  }
}


using namespace std;

int main (int argc, char **argv)
{

  cudaEvent_t e_Start, e_Stop;

  float elapsedTime = 0.0f;

  //Criando os vetores - Device
  char *d_Texto = NULL, *d_Padrao = NULL;
  int *d_resultado = NULL;

  //Vetores - Host
  char h_Texto[DOMINIO], h_Padrao[M];
  int h_resultado[DOMINIO];

  le_sequencia("dna.txt", h_Texto, DOMINIO);
  le_sequencia("padrao_dna.txt", h_Padrao, M);

  memset(h_resultado, 0,  DOMINIO * sizeof(int));

  unsigned int qtdeDados = DOMINIO * sizeof(char);
   
  //Aloca memória GPU
  CHECK_ERROR(cudaMalloc((void**) &d_Texto, DOMINIO * sizeof(char)));
  CHECK_ERROR(cudaMalloc((void**) &d_Padrao, M * sizeof(char)));
  CHECK_ERROR(cudaMalloc((void**) &d_resultado, DOMINIO * sizeof(int)));

  //Copiando o texto da CPU -> GPU 
  CHECK_ERROR(cudaMemcpy(d_Texto, h_Texto , DOMINIO * sizeof(char),  cudaMemcpyHostToDevice)); 
  CHECK_ERROR(cudaMemcpy(d_Padrao, h_Padrao, M * sizeof(char),  cudaMemcpyHostToDevice)); 
  CHECK_ERROR(cudaMemcpy(d_resultado, h_resultado, DOMINIO * sizeof(int),  cudaMemcpyHostToDevice)); 

  cudaDeviceProp deviceProp;        
  cudaGetDeviceProperties(&deviceProp, 0);

  cout << "\n\n  Algoritmo Boyer Moore Horspool\n\n\n";

   //Dados do Problema
  cout << "::Dados do Problema::\n" << endl;
  cout << "Tamanho do texto: " << DOMINIO << " caracteres" << endl;
  cout << "Blocos: " << BLOCOS << endl;
  cout << "Threads: " << LINHAS << endl;
  cout << "Padrao: " << h_Padrao << endl;

  //Reset no device
   CHECK_ERROR(cudaDeviceReset());

  //Criando eventos
   CHECK_ERROR(cudaEventCreate(&e_Start));
   CHECK_ERROR(cudaEventCreate(&e_Stop));


 //Alocando memória em GPU
 CHECK_ERROR(cudaMalloc(reinterpret_cast<void**> (&d_Texto), qtdeDados));
 CHECK_ERROR(cudaMalloc(reinterpret_cast<void**> (&d_Padrao), M * sizeof(char)));
 CHECK_ERROR(cudaMalloc(reinterpret_cast<void**> (&d_resultado), DOMINIO * sizeof(int))); 

 CHECK_ERROR(cudaEventRecord(e_Start, cudaEventDefault));
 
  //Lançando o KERNEL
  for(int k = 0; k < BLOCOS; k++)
     kernel<<<1, LINHAS, 1>>>(d_Texto + (SUBDOMINIO * k), d_Padrao, TAMLINHA, k, d_resultado + (SUBDOMINIO * k));
   
   CHECK_ERROR(cudaDeviceSynchronize());

   //GPU -> CPU
   CHECK_ERROR(cudaMemcpy(h_resultado, d_resultado, DOMINIO * sizeof(int), cudaMemcpyDeviceToHost));

   CHECK_ERROR(cudaEventRecord(e_Stop, cudaEventDefault));
   CHECK_ERROR(cudaEventSynchronize(e_Stop));
   CHECK_ERROR(cudaEventElapsedTime(&elapsedTime, e_Start, e_Stop));
   
   cout << "Tempo de execucao: " << elapsedTime / 1000.0f << " (s) \n\n\n";

   //Resultado
   for(int k = 0; k < DOMINIO; k++)
      if(h_resultado[k] == 1)
        cout << "Ocorrencia em: " << k << endl;
   

   
   CHECK_ERROR(cudaEventDestroy(e_Start));
   CHECK_ERROR(cudaEventDestroy(e_Stop));

   cout << "\nFIM\n";

   return EXIT_SUCCESS;
}
