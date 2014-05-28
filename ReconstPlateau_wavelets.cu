#include <stdio.h>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//
//                              ALTIPLANO
//
// Prototypes
//__global__ void Kernel(float*,float*,int,int,int,float*,float*,float*,float*,float*);
__global__ void Kernel(float*,float*,int,int,int,float*,float*,float*,float*,float*);
// Host function
using namespace std;
void SaveOutput(float*,int,int,int,string);
int loadData(float*,char*,int);
void LoadNDVI(float**,int,int,int,int,int,int,char*);
int GetDays_h(int,int);
int isLeap_h(int);
int AcumularDias(int,int*,int,int,int);
void AgregarCeros(float*,float*,int*);
int decomposexEst(float*,int,float*,float*,int,int);

__device__ int decompose(float*,int,float*,float*,int);
__device__ int decompose_without_ruido(float*,int,float*,int);
__device__ void applyReplaceZero(float*,int,float*);
__device__ void applyReconstruction(float*,float*,int,int);
__device__ void applyReplaceZeroxEst(float*,int,float*,int);
__device__ void applyReconstructionxEst(float*,float*,int,int,int);
__device__ void eliminateNegative(int,float*);
__device__ int AcumularLluvia(float*,int,float*,int,int,int);
__device__ int GetDays_d(int,int);
__device__ int isLeap_d(int);

__constant__ float lluvia1[1297];  // Arapa
__constant__ float lluvia2[1297];  // Azangaro
__constant__ float lluvia3[1297];  // Capachica
__constant__ float lluvia4[1297];  // Cojata
__constant__ float lluvia5[1297];  // Huancane
__constant__ float lluvia6[1297];  // Lagunillas
__constant__ float lluvia7[1297];  // Lampa
__constant__ float lluvia8[1297];  // Tambopata

__constant__ int diasAcum[128];

int main(int argc, char** argv)
{
printf ("Proceso inicial: Levantando datos de NDVI\n");
//time_t timer1,timer2;
//time(&timer1);
// acumular dias
int day,month,year;
day=1; month=1; year=1999;
int *AcumDias = (int*)malloc(128*sizeof(int));
AcumularDias(1297,AcumDias,day,month,year);
cudaMemcpyToSymbol(diasAcum,AcumDias,128*sizeof(int));
// cargar datos de lluvia y copio en la memoria constante del device
float* RainDiariaCeros=new float[2048];
float* ruido=new float[1024];
float *lluvia = (float*)malloc(1297*sizeof(float));
char archLluvia[256];
float *ruido1 = (float*)malloc(1024*8*sizeof(float));
float *ruido2 = (float*)malloc(512*8*sizeof(float));
float *ruido3 = (float*)malloc(256*8*sizeof(float));
float *ruido4 = (float*)malloc(128*8*sizeof(float));
// cargo Arapa
strcpy(archLluvia,"D:\\Proyectos\\RecEsp\\RETest\\Arapa.txt");
loadData(lluvia,archLluvia,1297);
cudaMemcpyToSymbol(lluvia1, lluvia, 1297*sizeof(float));
AgregarCeros(RainDiariaCeros,lluvia,AcumDias);
decomposexEst(RainDiariaCeros,2048,RainDiariaCeros,ruido1,0,0);
decomposexEst(RainDiariaCeros,1024,RainDiariaCeros,ruido2,0,0);
decomposexEst(RainDiariaCeros,512,RainDiariaCeros,ruido3,0,0);
decomposexEst(RainDiariaCeros,256,RainDiariaCeros,ruido4,0,0);
// cargo Azangaro
strcpy(archLluvia,"D:\\Proyectos\\RecEsp\\RETest\\Azangaro.txt");
loadData(lluvia,archLluvia,1297);
cudaMemcpyToSymbol(lluvia2, lluvia, 1297*sizeof(float));
AgregarCeros(RainDiariaCeros,lluvia,AcumDias);
decomposexEst(RainDiariaCeros,2048,RainDiariaCeros,ruido1,0,1);
decomposexEst(RainDiariaCeros,1024,RainDiariaCeros,ruido2,0,1);
decomposexEst(RainDiariaCeros,512,RainDiariaCeros,ruido3,0,1);
decomposexEst(RainDiariaCeros,256,RainDiariaCeros,ruido4,0,1);
// cargo Capachica
strcpy(archLluvia,"D:\\Proyectos\\RecEsp\\RETest\\Capachica.txt");
loadData(lluvia,archLluvia,1297);
cudaMemcpyToSymbol(lluvia3, lluvia, 1297*sizeof(float));
AgregarCeros(RainDiariaCeros,lluvia,AcumDias);
decomposexEst(RainDiariaCeros,2048,RainDiariaCeros,ruido1,0,2);
decomposexEst(RainDiariaCeros,1024,RainDiariaCeros,ruido2,0,2);
decomposexEst(RainDiariaCeros,512,RainDiariaCeros,ruido3,0,2);
decomposexEst(RainDiariaCeros,256,RainDiariaCeros,ruido4,0,2);
// cargo Cojata
strcpy(archLluvia,"D:\\Proyectos\\RecEsp\\RETest\\Cojata.txt");
loadData(lluvia,archLluvia,1297);
cudaMemcpyToSymbol(lluvia4, lluvia, 1297*sizeof(float));
AgregarCeros(RainDiariaCeros,lluvia,AcumDias);
decomposexEst(RainDiariaCeros,2048,RainDiariaCeros,ruido1,0,3);
decomposexEst(RainDiariaCeros,1024,RainDiariaCeros,ruido2,0,3);
decomposexEst(RainDiariaCeros,512,RainDiariaCeros,ruido3,0,3);
decomposexEst(RainDiariaCeros,256,RainDiariaCeros,ruido4,0,3);
// cargo Huancane
strcpy(archLluvia,"D:\\Proyectos\\RecEsp\\RETest\\Huancane.txt");
loadData(lluvia,archLluvia,1297);
cudaMemcpyToSymbol(lluvia5, lluvia, 1297*sizeof(float));
AgregarCeros(RainDiariaCeros,lluvia,AcumDias);
decomposexEst(RainDiariaCeros,2048,RainDiariaCeros,ruido1,0,4);
decomposexEst(RainDiariaCeros,1024,RainDiariaCeros,ruido2,0,4);
decomposexEst(RainDiariaCeros,512,RainDiariaCeros,ruido3,0,4);
decomposexEst(RainDiariaCeros,256,RainDiariaCeros,ruido4,0,4);
// cargo Lagunillas
strcpy(archLluvia,"D:\\Proyectos\\RecEsp\\RETest\\Lagunillas.txt");
loadData(lluvia,archLluvia,1297);
cudaMemcpyToSymbol(lluvia6, lluvia, 1297*sizeof(float));
AgregarCeros(RainDiariaCeros,lluvia,AcumDias);
decomposexEst(RainDiariaCeros,2048,RainDiariaCeros,ruido1,0,5);
decomposexEst(RainDiariaCeros,1024,RainDiariaCeros,ruido2,0,5);
decomposexEst(RainDiariaCeros,512,RainDiariaCeros,ruido3,0,5);
decomposexEst(RainDiariaCeros,256,RainDiariaCeros,ruido4,0,5);
// cargo Lampa
strcpy(archLluvia,"D:\\Proyectos\\RecEsp\\RETest\\Lampa.txt");
loadData(lluvia,archLluvia,1297);
cudaMemcpyToSymbol(lluvia7, lluvia, 1297*sizeof(float));
AgregarCeros(RainDiariaCeros,lluvia,AcumDias);
decomposexEst(RainDiariaCeros,2048,RainDiariaCeros,ruido1,0,6);
decomposexEst(RainDiariaCeros,1024,RainDiariaCeros,ruido2,0,6);
decomposexEst(RainDiariaCeros,512,RainDiariaCeros,ruido3,0,6);
decomposexEst(RainDiariaCeros,256,RainDiariaCeros,ruido4,0,6);
// cargo Tambopata
strcpy(archLluvia,"D:\\Proyectos\\RecEsp\\RETest\\Tambopata.txt");
loadData(lluvia,archLluvia,1297);
cudaMemcpyToSymbol(lluvia8, lluvia, 1297*sizeof(float));
AgregarCeros(RainDiariaCeros,lluvia,AcumDias);
decomposexEst(RainDiariaCeros,2048,RainDiariaCeros,ruido1,0,7);
decomposexEst(RainDiariaCeros,1024,RainDiariaCeros,ruido2,0,7);
decomposexEst(RainDiariaCeros,512,RainDiariaCeros,ruido3,0,7);
decomposexEst(RainDiariaCeros,256,RainDiariaCeros,ruido4,0,7);
//
float* ruido1_d;
cudaMalloc((void**)&ruido1_d, 1024*8*sizeof(float));
cudaMemcpy(ruido1_d,ruido1, 1024*8*sizeof(float), cudaMemcpyHostToDevice);

float* ruido2_d;
cudaMalloc((void**)&ruido2_d, 512*8*sizeof(float));
cudaMemcpy(ruido2_d,ruido2, 512*8*sizeof(float), cudaMemcpyHostToDevice);

float* ruido3_d;
cudaMalloc((void**)&ruido3_d, 256*8*sizeof(float));
cudaMemcpy(ruido3_d,ruido3, 256*8*sizeof(float), cudaMemcpyHostToDevice);

float* ruido4_d;
cudaMalloc((void**)&ruido4_d, 128*8*sizeof(float));
cudaMemcpy(ruido4_d,ruido4, 128*8*sizeof(float), cudaMemcpyHostToDevice);

delete[] RainDiariaCeros;
delete[] ruido;
free(lluvia);
free(ruido1);
free(ruido2);
free(ruido3);
free(ruido4);
free(AcumDias);
// cargar datos de ndvi y copio en la memoria del device
char archNdvi[256];
strcpy(archNdvi,"D:\\Proyectos\\RecEsp\\RETest\\ndvi99-06.txt");
int X_MAX=1024;
int Y_MAX=1024;
int Z_MAX=128;
int X_MAX2=225;
int Y_MAX2=225;

int agregar=13;  // se agregan 3 bandas ya que el mayor lag de las estaciones es de 3
int size = X_MAX2 * Y_MAX2 * (Z_MAX+agregar) * sizeof(float);
float *NDVI =(float*)malloc(size); // Asigno memoria en el CPU
int lag=0; // indico cero pq cargaremos ndvi desde la primera banda
LoadNDVI(&NDVI,X_MAX,Y_MAX,Z_MAX+agregar,lag,X_MAX2,Y_MAX2,archNdvi);
float* NDVId;
cudaMalloc((void**)&NDVId, size);
cudaMemcpy(NDVId,NDVI, size, cudaMemcpyHostToDevice);
free(NDVI);
// creamos espacio en device para las salidas
int size2 = 225 * 225 * 1297 * sizeof(float);
int sizeSal=1297*sizeof(float);
float* Pd;
float* salNDVI;
cudaMalloc((void**)&Pd, size2);
cudaMalloc((void**)&salNDVI, sizeSal);
// creamos salidas en el host
float* varios = (float*)malloc(sizeSal);
float* reconst = (float*)malloc(size2);

// ejecuto kernel
dim3 dimBlock(32, 32);
//dim3 dimGrid(X_MAX/32, Y_MAX/32);
dim3 dimGrid(8, 8);
//Kernel<<<dimGrid, dimBlock>>>(NDVId,Pd,X_MAX2,Y_MAX2,Z_MAX,salNDVI,ruido1_d,ruido2_d,ruido3_d,ruido4_d);
Kernel<<<dimGrid, dimBlock>>>(NDVId,Pd,X_MAX2,Y_MAX2,Z_MAX,ruido1_d,ruido2_d,ruido3_d,ruido4_d,salNDVI);
// Transfiero el vector resultante del device al host
cudaMemcpy(reconst,Pd,size2,cudaMemcpyDeviceToHost);
cudaMemcpy(varios,salNDVI,sizeSal,cudaMemcpyDeviceToHost);
// Ejecuto proceso para guardar los resultados en un archivo texto
SaveOutput(reconst,225,225,1297,"salida.txt");
//SaveOutput(varios,1,1,1297,"ndvisalida.txt");

char buffer[5];
bool continuar=true;
while (continuar)
{
printf("");
printf("ingrese coordenada x (1-225): ");
fgets( buffer, 5, stdin );
int coordenadaX = atoi(buffer);
coordenadaX=coordenadaX-1;
printf("ingrese coordenada y (1-225): ");
fgets( buffer, 5, stdin );
int coordenadaY = atoi(buffer);
coordenadaY=coordenadaY-1;
	for (int z=0; z<1297; z++) {
	for (int x=0; x<X_MAX2; x++){ // fila
    for (int y=0; y<Y_MAX2; y++) { //columna
	  if(coordenadaX==y && coordenadaY==x)
	  {
	     varios[z]=reconst[z*Y_MAX2*X_MAX2+x*Y_MAX2 + y];
	  }
	}
    }
	}

SaveOutput(varios,1,1,1297,"pixelsolicitado.txt");

printf("");
printf("Se guardo el pixel solicitado !!!");
printf("");
printf("Presionar 9999 para salir:");
fgets( buffer, 5, stdin );
int salir = atoi(buffer);
if(salir==9999) continuar=false;

}

free(reconst);
free(varios);
cudaFree(NDVId);
cudaFree(ruido1_d);
cudaFree(ruido2_d);
cudaFree(ruido3_d);
cudaFree(ruido4_d);
cudaFree(salNDVI);
cudaFree(Pd);
printf("Finalizo ...\n");
system("PAUSE");
return 0;
}
//--------------------------------------------------------------------------------------------------------------------------------------
//__global__ void Kernel(float* NDVId, float *Pd,int WidthX,int WidthY,int WidthZ,float* salNDVI,float* RUIDO1,float* RUIDO2,float* RUIDO3,float* RUIDO4)
__global__ void Kernel(float* NDVId, float *Pd,int WidthX,int WidthY,int WidthZ,float* RUIDO1,float* RUIDO2,float* RUIDO3,float* RUIDO4,float* salNDVI)
{
  // Calculate the column index of the Pd element, denote by x
  int x = threadIdx.x + blockIdx.x * blockDim.x;  // columna
  // Calculate the row index of the Pd element, denote by y
  int y = threadIdx.y + blockIdx.y * blockDim.y;  // fila


if(y<225 && x<225)
{
  // encuentro la estación mas cercana
  int cX[8],cY[8]; // coordenadas
  cX[0]=100;cX[1]=92;cX[2]=130;cX[3]=185;cX[4]=140;cX[5]=39;cX[6]=71;cX[7]=208;
  cY[0]=128;cY[1]=103;cY[2]=182;cY[3]=115;cY[4]=136;cY[5]=199;cY[6]=153;cY[7]=26;
  int lag[8];
  lag[0]=5;lag[1]=5;lag[2]=4;lag[3]=5;lag[4]=4;lag[5]=8;lag[6]=6;lag[7]=13;

  float distancia;
  float mindis=0.0;
  int estacion=0;
  for(int i=0;i<8;i++)
  {
	float fact1=cX[i]-x;
	float pot1=pow(fact1,2);
	float fact2=cY[i]-y;
	float pot2=pow(fact2,2);
	distancia=sqrt(pot1+pot2);
	if(i==0)
	{
	  mindis=distancia;
	  estacion=0;
	}
	if(distancia<mindis)
	{
	  mindis=distancia;
	  estacion=i;
	}
  }
  // cargo NDVI considerando el lag de la estacion mas cercana
  float NDVI[2048];  // NDVI deberia ser 128 pero se va a reutilizar mas adelante por eso se dimensiona a 2048
  int ind;
  for (int z=0+lag[estacion]; z<WidthZ+lag[estacion]; z++) {
	 ind=(z*WidthY*WidthX)+(y*WidthY)+(x);
	 NDVI[z-lag[estacion]]=NDVId[ind];
  }

  // acumulo los datos de lluvia de la estacion mas cercana
  int day,month,year;
  day=1;month=1;year=1999;
  float RAIN[2048]; // RAIN deberia ser 128 pero se va a reutilizar mas adelante por eso se dimensiona a 2048
  if(estacion==0) AcumularLluvia(lluvia1,1297,RAIN,day,month,year);
  if(estacion==1) AcumularLluvia(lluvia2,1297,RAIN,day,month,year);
  if(estacion==2) AcumularLluvia(lluvia3,1297,RAIN,day,month,year);
  if(estacion==3) AcumularLluvia(lluvia4,1297,RAIN,day,month,year);
  if(estacion==4) AcumularLluvia(lluvia5,1297,RAIN,day,month,year);
  if(estacion==5) AcumularLluvia(lluvia6,1297,RAIN,day,month,year);
  if(estacion==6) AcumularLluvia(lluvia7,1297,RAIN,day,month,year);
  if(estacion==7) AcumularLluvia(lluvia8,1297,RAIN,day,month,year);

/*
  if(x==100 && y==128)
  {
	for(int z=0;z<50;z++)
	{
	  salNDVI[z]=13.88;
	}
  }
*/


//  Ajuste

   float SUMx = 0;     //sum of x values
   float SUMy = 0;     //sum of y values
   float SUMxy = 0;    //sum of x * y
   float SUMxx = 0;    //sum of x^2
   float slope = 0;    //slope of regression line
   float y_intercept = 0; //y intercept of regression line
   float AVGy = 0;     //mean of y
   float AVGx = 0;     //mean of x

   //calculate various sums
   for (int i = 0; i < 128; i++)
   {
	  //sum of x
	  SUMx = SUMx + NDVI[i];
	  //sum of y
	  SUMy = SUMy + RAIN[i];
	  //sum of squared x*y
	  SUMxy = SUMxy + NDVI[i] * RAIN[i];
	  //sum of squared x
	  SUMxx = SUMxx + NDVI[i] * NDVI[i];
   }

   //calculate the means of x and y
   AVGy = SUMy / 128;
   AVGx = SUMx / 128;

   //slope or a1
   float calc=128 * SUMxx - SUMx*SUMx;
   if(calc<=0.0)
   {
	 slope=0.0;
	 y_intercept=0.0;
   }
   else
   {
	 slope = (128 * SUMxy - SUMx * SUMy) / (128 * SUMxx - SUMx*SUMx);
   //y itercept or a0
   y_intercept = AVGy - slope * AVGx;
   }
   for (int z=0; z<WidthZ; z++) {
	 NDVI[z]=slope*NDVI[z]+y_intercept; // se ajusta los datos de NDVI
   }

	float LD1[64];
	float LD2[32];
	float LD3[16];

	decompose(RAIN,128,RAIN,LD1,0); // descomponemos la lluvia acumulada en tendencia y ruido, la tendencia se guarda nuevamente en el vector RAIN
	decompose(RAIN,64,RAIN,LD2,0);
	decompose(RAIN,32,RAIN,LD3,0);

	decompose_without_ruido(NDVI,128,NDVI,0);
	decompose_without_ruido(NDVI,64,NDVI,0);
	decompose_without_ruido(NDVI,32,NDVI,0);

	for(int i=0;i<16;i++)
	{
	  RAIN[i]=NDVI[i];
	}


	applyReplaceZero(LD3,16,RAIN);
	applyReconstruction(LD3,RAIN,16,0);
	eliminateNegative(32,RAIN);

	applyReplaceZero(LD2,32,RAIN);
	applyReconstruction(LD2,RAIN,32,0);
	eliminateNegative(64,RAIN);

	applyReplaceZero(LD1,64,RAIN);
	applyReconstruction(LD1,RAIN,64,0);
	eliminateNegative(128,RAIN);  // aqui el vector RAIN guarda la lluvia acumulada reconstruida (128 datos)


// hasta aqui tenemos la reconstruccion acumulada de la lluvia almacenada en NAm

// a los datos de lluvia diaria se le agrega ceros
  int cont=0;
  int j=0;
  int contrain=0;
  for(int i=0;i<128;i++)
  {
	cont=0;
	for(int x=0;x<16;x++)
	{
	  cont++;
	  if(cont<=diasAcum[i])
	  { // aqui se agrega ceros a la lluvia diaria y se usa el vector NDVI para guardar la lluvia diaria aumentada
		if(estacion==0) NDVI[j]=lluvia1[contrain];
		if(estacion==1) NDVI[j]=lluvia2[contrain];
		if(estacion==2) NDVI[j]=lluvia3[contrain];
		if(estacion==3) NDVI[j]=lluvia4[contrain];
		if(estacion==4) NDVI[j]=lluvia5[contrain];
		if(estacion==5) NDVI[j]=lluvia6[contrain];
		if(estacion==6) NDVI[j]=lluvia7[contrain];
		if(estacion==7) NDVI[j]=lluvia8[contrain];
		contrain++;
	  }
	  else
	  {
		NDVI[j]=0.0;
	  }
	  j++;
	}
  }
// usaremos el vector NDVI para guardar la tendencia al 4 nivel de descomposicion (128 datos) de la lluvia diaria aumentada con ceros
  decompose_without_ruido(NDVI,2048,NDVI,0);
  decompose_without_ruido(NDVI,1024,NDVI,0);
  decompose_without_ruido(NDVI,512,NDVI,0);
  decompose_without_ruido(NDVI,256,NDVI,0);

  float minN,maxN,minT,maxT;
  minN=1000;
  minT=1000;
  maxN=-1000;
  maxT=-1000;
  for(int i=0;i<128;i++)
  {
	if(RAIN[i]<minN)
	{
	  minN=RAIN[i];
	}
	if(NDVI[i]<minT)
	{
	  minT=NDVI[i];
	}
	if(RAIN[i]>maxN)
	{
	  maxN=RAIN[i];
	}
	if(NDVI[i]>maxT)
	{
	  maxT=NDVI[i];
	}
  }
  float factor,b;

  if(maxN-minN==0.0)
  {
	factor=0.0;
  }
  else
  {
	factor=(maxT-minT)/(maxN-minN);
  }
  b=maxT-(factor*maxN);

  for(int i=0;i<128;i++)
  {
	RAIN[i]=(RAIN[i] * factor)+b;
  }


// reconstruccion

applyReplaceZeroxEst(RUIDO4,128,RAIN,estacion);
applyReconstructionxEst(RUIDO4,RAIN,128,0,estacion);

applyReplaceZeroxEst(RUIDO3,256,RAIN,estacion);
applyReconstructionxEst(RUIDO3,RAIN,256,0,estacion);

applyReplaceZeroxEst(RUIDO2,512,RAIN,estacion);
applyReconstructionxEst(RUIDO2,RAIN,512,0,estacion);

applyReplaceZeroxEst(RUIDO1,1024,RAIN,estacion);
applyReconstructionxEst(RUIDO1,RAIN,1024,0,estacion);

eliminateNegative(2048,RAIN);

// quitar ceros

j=0;
cont=0;
int contar=0;
for(int i=0;i<2048;i++)
{
  contar++;
  if(contar<=diasAcum[j])
  {
	RAIN[cont]=RAIN[i];
	cont++;
  }
  if(contar==16)
  {
	contar=0;
	j++;
  }
}

	for (int z=0; z<1297; z++) {
	  Pd[z*225*225+y*225 + x]= RAIN[z];
	}



}

}
// ------------------------------------------------------------------------------------------------------------------------------------------------
int loadData(float* vector,char* fileName,int num)
{
  int i,count;
  float data;
  FILE *fp;
  fp = fopen( fileName, "r");
  rewind(fp);
  count=0;
  for (i = 0;(fscanf(fp, "%f",&(data)) == 1); i++)
  {
	vector[i]=data;
	count++;
	if(count==num) break;
  }
  fclose(fp);
  return count;
}
//---------------------------------------------------------------------------
__device__ void eliminateNegative(int countNAm,float* NAm)
{
  for(int i=0;i<countNAm;i++)
  {
    if(NAm[i]<0.0)
    {
	  NAm[i]=0.0;
    }
  }
}
//--------------------------------------------------------------------------------------------------------------------------------------
__device__ void applyReplaceZero(float* pvector,int numreg, float* NAm)
{
//  double* vector=NULL;
//  vector=*pvector;
  for(int i=0;i<numreg;i++)
  {
	if(pvector[i]==0.0)
	{
	  NAm[i]=0.0;
	}
  }
//  vector=NULL;
}
//--------------------------------------------------------------------------------------------------------------------------------------
__device__ void applyReplaceZeroxEst(float* pvector,int numreg, float* NAm,int estacion)
{
  int ind=0;
  for(int i=0;i<numreg;i++)
  {
	ind=(estacion*numreg)+i;
	if(pvector[ind]==0.0)
	{
	  NAm[i]=0.0;
	}
  }
}
//--------------------------------------------------------------------------------------------------------------------------------------
void SaveOutput(float *salida,int X_MAX, int Y_MAX, int Z_MAXDiario,string out) {
//  creo matriz de salida
  float*** Matrix3Dfinal = new float**[X_MAX];
  for(int x = 0; x < X_MAX;x++)
  {
    Matrix3Dfinal[x] = new float*[Y_MAX];
    for(int y = 0; y < Y_MAX; y++)
    {
      Matrix3Dfinal[x][y] = new float[Z_MAXDiario];
    }
  }
// paso de vector salida host a la matriz de salida
	for (int z=0; z<Z_MAXDiario; z++) {
	for (int x=0; x<X_MAX; x++){
    for (int y=0; y<Y_MAX; y++) {
	   Matrix3Dfinal[x][y][z]=salida[z*Y_MAX*X_MAX+x*Y_MAX + y];
	}
    }
	}
// guardo matriz salida a un archivo
FILE *stream2=NULL;
string archivo=out;
stream2 = fopen (archivo.c_str(),"w");
rewind(stream2);
string cadena="";
string cadena2;
char* dato;
char* dato2;
for(int z = 0; z < Z_MAXDiario;z++)
{
  for(int x = 0; x < X_MAX;x++)
  {
    cadena="";
    for(int y = 0; y < Y_MAX; y++)
    {
      float valor=Matrix3Dfinal[x][y][z];
      cadena2=std::to_string(long double(valor));
      if(y==Y_MAX-1)
      {
        cadena= cadena + cadena2;
      }
      else
      {
        cadena= cadena + cadena2 + " ";
      }
    }
    dato=new char[cadena.size()+1];
    strcpy(dato,cadena.c_str());
    fprintf(stream2,"%s\n",dato);
	delete(dato);
  }
  cadena="";
  dato2=new char[cadena.size()+1];
  strcpy(dato2,cadena.c_str());
  fprintf(stream2,"%s\n",dato2);
  delete(dato2);
}
fclose(stream2);
// elimino matriz de salida
  for(int x = 0; x < X_MAX; x++)
  {
    for(int y = 0; y < Y_MAX; y++)
    {
      delete[] Matrix3Dfinal[x][y];
    }
    delete[] Matrix3Dfinal[x];
  }
  delete[] Matrix3Dfinal;
  Matrix3Dfinal=0;
}
//---------------------------------------------------------------------------
void LoadNDVI(float** Mp,int X_MAX,int Y_MAX,int Z_MAXDiario,int lag,int X_MAX2,int Y_MAX2,char* arch) {
  float* M=NULL;
  M=*Mp;

  float*** Matrix3Dfinal = new float**[X_MAX2];
  for(int x = 0; x < X_MAX2;x++)
  {
	Matrix3Dfinal[x] = new float*[Y_MAX2];
	for(int y = 0; y < Y_MAX2; y++)
	{
	  Matrix3Dfinal[x][y] = new float[Z_MAXDiario];
	}
  }

  for(int z = 0; z < Z_MAXDiario; z++)
  {
	for(int x = 0; x < X_MAX2;x++)
	{
	  for(int y = 0; y < Y_MAX2; y++)
	  {
		  Matrix3Dfinal[x][y][z]=0.0;
	  }
	}
  }
  FILE* pDato;
  pDato = fopen (arch,"r");
  rewind (pDato);

  int contlag=0;
  float valor=0.0;
  for(int z = 0; z < Z_MAXDiario+lag; z++)
  {
	contlag++;
	for(int x = 0; x < X_MAX2;x++)
	{
      for(int y = 0; y < Y_MAX2; y++)
      {
        fscanf(pDato,"%f", &valor);

	    if(contlag>lag)
		{
		  Matrix3Dfinal[x][y][z-lag]=valor;
		}
      }
    }
  }
  fclose (pDato);


	for (int z=0; z<Z_MAXDiario; z++) {
	for (int x=0; x<X_MAX2; x++){
	for (int y=0; y<Y_MAX2; y++) {
	   M[z*Y_MAX2*X_MAX2+x*Y_MAX2 + y]=Matrix3Dfinal[x][y][z];
	}
	}
	}
 M=0;
}
//---------------------------------------------------------------------------
__device__ int decompose(float* vector,int numreg,float* tendencia,float* ruido,int kindwave)
{
  int i;
  float h0,h1,h2,h3,g0,g1,g2,g3;
  switch (kindwave) {
  case 0: // haar
	h0= 0.7071067814;
	h1= 0.7071067814;
	h2= 0.0;
	h3= 0.0;
	g0= 0.7071067814;
	g1= -0.7071067814;
	g2= 0.0;
	g3= 0.0;
	break;
  case 1: // symmlet2
	h0= 0.482962913;
	h1= 0.836516303;
	h2= 0.224143868;
	h3= -0.129409522;
	g0= -0.129409522;
	g1= -0.224143868;
	g2= 0.836516303;
	g3= -0.482962913;
	break;
  default:
	break;
  }

  int cont=0;
  float aa3T;
  float aa3R;

  aa3T=0.0;
  aa3T+=vector[0]*h2;
  aa3T+=vector[1]*h3;
  aa3T+=vector[numreg-2]*h0;
  aa3T+=vector[numreg-1]*h1;

  aa3R=0.0;
  aa3R+=vector[0]*g2;
  aa3R+=vector[1]*g3;
  aa3R+=vector[numreg-2]*g0;
  aa3R+=vector[numreg-1]*g1;

  float aa3T_back=aa3T;
  float aa3R_back=aa3R;

  for(i=0;i<=numreg-3;i=i+2)
  {
	aa3T=0.0;
	aa3R=0.0;

	aa3T+=vector[i]*h0;
	aa3T+=vector[i+1]*h1;
	aa3T+=vector[i+2]*h2;
	aa3T+=vector[i+3]*h3;

	aa3R+=vector[i]*g0;
	aa3R+=vector[i+1]*g1;
	aa3R+=vector[i+2]*g2;
	aa3R+=vector[i+3]*g3;

	tendencia[cont]=aa3T;
	ruido[cont]=aa3R;
	cont++;
  }

  tendencia[(numreg/2)-1]=aa3T_back;
  ruido[(numreg/2)-1]=aa3R_back;

  return numreg/2;
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
__device__ int decompose_without_ruido(float* vector,int numreg,float* tendencia,int kindwave)
{
  int i;
  float h0,h1,h2,h3;
//  float g0,g1,g2,g3;
  switch (kindwave) {
  case 0: // haar
	h0= 0.7071067814;
	h1= 0.7071067814;
	h2= 0.0;
	h3= 0.0;
//    g0= 0.7071067814;
//    g1= -0.7071067814;
//    g2= 0.0;
//    g3= 0.0;
    break;
  case 1: // symmlet2
    h0= 0.482962913;
    h1= 0.836516303;
    h2= 0.224143868;
    h3= -0.129409522;
//    g0= -0.129409522;
//    g1= -0.224143868;
//    g2= 0.836516303;
//    g3= -0.482962913;
    break;
  default:
    break;
  }

  int cont=0;
  float aa3T;

  aa3T=0.0;
  aa3T+=vector[0]*h2;
  aa3T+=vector[1]*h3;
  aa3T+=vector[numreg-2]*h0;
  aa3T+=vector[numreg-1]*h1;


  float aa3T_back=aa3T;

  for(i=0;i<=numreg-3;i=i+2)
  {
    aa3T=0.0;

	aa3T+=vector[i]*h0;
    aa3T+=vector[i+1]*h1;
    aa3T+=vector[i+2]*h2;
    aa3T+=vector[i+3]*h3;

    tendencia[cont]=aa3T;
    cont++;
  }

  tendencia[(numreg/2)-1]=aa3T_back;

  return numreg/2;
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
__device__  void applyReconstruction(float* pvector,float* NAmm,int numreg,int kindwave)
{
  float h0,h1,h2,h3,g0,g1,g2,g3;
  if(kindwave==0)
  {
    h0= 0.7071067814;
    h1= 0.7071067814;
    h2= 0.0;
    h3= 0.0;
    g0= 0.7071067814;
    g1= -0.7071067814;
    g2= 0.0;
    g3= 0.0;
  }
  else
  {
    h0= 0.482962913;
    h1= 0.836516303;
    h2= 0.224143868;
    h3= -0.129409522;
    g0= -0.129409522;
    g1= -0.224143868;
    g2= 0.836516303;
    g3= -0.482962913;
  }



  int res=0;
  float aa3=0.0;
  int i,j;
  float temp[2048];
  for(i=0;i<numreg;i++)
  {
    temp[i]=NAmm[i];
  }

  aa3=0.0;
  aa3+=temp[0]*h0;
  aa3+=pvector[0]*g0;
  aa3+=temp[numreg-1]*h2;
  aa3+=pvector[numreg-1]*g2;
  NAmm[0]=aa3;

  aa3=0.0;
  aa3+=temp[0]*h1;
  aa3+=pvector[0]*g1;
  aa3+=temp[numreg-1]*h3;
  aa3+=pvector[numreg-1]*g3;
  NAmm[1]=aa3;

  for(i=2;i<2*numreg;i++)
  {
    res=i%2;
	aa3=0.0;
	if(res==0)  // es par
	{
	     j=i-2;
         aa3+=temp[j/2]*h2;
         aa3+=pvector[j/2]*g2;
         aa3+=temp[(j/2)+1]*h0;
         aa3+=pvector[(j/2)+1]*g0;
	}
	else  // es impar
	{
	     j=i-2-1;
         aa3+=temp[j/2]*h3;
         aa3+=pvector[j/2]*g3;
         aa3+=temp[(j/2)+1]*h1;
         aa3+=pvector[(j/2)+1]*g1;
	}
	NAmm[i]=aa3;
  }
}
//--------------------------------------------------------------------------------------------------------------------------------------
__device__  void applyReconstructionxEst(float* pvector,float* NAmm,int numreg,int kindwave,int estacion)
{
  float h0,h1,h2,h3,g0,g1,g2,g3;
  if(kindwave==0)
  {
    h0= 0.7071067814;
    h1= 0.7071067814;
    h2= 0.0;
    h3= 0.0;
    g0= 0.7071067814;
    g1= -0.7071067814;
    g2= 0.0;
    g3= 0.0;
  }
  else
  {
    h0= 0.482962913;
    h1= 0.836516303;
    h2= 0.224143868;
    h3= -0.129409522;
    g0= -0.129409522;
    g1= -0.224143868;
    g2= 0.836516303;
    g3= -0.482962913;
  }

  int res=0;
  float aa3=0.0;
  int i,j;
  float temp[2048];
  for(i=0;i<numreg;i++)
  {
	temp[i]=NAmm[i];
  }

  aa3=0.0;
  aa3+=temp[0]*h0;
  aa3+=pvector[estacion*numreg+0]*g0;
  aa3+=temp[numreg-1]*h2;
  aa3+=pvector[estacion*numreg+(numreg-1)]*g2;
  NAmm[0]=aa3;

  aa3=0.0;
  aa3+=temp[0]*h1;
  aa3+=pvector[estacion*numreg+0]*g1;
  aa3+=temp[numreg-1]*h3;
  aa3+=pvector[estacion*numreg+(numreg-1)]*g3;
  NAmm[1]=aa3;

  for(i=2;i<2*numreg;i++)
  {
	res=i%2;
	aa3=0.0;
	if(res==0)  // es par
	{
		 j=i-2;
		 aa3+=temp[j/2]*h2;
		 aa3+=pvector[estacion*numreg+(j/2)]*g2;
		 aa3+=temp[(j/2)+1]*h0;
		 aa3+=pvector[estacion*numreg+((j/2)+1)]*g0;
	}
	else  // es impar
	{
		 j=i-2-1;
		 aa3+=temp[j/2]*h3;
		 aa3+=pvector[estacion*numreg+(j/2)]*g3;
		 aa3+=temp[(j/2)+1]*h1;
		 aa3+=pvector[estacion*numreg+((j/2)+1)]*g1;
	}
	NAmm[i]=aa3;
  }
}
//--------------------------------------------------------------------------------------------------------------------------------------
__device__ int AcumularLluvia(float* lluvia,int numdatos,float* rainAcum,int day,int month,int year)
{
  int decadal;
  if(day<=10) decadal=1;
  if(day>10 && day<=20) decadal=2;
  if(day>20) decadal=3;
  // acumulo la lluvia en decadales
  float acum=0.0;
  int contDias=0;
  int dia=day;
  int contAcum=0;
  int TotalDaysThisDecadal;
  for(int i=0;i<numdatos;i++)
  {
	acum=acum+lluvia[i];
	contDias++;
	if(decadal==1 && dia==10){rainAcum[contAcum]=acum;decadal++;dia=0;acum=0;contAcum++;contDias=0;}
	if(decadal==2 && dia==10){rainAcum[contAcum]=acum;decadal++;dia=0;acum=0;contAcum++;contDias=0;}
	if(decadal==3){TotalDaysThisDecadal=GetDays_d(month,year);}
	if(decadal==3 && dia==TotalDaysThisDecadal)
	{
	  rainAcum[contAcum]=acum;
	  decadal=1;
	  dia=0;
	  acum=0.0;
	  contAcum++;
	  if(month==12)
	  {month=1;year++;}
	  else
	  {month++;}
	  contDias=0;
	}
	dia++;
  }
  return contAcum;
}
//---------------------------------------------------------------------------
int AcumularDias(int numdatos,int* diasAcum,int day,int month,int year)
{
  int decadal;
  if(day<=10) decadal=1;
  if(day>10 && day<=20) decadal=2;
  if(day>20) decadal=3;
  // acumulo la lluvia en decadales
  int contDias=0;
  int dia=day;
  int contAcum=0;
  int TotalDaysThisDecadal;
  for(int i=0;i<numdatos;i++)
  {
	contDias++;
	if(decadal==1 && dia==10){diasAcum[contAcum]=contDias;decadal++;dia=0;contAcum++;contDias=0;}
	if(decadal==2 && dia==10){diasAcum[contAcum]=contDias;decadal++;dia=0;contAcum++;contDias=0;}
	if(decadal==3){TotalDaysThisDecadal=GetDays_h(month,year);}
	if(decadal==3 && dia==TotalDaysThisDecadal)
	{
	  diasAcum[contAcum]=contDias;
	  decadal=1;
	  dia=0;
	  contAcum++;
	  if(month==12)
	  {month=1;year++;}
	  else
	  {month++;}
	  contDias=0;
	}
	dia++;
  }
  return contAcum;
}
//---------------------------------------------------------------------------
int GetDays_h(int month,int year)
{
int dato;
switch (month) {
  case 1 :
	return 11;
  case 2 :
	if(isLeap_h(year)){dato=9;}
	else{dato=8;}
	return dato;
  case 3 :
	return 11;
  case 4 :
	return 10;
  case 5 :
	return 11;
  case 6 :
	return 10;
  case 7 :
	return 11;
  case 8 :
	return 11;
  case 9 :
	return 10;
  case 10 :
	return 11;
  case 11 :
	return 10;
  case 12 :
	return 11;
}
return 0;
}
//---------------------------------------------------------------------------
__device__ int GetDays_d(int month,int year)
{
int dato;
switch (month) {
  case 1 :
	return 11;
  case 2 :
	if(isLeap_d(year)){dato=9;}
	else{dato=8;}
	return dato;
  case 3 :
	return 11;
  case 4 :
	return 10;
  case 5 :
	return 11;
  case 6 :
	return 10;
  case 7 :
	return 11;
  case 8 :
	return 11;
  case 9 :
	return 10;
  case 10 :
	return 11;
  case 11 :
	return 10;
  case 12 :
	return 11;
}
return 0;
}
//---------------------------------------------------------------------------
int isLeap_h(int year)
{
	return ((year % 4 == 0 && year % 100 != 0) || year % 400 == 0);
}
//---------------------------------------------------------------------------
__device__ int isLeap_d(int year)
{
	return ((year % 4 == 0 && year % 100 != 0) || year % 400 == 0);
}
//---------------------------------------------------------------------------
void AgregarCeros(float* RainDiariaCeros,float* lluvia,int* diasAcum)
{
  int cont=0;
  int j=0;
  int contrain=0;
  for(int i=0;i<128;i++)
  {
	cont=0;
	for(int x=0;x<16;x++)
	{
	  cont++;
	  if(cont<=diasAcum[i])
	  { // aqui se agrega ceros a la lluvia diaria y se usa el vector NDVI para guardar la lluvia diaria aumentada
		RainDiariaCeros[j]=lluvia[contrain];
		contrain++;
	  }
	  else
	  {
		RainDiariaCeros[j]=0.0;
	  }
	  j++;
	}
  }
}
//---------------------------------------------------------------------------
int decomposexEst(float* vector,int numreg,float* tendencia,float* ruido,int kindwave,int est)
{
  int i;
  float h0,h1,h2,h3,g0,g1,g2,g3;
  switch (kindwave) {
  case 0: // haar
	h0= 0.7071067814;
	h1= 0.7071067814;
	h2= 0.0;
	h3= 0.0;
	g0= 0.7071067814;
	g1= -0.7071067814;
	g2= 0.0;
	g3= 0.0;
	break;
  case 1: // symmlet2
	h0= 0.482962913;
	h1= 0.836516303;
	h2= 0.224143868;
	h3= -0.129409522;
	g0= -0.129409522;
	g1= -0.224143868;
	g2= 0.836516303;
	g3= -0.482962913;
	break;
  default:
	break;
  }

  int cont=0;
  float aa3T;
  float aa3R;

  aa3T=0.0;
  aa3T+=vector[0]*h2;
  aa3T+=vector[1]*h3;
  aa3T+=vector[numreg-2]*h0;
  aa3T+=vector[numreg-1]*h1;

  aa3R=0.0;
  aa3R+=vector[0]*g2;
  aa3R+=vector[1]*g3;
  aa3R+=vector[numreg-2]*g0;
  aa3R+=vector[numreg-1]*g1;

  float aa3T_back=aa3T;
  float aa3R_back=aa3R;

  for(i=0;i<=numreg-3;i=i+2)
  {
	aa3T=0.0;
	aa3R=0.0;

	aa3T+=vector[i]*h0;
	aa3T+=vector[i+1]*h1;
	aa3T+=vector[i+2]*h2;
	aa3T+=vector[i+3]*h3;

	aa3R+=vector[i]*g0;
	aa3R+=vector[i+1]*g1;
	aa3R+=vector[i+2]*g2;
	aa3R+=vector[i+3]*g3;

	tendencia[cont]=aa3T;
	ruido[(est*(numreg/2))+cont]=aa3R;
	cont++;
  }

  tendencia[(numreg/2)-1]=aa3T_back;
  ruido[(est*(numreg/2))+((numreg/2)-1)]=aa3R_back;

  return numreg/2;
}
//---------------------------------------------------------------------------