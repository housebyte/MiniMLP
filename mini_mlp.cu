#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include "1dkern.cu"
#include <ctime>
#include <time.h>
#define USE_SIG 1
#define INPUT_SIZE 2
int USE_GPU=1;


/*A little mini mlp - minimal CUDA mlp for use with Fuzzy Cellular Automata*/

using namespace std;


/*Random Number Generator - Option Replace with CookBook*/

float getlrand(float lower,float upper){

	        return ((float) rand()/ RAND_MAX) * (upper-lower) + lower;

}

/*Begin Kernel functions*/

__device__ float mytanh_(float x)
{
  float exp2x = expf(2*x);
  return (exp2x - 1) / (exp2x + 1);
}

__device__ float sigmoid(float v)
{
	return 1 / (1 + expf(-v));
}


__device__ float InvSig(const float sum,const float act)
{
	return (act*(1-act)*sum) ;
}

/* Call all Act functions - includes rng for noisy Acts */

__device__ float ActFun(int type, float v,int dir,float u, int seed){

if(type==0){
/*Tanh*/
if(dir==0){
return mytanh_(v);
}else{
return (1-pow((double)mytanh_(u),2.0))*v;
//return (1-(mytanh_(u)*mytanh_(u)))*v;
//return (1-mytanh_(u))*v;
			}

			}
			
if(type==1){
/*Sigmoid*/
if(dir==0){
return sigmoid(v);
}else{
return InvSig(v,u);
			}

			}
if(type==2){
/*Norm*/
if(dir==0){
return v;
}else{
return v;
			}

			}			
return 0;			

}


						/***************************FEEDFORWARD FUNCTIONS******************************/

__global__ void gpu_act(float* const __restrict__	Act,const int* const __restrict__ DIM,const int A_f){	

const int NOUT = DIM[1];
const int SOUT = DIM[3];

			
GPU_1D_KERN_LOOP(index, NOUT){

const int ls = (index / 1 )                 % NOUT;		//layer size

//Act[SOUT+ls]=sigmoid(Act[SOUT+ls]);

Act[SOUT+ls]=ActFun(A_f,Act[SOUT+ls],0,0,1);

									}

}

				/**********WE NEED TO ADD TANH AND USE IT INSTEAD OF SIGMOID FOR XOR*********/
				/**********SIGMOID FOR RL ??**************/
				/**********INSTALL POINTER TO FUNCTION FOR THIS??*************/
				/* Prob don't need pointer to function as we are not using class for neurons - globals are already globals!! */
				/* Choose which activation function 1-n && choose parameters for noise ACTFUN[NRN][0] = 1-n ACTFUN[NRN][1] = (int)seed */
				/* Implement noisy transfuncs in CUDA??? Unique activation functions for all Neurons (With Learnable Parameters??)*/
				
				


__global__ void gpu_ff_allconn(const float* const __restrict__ Wgts,const float* const __restrict__ Wgts_,float* const __restrict__ Values,float* const __restrict__ Labels,float* const __restrict__ Act,float* const __restrict__ Bias,int* const __restrict__ P_l,int* const __restrict__ W_l,const int LAY,const int NL,const int* const __restrict__ DIM,const int ITER,const int A_f){

const int NIN  = DIM[0];
const int NOUT = DIM[1];
const int SIN  = DIM[2];
const int SOUT = DIM[3];
const int NVAL = DIM[4];

if(LAY==1){

GPU_1D_KERN_LOOP(index,NIN ){

	const int ls = (index / 1 )         % NIN;		//Outputs / layer size
	
														//Act[SIN+ls]=Values[ITER*NVAL+ls];
	Act[SIN+ls]=ActFun(0,Values[ITER*NVAL+ls],0,0,1);

}

}

		
GPU_1D_KERN_LOOP(index,NOUT ){

	const int ls = (index / 1 )         % NOUT;		//Outputs / layer size
	
	Act[SOUT+ls]=0;


}


/*Feedforward to MLP layer*/
GPU_1D_KERN_LOOP(index, NIN*NOUT){

	
	
		const int ls = (index / 1 )         %NOUT;		//Outputs / layer size
		const int col = (index / NOUT ) 	 %NIN;		//Inputs /  size of feature map
		
									
atomicAdd(&Act[SOUT+ls],Wgts[W_l[LAY-1]+(col+ls*NIN)]*Act[SIN+col]);
atomicAdd(&Act[SOUT+ls],Bias[SOUT+ls]);
}

if(1){

GPU_1D_KERN_LOOP(index, NOUT){

		const int ls = (index / 1 )         %NOUT;		//Outputs / layer size
														//Labels[nl]=sigmoid(Labels[nl]);

Act[SOUT+ls] = ActFun(A_f,Act[SOUT+ls],0,0,1);
									}

}

if(NL>0){												//Begin Label calc
	
			
	GPU_1D_KERN_LOOP(index, NL){

		const int nl = (index / 1 )                 % NL;		//number of labels
		Labels[nl]=0;
}

									
/*Feedforward MLP layer to labelled output*/									
	GPU_1D_KERN_LOOP(index, NL*NIN){

	
		const int nl = (index / 1 )                 % NL;		//number of labels
		const int ls = (index / NL)          	    % NIN;		//layer size

atomicAdd(&Labels[nl],Wgts_[nl*NOUT+ls]*Act[SIN+ls]);
atomicAdd(&Labels[nl],Bias[SIN+nl]);	

}

	
			/*Sigmoid*/	/***************NO SIGMOID FOR OUTPUT*****************/							

if(1){

GPU_1D_KERN_LOOP(index, NL){
const int nl = (index / 1 )                 % NL;		//layer size
		
														//Labels[nl]=sigmoid(Labels[nl]);

Labels[nl] = ActFun(A_f,Labels[nl],0,0,1);
									}
	}
											}/*End Label Loop*/
														
}

					/**********************************FEEDBACK ERROR***********************************/

__global__ void gpu_label_err(const float* const __restrict__ Labels,const float* const __restrict__ Val,float* const __restrict__ ERR,const int ITER,const int* const __restrict__ DIM)
{

const int NOUT = DIM[1];
const int SOUT = DIM[3];
//const int VS   = DIM[4];

GPU_1D_KERN_LOOP(index,NOUT ){			/*THIS CHECKSOUT*/

	const int nl = (index / 1 )         % NOUT;		//Outputs / layer size
	
	ERR[SOUT+nl]=Val[ITER*3+2];	//*Labels[nl]*(1-Labels[nl]);			
	atomicAdd(&ERR[SOUT+nl],-Labels[0]);												//Do a proper Error here

//Need to put Val in properly here

								}

				}

__global__ void gpu_inv_label_err(float* const __restrict__ ERR,float* const __restrict__ ACT,float* const __restrict__ Bias,float* const __restrict__ Labels,const int ITER,const int* const __restrict__ DIM,const float* const __restrict__ C_e,const int A_f)
{
	const int NOUT = DIM[1];			/*THIS CHECKSOUT*/
	const int SOUT = DIM[3];
	const float Lrt  = C_e[0];
	
GPU_1D_KERN_LOOP(index, NOUT){

	const int us = (index / 1 )                % NOUT;		//layer size
 
	ERR[SOUT+us] = ActFun(A_f,ERR[SOUT+us],1,Labels[us],1);
	atomicAdd(&Bias[SOUT+us],Lrt*ActFun(A_f,ERR[SOUT+us],1,Labels[us],1));
	

								}
			
				}


				
__global__ void gpu_fbck_label_err(const float* const __restrict__ Wgts_,float* const __restrict__ ERR,const int ITER,const int* const __restrict__ DIM)
{
	const int NIN  = DIM[0];
	const int NOUT = DIM[1];
	const int SIN  = DIM[2];
	const int SOUT = DIM[3];

GPU_1D_KERN_LOOP(index,NIN ){

	const int ls = (index / 1 )         % NIN;		//Outputs / layer size
	
	ERR[SIN+ls]=0;


}

	
GPU_1D_KERN_LOOP(index, NOUT*NIN){

	const int nl = (index / 1 )                 % NOUT;		//Number of labels
	const int us = (index / NOUT )                % NIN;		//layer size		
															//Do a proper Error here
atomicAdd(&ERR[SIN+us], Wgts_[nl*NIN+us]*ERR[SOUT+nl]);			
										}

									

}

				

__global__ void gpu_fbck_layer_err(const float* const __restrict__ Wgts,float* const __restrict__ ERR,const int ITER,const int* const __restrict__ DIM){

/* Adapt later for Batch Val x BATCH --- FASTER --- NOT SURE ABOUT BATCH FOR THIS*/

	const int NIN  = DIM[0];
	const int NOUT = DIM[1];
	const int SIN  = DIM[2];
	const int SOUT = DIM[3];

GPU_1D_KERN_LOOP(index,NIN ){

	const int ls = (index / 1 )         % NIN;		//Outputs / layer size
	
	ERR[SIN+ls]=0;


}
	
	
GPU_1D_KERN_LOOP(index, NOUT*NIN){

	const int nl = (index / 1 )                 % NOUT;		//Number of labels
	const int us = (index / NOUT )                % NIN;		//layer size		
															//Do a proper Error here

atomicAdd(&ERR[SIN+us], Wgts[nl*NIN+us]*ERR[SOUT+nl]);			


}	
	
}

__global__ void gpu_inv_layer_err(float* const __restrict__ ERR,float* const __restrict__ ACT,float* const __restrict__ Bias,const int ITER,const int* const __restrict__ DIM,const float* const __restrict__ C_e,const int A_f)
{
	const int NIN  = DIM[0];
	const int SIN  = DIM[2];
	const float Lrt  = C_e[0];
GPU_1D_KERN_LOOP(index, NIN){

	const int us = (index / 1 )                % NIN;		//layer size

															//ERR[SIN+us] = ACT[SIN+us]*(1-ACT[SIN+us])*ERR[SIN+us];

ERR[SIN+us] = ActFun(A_f,ERR[SIN+us],1,ACT[SIN+us],1);		
atomicAdd(&Bias[SIN+us],Lrt*ActFun(A_f,ERR[SIN+us],1,ACT[SIN+us],1));

}
}




					/***************************WEIGHT UPDATE**************************************/

__global__ void gpu_label_wgt_updt(float* const __restrict__ WL_a,float* const __restrict__ P_w,float* const __restrict__ P_a,const float* const __restrict__ A_a,const float* const __restrict__ E_a,const int ITER,const int* const __restrict__ DIM,const float* const __restrict__ C_e)
{

	const int NIN  = DIM[0]; //Lower layer
	const int NOUT = DIM[1]; //Upper layer (Labels)
	const int SIN  = DIM[2];
	const int SOUT = DIM[3];
	const int WSTR = DIM[8];
	
	const float Lrt  = C_e[0];
	const float Mmt  = C_e[1];
	const float Dcy  = C_e[2];

GPU_1D_KERN_LOOP(index,NIN*NOUT ){

	const int nl = (index / 1 )                 % NOUT;		//Number of labels
	const int us = (index / NOUT )                % NIN;		//layer size	


atomicAdd(&WL_a[nl*NIN+us],Mmt*P_a[WSTR+nl*NIN+us]);				//Deltas	
	
atomicAdd(&WL_a[nl*NIN+us],Lrt*E_a[SOUT+nl]*A_a[SIN+us]);			//Update without Mmntm or Decay

atomicAdd(&WL_a[nl*NIN+us],-Dcy*P_w[WSTR+nl*NIN+us]);				//Preweights

	
	

}


}


__global__ void gpu_prewgt(const float* const __restrict__ W_a,float* const __restrict__ P_w,const int* const __restrict__ DM)
{	
		/* Run this before or after weight update to record previous weight*/

	const int WSTR = DM[8];
	const int WEND = DM[9];
	const int WSZE=WEND-WSTR;

GPU_1D_KERN_LOOP(index,WSZE ){

	const int nl = (index / 1 )                 % WSZE;		//SIZE OF WGTS
		

	P_w[nl] = W_a[nl];
										}

}

__global__ void gpu_pdlta(const float* const __restrict__ W_a,float* const __restrict__ P_a,float* const __restrict__ D_a,const int* const __restrict__ DM)
{	
		/* Run this before or after weight update to record previous weight*/
	
	const int WSTR = DM[8];
	const int WEND = DM[9];
	const int WSZE=WEND-WSTR;
	
GPU_1D_KERN_LOOP(index,WSZE ){

	const int nl = (index / 1 )                 % WSZE;		//SIZE OF WGTS
		
	P_a[nl] = D_a[nl];
		
		}
		
}


__global__ void gpu_dlta(const float* const __restrict__ W_a,float* const __restrict__ P_w,float* const __restrict__ D_a,const int* const __restrict__ DM)
{	
		/* Run this before or after weight update to record previous weight*/

	const int WSTR = DM[8];
	const int WEND = DM[9];
	const int WSZE=WEND-WSTR;
	
GPU_1D_KERN_LOOP(index,WSZE ){

	const int nl = (index / 1 )                 % WSZE;		//SIZE OF WGTS
		
	D_a[nl] = W_a[nl];
		
		}


GPU_1D_KERN_LOOP(index,WSZE ){

	const int nl = (index / 1 )                 % WSZE;		//SIZE OF WGTS
		
	atomicAdd(&D_a[nl],-P_w[nl]);
	

		}

}	



__global__ void gpu_wgt_updt_lrt(float* const __restrict__ W_a,float* const __restrict__ P_w,float* const __restrict__ P_a,const float* const __restrict__ A_a,const float* const __restrict__ E_a,const int ITER,const int* const __restrict__ DIM,const float* const __restrict__ C_e)
{
		/* This is the first weight update */
		
    const int NIN  = DIM[0]; //Lower layer
	const int NOUT = DIM[1]; //Upper layer (Labels)
	const int SIN  = DIM[2];
	const int SOUT = DIM[3];
	const int WSTR = DIM[8];
	
	const float Lrt  = C_e[0];
	const float Mmt  = C_e[1];
	const float Dcy  = C_e[2];
	
	
	
	GPU_1D_KERN_LOOP(index,NIN*NOUT ){

	const int nl = (index / 1 )                 % NOUT;		//Number of labels
	const int us = (index / NOUT )                % NIN;		//layer size	

	
	
	atomicAdd(&W_a[WSTR+nl*NIN+us],Mmt*P_a[WSTR+nl*NIN+us]);				//Deltas	
	
	atomicAdd(&W_a[WSTR+nl*NIN+us],Lrt*E_a[SOUT+nl]*A_a[SIN+us]);			//Lrate
	
	atomicAdd(&W_a[WSTR+nl*NIN+us],-Dcy*P_w[WSTR+nl*NIN+us]);				//Preweights
										}
}
	



class mlp{
	
public:

float *ACT,*BIAS,*B_a,*A_a,*W_a,*D_a,*P_a,*P_w,*E_a,*V_a;	/*LAYER X NEURON*/
float *ERR,*UE_a,*LE_a;   /*LAYER X ERROR*/			
float *WGT;  /*LAYER X NEURON X NEURON*/			
float *DLT;  /*LAYER X NEURON X NEURON*/
float *PREDLT,*PREWGT;	

float *LABL,*VAL;	/*LABEL */
float *LWGT,*L_a,*WL_a; /*LABEL X NEURON*/
int *LAYERS_,*PLAYR,*WPLAYR,*ACTFI;
int *P_l,*W_l,*D_m;
float *C_e;
int NL,NLAYS,BS,VS;
float Lrt,Mmnt,Decy;

clock_t startTime, endTime;

ofstream errfile;

			/*********************************************Initialise MLP****************************************/

mlp(int LAYERS[],const char* ACTF[],int NLAYERS,int NLABELS){
Lrt = 0.65; Mmnt = 0.0065; Decy = 0.000065;

errfile.open("errfile.dat",ios::out);

NLAYS = NLAYERS;
NL = NLABELS;
BS = 300;
LAYERS_ = new int[NLAYERS];
PLAYR = new int[NLAYERS];
WPLAYR = new int[NLAYERS];
ACTFI = new int[NLAYERS];

for(int i=0;i<NLAYS;i++){

ACTFI[i]=0;

if(ACTF[i][0]=='T'&&ACTF[i][1]=='A'&&ACTF[i][2]=='N'&&ACTF[i][3]=='H'){ACTFI[i]=0;}
if(ACTF[i][0]=='S'&&ACTF[i][1]=='I'&&ACTF[i][2]=='G'&&ACTF[i][3]=='M'){ACTFI[i]=1;}
if(ACTF[i][0]=='N'&&ACTF[i][1]=='O'&&ACTF[i][2]=='R'&&ACTF[i][3]=='M'){ACTFI[i]=2;}


/*
switch(ACTF[i]){

	case "TANH":
		ACTFI[i]=0;
	case "SIGM":
		ACTFI[i]=1;
	case "NORM":
		ACTFI[i]=2;

				}
*/

if(i==0){
PLAYR[0]  = LAYERS[0];
WPLAYR[0] = 0;
		}
if(i>0){
PLAYR[i]  = PLAYR[i-1]+LAYERS[i];				//Neurons
WPLAYR[i] = WPLAYR[i-1]+LAYERS[i]*LAYERS[i-1];	//Weights	
		}
LAYERS_[i] = LAYERS[i];		
							}

int MYTOT = WPLAYR[NLAYS-1]*3+PLAYR[NLAYS-1]*2+NL+NL*LAYERS_[NLAYS-1];

	//printf("MYTOT=%d\n",MYTOT);

int deviceCount = 0;
  
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  
 int dev;
 
 for (dev = 0; dev < deviceCount; ++dev) {
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
	
	float act_use = static_cast<float>(357.1+0.932297*(1024*sizeof(float)*MYTOT)/1000000000); 
	float act_tot = static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f);
	
	printf("  Total Memory usage:							%.0f MBytes \n",
			  act_use); 
				
	printf("  Total amount of global memory:				%.0f MBytes (%llu bytes)\n",
              act_tot,
              (unsigned long long)deviceProp.totalGlobalMem);
	
	
	
	if((act_use)>act_tot){
	printf("Your memory is full\n");
	exit(1);
	}
						
 }

ACT = new float[PLAYR[NLAYS-1]];
BIAS = new float[PLAYR[NLAYS-1]];  
ERR = new float[PLAYR[NLAYS-1]];
WGT = new float[WPLAYR[NLAYS-1]];
DLT = new float[WPLAYR[NLAYS-1]];
PREDLT = new float[WPLAYR[NLAYS-1]];
PREWGT = new float[WPLAYR[NLAYS-1]];

/*Show memory usage at start*/

for(int i=0;i<PLAYR[NLAYS-1];i++){

ACT[i] = 0;
BIAS[i] =getlrand(0,1);
ERR[i] = 0;					
										}
VAL  = new float[(NLABELS+PLAYR[0])*BS];						/* Input and Output / Labels *  Batch Size */
LABL = new float[NLABELS*5];
LWGT = new float[NLABELS*LAYERS[NLAYERS-2]];

for(int i=0;i<NLABELS;i++){
LABL[i]=0;}

for(int j=0;j<LAYERS[NLAYERS-2]*NLABELS;j++){
LWGT[j] = getlrand(0,1);
}


											/*Init weights and deltas*/
for(int i=0;i<WPLAYR[NLAYS-1];i++){

WGT[i] = getlrand(0,1);
DLT[i] = getlrand(0,1);
PREDLT[i] = getlrand(0,1);
PREWGT[i] = getlrand(0,1);
								
									}
							


}

				/**************************Report GPU memory******************************/

void report_mem(){

	
int deviceCount = 0;
  
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  
 int dev;
 
 for (dev = 0; dev < deviceCount; ++dev) {
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
	
	}
 
 // show memory usage of GPU

        size_t free_byte ;

        size_t total_byte ;

        cudaMemGetInfo( &free_byte, &total_byte ) ;


double free_db = (double)free_byte ;

        double total_db = (double)total_byte ;

        double used_db = total_db - free_db ;

        printf("GPU memory usage: used = %.1f, free = %.1f MB, total = %.1f MB\n",

            used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
 

}

				/*******************************GPU Memory********************************/


void dmemory(){


cudaMalloc((void **)&A_a, PLAYR[NLAYS-1] * sizeof(float));	//ACT
cudaMalloc((void **)&B_a, PLAYR[NLAYS-1] * sizeof(float));	//ACT
cudaMalloc((void **)&E_a, PLAYR[NLAYS-1] * sizeof(float));	//ERR
						
cudaMalloc((void **)&W_a, WPLAYR[NLAYS-1] *  sizeof(float));	//WGT
cudaMalloc((void **)&D_a, WPLAYR[NLAYS-1] *  sizeof(float));	//DLT
cudaMalloc((void **)&P_a, WPLAYR[NLAYS-1] *  sizeof(float));	//PREDLT
cudaMalloc((void **)&P_w, WPLAYR[NLAYS-1] *  sizeof(float));	//PWGT


cudaMalloc((void **)&V_a, BS * 3 * sizeof(float));	//VAL
cudaMalloc((void **)&L_a, NL * 5 * sizeof(float));	//LABL
cudaMalloc((void **)&WL_a, NL * LAYERS_[NLAYS-2] * sizeof(float));	//LWGT

cudaMalloc((void **)&P_l, NLAYS *  sizeof(int));	//Pointer to Act layers
cudaMalloc((void **)&W_l, NLAYS *  sizeof(int));	//Pointer to Wgt layers

cudaMemcpy(P_l, PLAYR, NLAYS * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(W_l, WPLAYR, NLAYS * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(A_a, ACT, PLAYR[NLAYS-1] * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(B_a, BIAS, PLAYR[NLAYS-1] * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(W_a, WGT,  WPLAYR[NLAYS-1] * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(WL_a, LWGT,  NL *  LAYERS_[NLAYS-2] * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(D_a, DLT,    WPLAYR[NLAYS-1] * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(P_a, PREDLT, WPLAYR[NLAYS-1] * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(P_w, PREWGT, WPLAYR[NLAYS-1] * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(L_a, LABL, NL * 5 * sizeof(float), cudaMemcpyHostToDevice);

}

void prep_val(float *VAL,int EPOCH){

VS = 3;					//Set VS
float xord[4][3];
xord[0][0]=0; xord[0][1]=0; xord[0][2]=-0.5;
xord[1][0]=0; xord[1][1]=1; xord[1][2]=0.5;
xord[2][0]=1; xord[2][1]=0; xord[2][2]=0.5;
xord[3][0]=1; xord[3][1]=1; xord[3][2]=-0.5;

int select=0;

for(int i=0;i<BS;i++){

if(0){
select = rand()%4;
}else{
select++;
if(select==4)select=0;
}

VAL[i*3]   = xord[select][0];
VAL[i*3+1] = xord[select][1];
VAL[i*3+2] = xord[select][2];

							
}

}

		/***************************************CPU Feedforward Call***********************************/

void cpu_allconn_call(int LAYER,int ITER){

//cout<<"Layer = "<<LAYER<<"\n";

int SIN,SOUT,NIN,NOUT;
float SUM=0;

		//Upper layers
if(LAYER>1){		
NIN = PLAYR[LAYER-1]-PLAYR[LAYER-2]; //NIN
NOUT = PLAYR[LAYER]-PLAYR[LAYER-1];	//NOUT
SIN = PLAYR[LAYER-2]; //SIN
SOUT = PLAYR[LAYER-1]; //SOUT
}else{
NIN = PLAYR[0]; //NIN
NOUT = PLAYR[1]-PLAYR[0];	//NOUT
SIN = 0; //SIN
SOUT = PLAYR[0]; //SOUT
}

if(ITER>BS-10){
cout<<"Layer == "<<LAYER<<"\n";
cout<<"Number neurons IN == "<<NIN<<"\n";
cout<<"Number neurons OUT == "<<NOUT<<"\n";
cout<<"Start Position of IN == "<<SIN<<"\n";
cout<<"Start Position of OUT == "<<SOUT<<"\n";
				}

if(LAYER==1){		/* Set Activations for Input Layer 0 */

for(int ls=0;ls<NIN;ls++){

ACT[SIN+ls] = ActCPU(ACTFI[0],VAL[ITER*3+ls],0,0,0); 
//VAL[ITER*3+ls];
							}
}

if(LAYER!=NLAYS-1){

for(int ls=0;ls<NOUT;ls++){

for(int col=0;col<NIN;col++){

SUM+=WGT[WPLAYR[LAYER-1]+col*NOUT+ls]*ACT[SIN+col];

}

ACT[SOUT+ls]=ActCPU(ACTFI[LAYER],SUM+BIAS[SOUT+ls],0,0,0);
							
SUM=0;
}

}else{

SUM=0;


for(int ls=0;ls<NOUT;ls++){

for(int col=0;col<NIN;col++){

SUM+=LWGT[col*NOUT+ls]*ACT[SIN+col];

}

LABL[ls] = ActCPU(ACTFI[LAYER],SUM+BIAS[SOUT+ls],0,0,0);

}
					}

}


void cpu_bperr_call(int EPOCH){

int NIN,NOUT,SIN,SOUT,WSTR,WEND,WSZE;
float DLT_;

prep_val(VAL,EPOCH);

for(int ITER=0;ITER<BS;ITER++){				

											/* Feedforward */	
for(int i=1;i<NLAYS;i++)
cpu_allconn_call(i,ITER);						/* Call layer ff for hidden layers */

for(int LAYER=NLAYS-1;LAYER>0;LAYER--){	

if(LAYER>1){
NIN  = PLAYR[LAYER-1]-PLAYR[LAYER-2];
NOUT = PLAYR[LAYER]-PLAYR[LAYER-1];
SIN  = PLAYR[LAYER-2];
SOUT = PLAYR[LAYER-1];
}else{
NIN  = PLAYR[0];			//Number of neurons before layer
NOUT = PLAYR[1]-PLAYR[0];	//Number of neurons in layer
SIN  = 0;					//Start of input neurons
SOUT = PLAYR[0];			//Start of output neurons
}
WSTR = WPLAYR[LAYER-1];
WEND = WPLAYR[LAYER];
WSZE=WEND-WSTR;

/* Feedback Error */

float sum=0;

if(LAYER==NLAYS-1){		/*Label Layer*/

for(int i=0;i<NL;i++){	//upper neuron (label)

ERR[SOUT+i] =  ActCPU(ACTFI[NLAYS-1],VAL[ITER*3+2]-LABL[i],1,LABL[i],1);

BIAS[SOUT+i] += Lrt*ERR[SOUT+i];

//LABL[i]-VAL[ITER*3+2]; //Calc output error
//(VAL[ITER*3+2]-LABL[i])*LABL[i]*(1-LABL[i]);

//errfile<<sqrt(pow(ERR[SOUT],2))<<"\n";
//errfile<<ERR[SOUT]<<"\n";
errfile<<sqrt(pow(VAL[ITER*3+2]-LABL[i],2))<<"\n";	

	}

						

for(int j=0;j<NIN;j++){	//lower neuron
for(int i=0;i<NL;i++){	//upper neuron (label)

sum += LWGT[j*NL+i]*ERR[SOUT+i];//sum for jth neuron in layer n-2

							}

ERR[SIN+j] = ActCPU(ACTFI[NLAYS-2],sum,1,ACT[SIN+j],1);	

BIAS[SIN+j] += Lrt*ERR[SIN+j];

sum = 0;						
								}




}else{					/*Hidden Layers*/

//LAYER is the layer we are feeding back error from

sum =0;

for(int j=0;j<NIN;j++){	//lower neuron
for(int i=0;i<NOUT;i++){//upper neuron

sum += WGT[WSTR+j*NOUT+i]*ERR[SOUT+i];

								}//upper	
ERR[SIN+j] = ActCPU(ACTFI[LAYER],sum,1,ACT[SIN+j],1);	

BIAS[SIN+j] += Lrt*ERR[SIN+j];

sum = 0;
									}//lower


}//End Error backprop
						/*Begin layer wise weight updates*/

for(int j=0;j<NIN;j++){	//lower neuron
for(int i=0;i<NOUT;i++){//upper neuron

/* The correct way N.B calc predlt and prewgt after update */

DLT_=Lrt*ERR[SOUT+i]*ACT[SIN+j]+Mmnt*PREDLT[WSTR+j*NOUT+i]-Decy*PREWGT[WSTR+j*NOUT+i];

if(LAYER==NLAYS-1){
LWGT[j*NOUT+i]+=DLT_;
}else{
WGT[WSTR+j*NOUT+i]+=DLT_;
}

PREDLT[WSTR+j*NOUT+i]=DLT_;

if(LAYER==NLAYS-1){
PREWGT[WSTR+i*NIN+j]=LWGT[j*NOUT+i];
}else{
PREWGT[WSTR+i*NIN+j]=WGT[WSTR+j*NOUT+i];
}

/* Makes no sense touse LWGT */

}}

if(ITER>BS-10){
if(LAYER==NLAYS-1){
cout<<"\n\n									****************New Run*************\n";
cout<<"Batch No."<<ITER<<"\n";
cout<<"Values="<<VAL[ITER*3+0]<<","<<VAL[ITER*3+1]<<","<<VAL[ITER*3+2]<<"\n";
}

cout<<"\n\n************Layer = "<<LAYER<<"*************\n";

cout<<"\nLayer Err's:\n";
for(int i=PLAYR[LAYER-1];i<PLAYR[LAYER];i++){

cout<<ERR[i]<<",";
							}
cout<<"\nLayer Act's:\n";						
for(int i=PLAYR[LAYER-1];i<PLAYR[LAYER];i++){

cout<<ACT[i]<<",";
							}	

if(LAYER==NLAYS-1){

cout<<"\nLayer Label's:\n";
for(int i=0;i<NL;i++){

cout<<LABL[i]<<",";
											}
cout<<"\nLabel Wgts&&Deltas:\n";
for(int i=0;i<NL*LAYERS_[NLAYS-2];i++){		

cout<<LWGT[i]<<":"<<DLT[i+WPLAYR[NLAYS-2]]<<" , ";
		
											}											
				}else{		
cout<<"\nLayer Wgts:\n";
for(int i=WPLAYR[LAYER-1];i<WPLAYR[LAYER];i++){		

if(i>WPLAYR[LAYER-1]-1&&i<WPLAYR[LAYER]){
cout<<"*";
}

cout<<WGT[i]<<":"<<DLT[i];

if(i>WPLAYR[LAYER-1]-1&&i<WPLAYR[LAYER]){
cout<<"*";
}		

cout<<",";
		}
		}

}
										}//Layers

if(ITER>BS-10){

cout<<"\n\n************Layer = "<<0<<"*************\n";

cout<<"\nLayer Err's:\n";
for(int i=0;i<PLAYR[0];i++){

cout<<ERR[i]<<",";
							}
	
cout<<"\nLayer Act's:\n";

for(int i=0;i<PLAYR[0];i++){

cout<<ACT[i]<<",";
							}

cout<<"\n\n									****************End Run*************\n";

				}

											}//Iter
cout<<"Done cpu backprop\n";

}

float ActCPU(int type, float v,int dir,float u, int seed){

if(type==0){
/*Tanh*/
if(dir==0){
return tanh(v);
}else{
return (1-pow((double)tanh(u),2.0))*v;
			}

			}
			
if(type==1){
/*Sigmoid*/
float sigmoid;

if(dir==0){
/*
if(v>0.5){					//this is actually hardlim not sigmoid
 sigmoid =  0.5;}else
 if(v<-0.5){
 sigmoid = 0;}
 else
 */
 sigmoid = 1 / (1 + expf(-v));

}else{
sigmoid = (u*(1-u)*v) ;
			}

return sigmoid;
			}
if(type==2){
/*Norm*/
if(dir==0){
return v;
}else{
return v;
			}

			}			
return 0;			

}

					/**********************************Backprop Call**************************************/

void gpu_bperr_call(int EPOCH){		/* Call ffwd to generate labels first*/

GpuInit myinit_aconn(1024,1024,1,1);	//GPU using 1024 grids and 1024 blocks
GpuInit* myinitaconn;
myinitaconn = &myinit_aconn;

									/*** Run this once for all epoch data ***/
prep_val(VAL,EPOCH);			   /* Do this to unpack data into batch size chunks for every epoch */
									/*** Put all data onto device at start not everytime you run this ***/

cudaMemcpy(V_a,VAL, BS * 3 * sizeof(float), cudaMemcpyHostToDevice);  /*Batch load Values Transfer to device*/
									

for(int ITER=0;ITER<BS;ITER++){				/*Remove this loop when batch inside kernel*/

if(0){
cout<<"\nBatch No."<<ITER<<"\n";
cout<<"Values="<<VAL[ITER*3+0]<<","<<VAL[ITER*3+1]<<"\n";
		}								
										/* Load Inputs do later in batch PLAYR[0] = Number Inputs */

	
											/* Batch needs to be on device so that all dims have batch BS */
											/* Feedoforward create batch x output ... batch x error */		

											/* Feedforward */	
for(int i=1;i<NLAYS;i++)
gpu_allconn_call(i,ITER);							
											/* Call layer ff for hidden layers */



int *DIM;
DIM = new int[11];
float CE[3];
CE[0] = Lrt;
CE[1] = Mmnt;
CE[2] = Decy;														

cudaMalloc((void **)&C_e, 3 * sizeof(float));	//D_m
cudaMemcpy(C_e, CE, 3 * sizeof(float), cudaMemcpyHostToDevice);

for(int LAYER=NLAYS-1;LAYER>0;LAYER--){								/*Backprop Error*/

//int NL_=0;
//if(LAYER==NLAYS-1)NL_=NL;
		
if(LAYER>1){		
DIM[0] = PLAYR[LAYER-1]-PLAYR[LAYER-2]; //NIN
DIM[1] = PLAYR[LAYER]-PLAYR[LAYER-1];	//NOUT
DIM[2] = PLAYR[LAYER-2]; //SIN
DIM[3] = PLAYR[LAYER-1]; //SOUT
}else{
DIM[0] = PLAYR[0]; //NIN
DIM[1] = PLAYR[1]-PLAYR[0];	//NOUT
DIM[2] = 0; //SIN
DIM[3] = PLAYR[0]; //SOUT
}
DIM[4] = 3;

DIM[8] = WPLAYR[LAYER-1];	//Weight start
DIM[9] = WPLAYR[LAYER];		//Weight end
DIM[10] = WPLAYR[NLAYS-1];	//Total weight size


cudaMalloc((void **)&D_m, 10 * sizeof(int));	//D_m
cudaMemcpy(D_m, DIM, 10 * sizeof(int), cudaMemcpyHostToDevice);


			/*************************************FEEEDBACK ERROR*******************************************/
if(1){

if(LAYER==NLAYS-1){		/*Label Layer*/

gpu_label_err<<<myinitaconn->grid,myinitaconn->block,0,myinitaconn->stream1>>>(L_a,V_a,E_a,ITER,D_m); /*calculate label error*/
gpu_inv_label_err<<<myinitaconn->grid,myinitaconn->block,0,myinitaconn->stream1>>>(E_a,A_a,B_a,L_a,ITER,D_m,C_e,ACTFI[NLAYS-1]); /*Inverse label error*/
gpu_fbck_label_err<<<myinitaconn->grid,myinitaconn->block,0,myinitaconn->stream1>>>(WL_a,E_a,ITER,D_m);/*Feedback label error*/
gpu_inv_layer_err<<<myinitaconn->grid,myinitaconn->block,0,myinitaconn->stream1>>>(E_a,A_a,B_a,ITER,D_m,C_e,ACTFI[NLAYS-2]); /*Inverse layer error*/

}else{				   /*Hidden Layers*/

gpu_fbck_layer_err<<<myinitaconn->grid,myinitaconn->block,0,myinitaconn->stream1>>>(W_a,E_a,ITER,D_m); /*Feedback layer error*/
gpu_inv_layer_err<<<myinitaconn->grid,myinitaconn->block,0,myinitaconn->stream1>>>(E_a,A_a,B_a,ITER,D_m,C_e,ACTFI[LAYER-1]); /*Inverse layer error*/

}
							
	}
/*Update Weights beneath this layer using error at this layer*/

if(1){

if(LAYER==NLAYS-1){			

/*

										** ** ** ** ****** ** ** ** **

Label Weights can be included with Network Weights - keeping them seperate serves no purpose and cannot include Decay and Momentum

													******
		***********************THIS CAN BE DONE IN ONE GO RATHER THAN LAYER BY LAYER (OUTSIDE OF LAYER LOOP)*************************

*/
				/*************************************UPDATE WEIGHTS*******************************************/

				//calc Deltas
gpu_dlta<<<myinitaconn->grid,myinitaconn->block,0,myinitaconn->stream1>>>(WL_a,P_w+DIM[8],D_a+DIM[8],D_m);	

	//Record Previous Weights
gpu_prewgt<<<myinitaconn->grid,myinitaconn->block,0,myinitaconn->stream1>>>(WL_a,P_w+DIM[8],D_m);

				//Update label weights	
gpu_label_wgt_updt<<<myinitaconn->grid,myinitaconn->block,0,myinitaconn->stream1>>>(WL_a,P_w,P_a,A_a,E_a,ITER,D_m,C_e);

				//Update PreDeltas
gpu_pdlta<<<myinitaconn->grid,myinitaconn->block,0,myinitaconn->stream1>>>(WL_a,P_a+DIM[8],D_a+DIM[8],D_m);


}else{

if(1){
				//calc Deltas
gpu_dlta<<<myinitaconn->grid,myinitaconn->block,0,myinitaconn->stream1>>>(W_a+DIM[8],P_w+DIM[8],D_a,D_m);	

	//Record Previous Weights
gpu_prewgt<<<myinitaconn->grid,myinitaconn->block,0,myinitaconn->stream1>>>(W_a+DIM[8],P_w+DIM[8],D_m);

				//Update LAYER weights
gpu_wgt_updt_lrt<<<myinitaconn->grid,myinitaconn->block,0,myinitaconn->stream1>>>(W_a,P_w,P_a,A_a,E_a,ITER,D_m,C_e);

				//Update PreDeltas
gpu_pdlta<<<myinitaconn->grid,myinitaconn->block,0,myinitaconn->stream1>>>(W_a+DIM[8],P_a+DIM[8],D_a+DIM[8],D_m);

		}

}
					}

/* 
										** ** ** ** ****** ** ** ** ** 

Weight Updates can be done in parallel for entire network - so move all of this down outside LAYER to BS loop 
													
													******
*/

				/*********************************PRINT OUT************************************************/



if(ITER>BS-10){

if(1){
cudaMemcpy(ERR, E_a, PLAYR[NLAYS-1] * sizeof(float), cudaMemcpyDeviceToHost);
cudaMemcpy(ACT, A_a, PLAYR[NLAYS-1] * sizeof(float), cudaMemcpyDeviceToHost);
cudaMemcpy(BIAS, B_a, PLAYR[NLAYS-1] * sizeof(float), cudaMemcpyDeviceToHost);
cudaMemcpy(LABL, L_a, NL *  sizeof(float), cudaMemcpyDeviceToHost);
cudaMemcpy(WGT, W_a, WPLAYR[NLAYS-1] * sizeof(float), cudaMemcpyDeviceToHost);
cudaMemcpy(DLT, D_a, WPLAYR[NLAYS-1] * sizeof(float), cudaMemcpyDeviceToHost);
cudaMemcpy(LWGT,WL_a,  NL *  LAYERS_[NLAYS-2] * sizeof(float), cudaMemcpyDeviceToHost);
}

if(LAYER==NLAYS-1){
cout<<"\n\n									****************New Run*************\n";
cout<<"Batch No."<<ITER<<"\n";
cout<<"Values="<<VAL[ITER*3+0]<<","<<VAL[ITER*3+1]<<","<<VAL[ITER*3+2]<<"["<<ACT[0]<<"|"<<ACT[1]<<"]\n";
}

cout<<"\n\n************Layer = "<<LAYER<<"*************\n";

cout<<"\nLayer Err's:\n";
for(int i=PLAYR[LAYER-1];i<PLAYR[LAYER];i++){

cout<<ERR[i]<<",";
							}
	
cout<<"\nLayer Act's:\n";						
for(int i=PLAYR[LAYER-1];i<PLAYR[LAYER];i++){

cout<<ACT[i]<<":"<<BIAS[i]<<",";
							}
			

if(LAYER==NLAYS-1){

cout<<"\nLayer Label's:\n";
for(int i=0;i<NL;i++){

cout<<LABL[i]<<":"<<BIAS[PLAYR[NLAYS-2]+i]<<",";
											}


cout<<"\nLable Wgts&&Deltas:\n";
for(int i=0;i<NL*LAYERS_[NLAYS-2];i++){		

cout<<LWGT[i]<<":"<<DLT[i+WPLAYR[NLAYS-2]]<<" , ";
		
											}
}else{		
cout<<"\nLayer Wgts:\n";
for(int i=WPLAYR[LAYER-1];i<WPLAYR[LAYER];i++){		

if(i>WPLAYR[LAYER-1]-1&&i<WPLAYR[LAYER]){
cout<<"*";
}

cout<<WGT[i]<<":"<<DLT[i];

if(i>WPLAYR[LAYER-1]-1&&i<WPLAYR[LAYER]){
cout<<"*";
}		

cout<<",";
		}
		}
		}

												}		//LAYER
												
												
if(ITER>BS-10){
//if(1){

cout<<"\n\n************Layer = "<<0<<"*************\n";

cout<<"\nLayer Err's:\n";
for(int i=0;i<PLAYR[0];i++){

cout<<ERR[i]<<",";
							}
	
cout<<"\nLayer Act's:\n";

for(int i=0;i<PLAYR[0];i++){

cout<<ACT[i]<<",";
							}

cout<<"\n\n									****************End Run*************\n";

				}
				
float* TERR;
TERR = new float[1];
cudaMemcpy(TERR, E_a+PLAYR[NLAYS-1]-1,sizeof(float), cudaMemcpyDeviceToHost);
errfile<<sqrt(pow(TERR[0],2))<<"\n";


												}		//BS	(remove this after putting BS in kernel replace with epoch)



free_dmemory();
errfile.close();

cout<<"Ran BPERR\n";


}

	

				/***************************************GPU Feedforward Call***********************************/


void gpu_allconn_call(int LAYER,int ITER){

GpuInit myinit_aconn(1024,1024,1,1);	//GPU using 1024 grids and 1024 blocks
GpuInit* myinitaconn;
myinitaconn = &myinit_aconn;

int NL_=0;
if(LAYER==NLAYS-1)NL_=NL;

//cout<<"Layer = "<<LAYER<<"\n";

int *DIM;
DIM = new int[5];

		//Upper layers
if(LAYER>1){		
DIM[0] = PLAYR[LAYER-1]-PLAYR[LAYER-2]; //NIN
DIM[1] = PLAYR[LAYER]-PLAYR[LAYER-1];	//NOUT
DIM[2] = PLAYR[LAYER-2]; //SIN
DIM[3] = PLAYR[LAYER-1]; //SOUT
}else{
DIM[0] = PLAYR[0]; //NIN
DIM[1] = PLAYR[1]-PLAYR[0];	//NOUT
DIM[2] = 0; //SIN
DIM[3] = PLAYR[0]; //SOUT
}
DIM[4] = PLAYR[0]+(PLAYR[NLAYS-1]-PLAYR[NLAYS-2]);


cudaMalloc((void **)&D_m, 5 * sizeof(float));	//D_m
cudaMemcpy(D_m, DIM, 5 * sizeof(int), cudaMemcpyHostToDevice);



gpu_ff_allconn<<<myinitaconn->grid,myinitaconn->block,0,myinitaconn->stream1>>>(W_a,WL_a,V_a,L_a,A_a,B_a,P_l,W_l,LAYER,NL_,D_m,ITER,ACTFI[LAYER]);

//gpu_act<<<myinitaconn->grid,myinitaconn->block,0,myinitaconn->stream1>>>(A_a,D_m,ACTFI[LAYER]);



if(0&&LAYER==NLAYS-1){
cudaMemcpy(ACT, A_a, PLAYR[NLAYS-1] * sizeof(float), cudaMemcpyDeviceToHost);
cudaMemcpy(LABL, L_a, NL *sizeof(float), cudaMemcpyDeviceToHost);
					}


if(0){
cout<<"Ran ACONN\n";
	}
}	

void free_memory(){

delete(ACT);
delete(ERR);
delete(WGT);
delete(DLT);
delete(PREDLT);
delete(LABL);
delete(LWGT);
delete(LAYERS_);


}

void free_dmemory(){

cudaFree(A_a);
cudaFree(E_a);
cudaFree(W_a);
cudaFree(P_a);
cudaFree(L_a);
cudaFree(WL_a);


}

~mlp(){

free_memory();

}	
	
};

/*

int main(){

int topo[] = {2,8,1};
const char* acts[] = {"NORM","TANH","TANH"};
int nlayers = 3;
int nlabels = 1;

mlp mynet(topo,acts,nlayers,nlabels);		//10 inputs , 3 hidden , 2 hidden, 3 layers , 2 labels



mynet.dmemory();			//Call this once
mynet.report_mem();

				/***********************************Training********************************************/

cout<<"\n\nCommence Training (Xor) !!\n\n";

if(1){
mynet.gpu_bperr_call(0);
	}else{
mynet.cpu_bperr_call(0);
	}


mynet.free_dmemory();


return(0);
}

*/

/*Put the DIM array in init and dmemory straight to memory for all layers DIM[layersxdim]*/
/*Decide who goes where right at the start*/

/*Compile with 'nvcc -arch=sm_50 -o minml.exe mini_mlp.cu'*/