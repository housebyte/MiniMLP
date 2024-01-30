#include <algorithm>
#include <complex>
#include <iostream>
#include <math.h>
#include <vector>


/*Begin 1DKern definition */

/*This is a direct copy of Tensorflows 1DKern code*/

namespace detail {
template <typename T>
class GpuGridRange {

	struct Iterator {
		__device__ Iterator(T index, T delta) : index_(index), delta_(delta) {}
		__device__ T operator*() const { return index_;}
		__device__ Iterator& operator++() {
			index_ += delta_;
			return *this;
		}

		__device__ bool operator!=(const Iterator& other) const {
			bool greater = index_ > other.index_;
			bool less = index_ < other.index_;
			if(!other.delta_){
			return less;
			}
			if(!delta_){
			return greater;
			}	

		return less || greater;
		}	

		private:
		T index_;
		const T delta_;

	};	//end Iterator struct


	public:
       	__device__ GpuGridRange(T begin,T delta,T end)
		: begin_(begin),delta_(delta),end_(end) {}
	
	__device__ Iterator begin() const {return Iterator(begin_,delta_); }
	__device__ Iterator end() const {return Iterator(end_,0);}

	private:
	T begin_;
	T delta_;
	T end_;	
	


};	//end GPU class class
};	//end namespace detail

template <typename T>	//Allows you to use GPU iterator with all data types
__device__ detail::GpuGridRange<T> GpuGridRangeX(T count) {
return detail::GpuGridRange<T>(

	/*begin*/blockIdx.x * blockDim.x + threadIdx.x,
	/*delta*/gridDim.x * blockDim.x, /*end*/count
				);

}

template <typename T>	//Allows you to use GPU iterator with all data types
__device__ detail::GpuGridRange<T> GpuGridRangeY(T count) {
return detail::GpuGridRange<T>(

	/*begin*/blockIdx.y * blockDim.y + threadIdx.y,
	/*delta*/gridDim.y * blockDim.y, /*end*/count
				);

}
template <typename T>	//Allows you to use GPU iterator with all data types
__device__ detail::GpuGridRange<T> GpuGridRangeZ(T count) {
return detail::GpuGridRange<T>(

	/*begin*/blockIdx.z * blockDim.z + threadIdx.z,
	/*delta*/gridDim.z * blockDim.z, /*end*/count
				);

}


#define GPU_1D_KERN_LOOP(i, n) \
  for (int i : ::GpuGridRangeX<int>(n))

#define GPU_AXIS_KERNEL_LOOP(i, n, axis) \
  for (int i : ::GpuGridRange##axis<int>(n))


/*End 1DKern definition*/


class GpuInit{
public:

dim3 grid;
dim3 block;
cudaStream_t stream1;	//Could be a array of streams for multiple streams

GpuInit(unsigned int gridsizeX,unsigned int blocksizeX,unsigned int gridsizeY,unsigned int blocksizeY){

/*If using GPU_1D_KERN_LOOP */

grid.x = gridsizeX;   grid.y = gridsizeY;
block.x = blocksizeX; block.y= blocksizeY;

cudaStreamCreate(&stream1);


//cout<<"Grid dimensions are: "<<grid.x<<"--"<<grid.y<<"--"<<grid.z<<"\n";
//cout<<"Block dimensions are: "<<block.x<<"--"<<block.y<<"--"<<block.z<<"\n";

}

		};


__device__ double atomicAdd__(double* address, double val)
{
    unsigned long long int* address_as_ull =
                             (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}


__device__ double atomicAdd_(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

