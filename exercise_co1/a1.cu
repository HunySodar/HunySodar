#include <stdio.h>
#include <math.h>

#define M 1000
#define N 500
#define K 1000
#define MAX 100000

#define block_size 16

//使用统一内存，不需要显式的内存拷贝
__managed__ int a[M*N], b[N*K], c[M*K];

//共享内存，是一块可以被同一block中的所有线程访问的内存
__global__ void matix_gpu(int *a, int *b, int *c, int m, int n, int k)
{
    //定义共享内存
    __shared__ int a_tile[block_size][block_size];
    __shared__ int b_tile[block_size][block_size];
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    int tmp = 0;
    for (int i = 0;i<=n/block_size;i++)
    {
        //将a的一块数据拷贝到共享内存中
        //对于结果矩阵c[y][x],需要将a[y][threadIdx.x]保存到共享内存a_tile[threadIdx.y][threadIdx.x]中
        //完成一个分块矩阵的计算之后，将a[y][threadIdx.x+block_size]保存到共享内存a_tile[threadIdx.y][threadIdx.x]中
        if (i*block_size+threadIdx.x<n && y<m)
        {
            a_tile[threadIdx.y][threadIdx.x] = a[y*n+i*block_size+threadIdx.x];
        }
        else
        {
            a_tile[threadIdx.y][threadIdx.x] = 0;
        }
        //将b的一块数据拷贝到共享内存中
        //对于结果矩阵c[y][x],需要将b[threadIdx.y][x]保存到共享内存b_tile[threadIdx.y][threadIdx.x]中
        if (i*block_size+threadIdx.y<n && x<k)
        {
            b_tile[threadIdx.y][threadIdx.x] = b[(i*block_size+threadIdx.y)*k+x];
        }
        else
        {
            b_tile[threadIdx.y][threadIdx.x] = 0;
        }
        //同步，等待所有线程将数据拷贝到共享内存中
        __syncthreads();
        //对于结果矩阵的每个元素，需要找到对应的行和列，然后进行相乘求和
        //c[i][j] = a[i][0]*b[0][j] + a[i][1]*b[1][j] + ... + a[i][n-1]*b[n-1][j]
        for (int j = 0; j < block_size; j++)
        {
            tmp += a_tile[threadIdx.y][j] * b_tile[j][threadIdx.x];
        }
        //同步，等待所有线程完成计算
        __syncthreads();
    }
    if (y<m && x<k)
    {
        c[y*k+x] = tmp;
    }
}

//矩阵转置函数，使用共享内存用作缓存
//因为全局内存存在合并访存，所以使用共享内存用作缓存，先从全局内存中读取数据a[3][2]到共享内存a[3][2]对应位置中，
//然后从共享内存中读取数据，在每个block中进行考虑
__global__ void trans(int in[M][N], int out[N][M])
{
    //定义共享内存
    __shared__ int tile[block_size][block_size+1];
    //计算线程的全局索引
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x<N && y<M)
    {
        //将数据从输入矩阵中读取到共享内存中
        tile[threadIdx.y][threadIdx.x] = in[y][x];
    }
    
    //同步，等待所有线程将数据拷贝到共享内存中
    __syncthreads();
    //计算转置矩阵的索引
    //进行转置之后，block看作一个整体，它的索引发生改变为blockIdx.x, blockIdx.y
    //在每个block中找到对应的位置，为blockIdx.x*blockDim.x+threadIdx.y, blockIdx.y*blockDim.y+threadIdx.x
    int x1 = blockIdx.x * blockDim.x + threadIdx.y;
    int y1 = blockIdx.y * blockDim.y + threadIdx.x;
    if(x1<M && y1<N)
    {
        //将数据从共享内存中读取到输出矩阵中
        out[y1][x1] = tile[threadIdx.x][threadIdx.y];
    }
}

//对于cpu计算矩阵乘法，需要遍历结果矩阵m*k的每一个元素
void matrix_cpu(int *a, int *b, int *c, int m, int n, int k)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j <k; j++)
        {
            int sum = 0;
            //对于结果矩阵的每个元素，需要找到对应的行和列，然后进行相乘求和
            //c[i][j] = a[i][0]*b[0][j] + a[i][1]*b[1][j] + ... + a[i][n-1]*b[n-1][j]
            for (int l = 0; l < n; l++)
            {
                sum += a[i*n+l] * b[l*k+j];
            }
            c[i*k+j] = sum;
        }
    }
}

//求一个数组的所有元素的和
__global__ void sum_gpu(int *a, int *sum, int n)
{
    //定义共享内存
    __shared__ int tile[block_size];
    //在这里假设线程的数量远小于数组的长度，在读入数据到共享内存的时候，每个线程读入多个元素
    //计算线程的全局索引
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int shared_tmp = 0;
    for(int idx=x;idx<n;idx+=blockDim.x*gridDim.x)
    {
        shared_tmp += a[idx];
    }
    tile[threadIdx.x] = shared_tmp;
    
    //同步，等待所有线程将数据拷贝到共享内存中
    __syncthreads();
    //对于每个block中的数据进行求和
    int tmp = 0;
    for (int i = block_size/2; i >= 1; i/=2)
    {
        if (threadIdx.x<i)
        {
            tmp += tile[threadIdx.x+i];
        }
        __syncthreads();
        if(threadIdx.x<i)
        {
            tile[threadIdx.x] = tmp;
        }
    }
    //将每个block中的结果保存到全局内存中,此时每个block中的tile[0]保存的是每个block中的和
    if (threadIdx.x==0)
    {
        //对于同一内存写入时，需要使用原子操作，因为多个线程可能同时写入
        atomicAdd(sum, tile[0]);
    }
}

int main()
{
    for (int i = 0; i <N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            a[i*M+j] = rand()%100;
        }
    }
    for (int i = 0; i <K; i++)
    {
        for (int j = 0; j < N; j++)
        {
            b[i*N+j] = rand()%100;
        }
    }

    //使用二维线程块
    dim3 block(block_size, block_size);
    dim3 grid((K+block.x-1)/block.x, (M+block.y-1)/block.y);
}
