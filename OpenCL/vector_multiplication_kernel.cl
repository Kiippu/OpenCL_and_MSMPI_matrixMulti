
__kernel void mult_vec1_gpu(__global float* C, 
          __global float* A, 
          __global float* B,
          int N,
          int rowlength)
{
  
   int tx = get_global_id(0); 
   int ty = get_global_id(1);
   printf("%d",tx);
   printf("%d",ty);
   // value stores the element that is 
   // computed by the thread
   float value = 0;
   //if(tx < rowlength)
   {
      for (int k = 0; k < N; ++k)
      {
         float elementA = A[ty * N + k];
         float elementB = B[k * N + tx];
         value += elementA * elementB;
      }
   }
   // for (int k = 0; k < N; k++)
	// {
	// 	for (int i = 0; i < rowlength; i++)
	// 	{
			// for (int j = 0; j < N; j++)
         // {
         //    float elementA = A[k * rowlength + j];
         //    float elementB = B[j * rowlength + i];
         //    value += elementA * elementB;
         // }
	// 	}
	// }
 
   // Write the matrix to device memory each 
   // thread writes one element
   C[(tx * N) + ty] = value;
}


__kernel void mult_vec0_gpu(__global float* C, 
          __global float* A, 
          __global float* B, 
          int wA, int wB)
{
  
   int tx = get_global_id(0); 
   int ty = get_global_id(1);
 
   // value stores the element that is 
   // computed by the thread
   float value = 0;
   for (int k = 0; k < wA; ++k)
   {
      float elementA = A[ty * wA + k];
      float elementB = B[k * wB + tx];
      value += elementA * elementB;
   }
 
   // Write the matrix to device memory each 
   // thread writes one element
   C[ty * wA + tx] = value;
}