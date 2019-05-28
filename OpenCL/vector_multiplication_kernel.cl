
__kernel void mult_vec1_gpu(__global float* matrix_2, 
          __global float* matrix_0, 
          __global float* matrix_1,
          int rowlength)
{
  
   int row = get_global_id(0); 
   int col = get_global_id(1);
   float result = 0;
   float var0 = 0;
   float var1 = 0;
   {
      for (int k = 0; k < rowlength; ++k)
      {
         var0 = matrix_0[col * rowlength + k];
         var1 = matrix_1[k * rowlength + row];
         result += var0 * var1;
      }
   }
   matrix_2[(row * rowlength) + col] = result;
}