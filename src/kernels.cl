#define A(IX  ,  IY)         a[(IY)*width + (IX)]
#define BLOCK_SIZE 8

__kernel
void
reduction(
  __global float *c,
  __local float *scratch, //local to THIS workgroup
  __const int sz,
  __global float *result
  )
{
  int gsz = get_global_size(0);
  int gix = get_global_id(0);
  int lsz = get_local_size(0);
  int lix = get_local_id(0);

float acc = 0;
  for ( int cix = get_global_id(0); cix < sz; cix += gsz )
    acc += c[cix];

  scratch[lix] = acc;
  barrier(CLK_LOCAL_MEM_FENCE); // doesn't seem to have an effect?

  for(int offset = lsz/2; offset > 0; offset /= 2) {
    if ( lix < offset )
      scratch[lix] += scratch[lix+offset];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if ( lix == 0 )
    result[get_group_id(0)] = scratch[0];
}


__kernel
void
diffuseNew(
    __global float *a,
    __const int width,
    __const int height,
    __const float dif_coeff
    )
{


  int tx = get_local_id(1);
  int ty = get_local_id(0);
  
  int ix = get_global_id(1);
  int iy = get_global_id(0);
  
  int sx = get_local_size(1);
  int sy = get_local_size(0);

  __local float S[BLOCK_SIZE+2][BLOCK_SIZE+2];
    
  // copy to local buffer
  if(tx == 0 && ix > 0)
      S[0][ty+1] = A(ix-1, iy);

  if (ty==0 && iy>0)
      S[tx+1][0] = A(ix, iy-1);
  
  if((tx==sx-1) && (ix < width-1))
      S[BLOCK_SIZE+1][ty+1] = A(ix+1, iy);
  
  if(ty == sy-1 && iy < height-1)
      S[tx+1][BLOCK_SIZE+1] = A(ix, iy+1);

  S[tx+1][ty+1] = A(ix, iy);
  barrier(CLK_LOCAL_MEM_FENCE); 
  
  float h0 = S[tx+1][ty+1];
  float h =  -4.*h0;

  // --  Do the stencil operation --

  if(ix > 0)
     h += S[tx][ty+1];
  if(ix < width-1)
     h += S[tx+2][ty+1];
  if(iy > 0)
     h += S[tx+1][ty];
  if(iy < height-1)
     h += S[tx+1][ty+2];


  barrier(CLK_LOCAL_MEM_FENCE); // needed?

// -- copy to global buffer
  if(ix>0 && ix < width && iy > 0 && iy < height){
        A(ix,iy) = h0 + h*dif_coeff;
    }
}

