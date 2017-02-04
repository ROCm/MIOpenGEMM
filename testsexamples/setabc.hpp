#ifndef SETABC_HPP
#define SETABC_HPP

namespace setabc{

template <typename TFloat>     
void set_abc(std::vector<TFloat> & v_a, std::vector<TFloat> & v_b, std::vector<TFloat> & v_c, const tinygemm::TinyGemmGeometry & gg, const tinygemm::TinyGemmOffsets & toff){
    
  size_t n_a = gg.lda * (gg.tA == gg.isColMajor ? gg.m : gg.k) + toff.oa;
  size_t n_b = gg.ldb * (gg.tB == gg.isColMajor ? gg.k : gg.n) + toff.ob;
  size_t n_c = gg.ldc * (gg.tC == gg.isColMajor ? gg.m : gg.n) + toff.oc; 
  
  /* fill matrices with random floats. It is important to fill them with random floats, 
   * as if they're integers, the kernel can, and does, cheat! (runs faster) */

  v_a.resize(n_a); 
  for (size_t i = 0; i < n_a; ++i){
    v_a[i] = TFloat(rand() % 1000) / 1000. - 0.5;
  }
  
  v_b.resize(n_b);
  for (size_t i = 0; i < n_b; ++i){
    v_b[i] = TFloat(rand() % 1000) / 1000. - 0.5;
  }


  v_c.resize(n_c);
  for (size_t i = 0; i < n_c; ++i){
    v_c[i] = TFloat(rand() % 1000) / 500 - 1.;
  }
}

}

#endif
