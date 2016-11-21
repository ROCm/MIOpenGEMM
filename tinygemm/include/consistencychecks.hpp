#ifndef CONSISTENCYCHECKS_HPP
#define CONSISTENCYCHECKS_HPP

#include "problemgeometry.hpp"

namespace consistencychecks{

void check_ldx_mnk_consistent(const gemmgeometry::Geometry  & gemmgeomm);


void check_ldx_mnk_consistent(bool isColMajor, bool tA, bool tB, bool tC, unsigned lda, unsigned ldb, unsigned ldc, unsigned m, unsigned n, unsigned k);


}

#endif
