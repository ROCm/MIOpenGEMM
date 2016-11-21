#ifndef REDIRECTION_HPP
#define REDIRECTION_HPP

namespace redirection{
template <typename T> //defined for double, float, cl_mem.

void redirect(bool & isColMajor, bool & tA, bool & tB, bool & tC, unsigned & m, unsigned & n, unsigned & lda, unsigned & ldb, T & a, T & b);

void confirm_redirection(bool isColMajor, bool tA, bool tB, unsigned m, unsigned n);

} //end of namespace
#endif
