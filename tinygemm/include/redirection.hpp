#ifndef REDIRECTION_HPP
#define REDIRECTION_HPP

namespace tinygemm{
namespace redirection{

//template <typename T> //defined for double, float, cl_mem.
//void redirect(bool & isColMajor, bool & tA, bool & tB, bool & tC, unsigned & m, unsigned & n, unsigned & lda, unsigned & ldb, T & a, T & b);
template <typename TFloat>
void redirect(bool & isColMajor, bool & tA, bool & tB, bool & tC, unsigned & m, unsigned & n, unsigned & lda, unsigned & ldb, unsigned & a_offset, unsigned & b_offset, const TFloat *  & a, const TFloat * & b);



void redirect(bool & isColMajor, bool & tA, bool & tB, bool & tC, unsigned & m, unsigned & n, std::string & a, std::string & b);

//void redirect(bool & isColMajor, bool & tA, bool & tB, bool & tC, unsigned & m, unsigned & n, T & a, T & b);


void confirm_redirection(bool isColMajor, bool tA, bool tB, unsigned m, unsigned n);

} //end of namespace
}
#endif
