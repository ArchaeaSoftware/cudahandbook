/*
 * http://bytes.com/topic/c/answers/544283-implementation-kahan-sum-algorithm
 */

struct KahanAdder {
	double m_sum; /* The current working sum. */
	double m_carry; /* The carry from the previous operation */
	double m_temp; /* A temporary used to capture residual bits of precision */
	double m_y; /* A temporary used to capture residual bits of precision */

	KahanAdder() {
		m_sum = 0.0;
		m_carry = 0.0;
		m_temp = 0.0;
		m_y = 0.0;
	}
	KahanAdder& operator += ( double d ) {
		m_y = d - m_carry;
		m_temp = m_sum + m_y;
		m_carry = (m_temp - m_sum) - m_y;
		m_sum = m_temp;
		return *this;
	}
	operator double() { return m_sum; }
};

#ifdef UNIT_TEST
#include <stdio.h>
#include <stdlib.h>
int main(void)
{
KahanAdder_t k = {0};
double d;
double standard_sum = 0;
size_t s;

for (s = 0; s < 10000; s++) {
d = rand() / (rand() * 1.0 + 1.0);
add(d, &k);
standard_sum += d;
}
printf("Standard sum = %20.15f, Kahan sum = %20.15f\n",
standard_sum, k.sum_);
return 0;
}
#endif
