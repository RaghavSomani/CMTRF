// This is a sample program as a demo of how to obtain the optimal parameters for any experiment
#include "init.h"
#include "sigma.cpp"

int main()
{
	int ND;
	ifstream file("params.txt");
	file>>ND;
	Dataset data[ND];
	get_opt_params(file,data,ND);

	cout<<data[0].sigma_K.Kmap[10].KT.l1<<" "<<data[0].sigma_K.Kmap[10].KT.l2<<" "<<data[0].sigma_K.Kmap[10].KT.K<<endl;
	return 0;
}