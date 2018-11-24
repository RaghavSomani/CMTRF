#include "init.h"
#include "sigma.cpp"
#include <sys/resource.h>

double DCG(double a[], int len)
{
	double dcg = 0.0;
	for(int i=0;i<len;i++)
		dcg += (a[i]/log(i+2));
	return log(2)*dcg;
}

double NDCG(const VectorXd& arr1, const VectorXd& arr2, int len)
{
	vector<pair<double,double> > V;
	for(int i=0;i<len;i++)
		V.push_back(make_pair(arr1(i),arr2(i)));
	sort(V.begin(),V.end(),greater<pair<double,double> >());
	double a[len], b[len];
	for(int i=0;i<len;i++)
	{
		a[i] = V[i].first;
		b[i] = V[i].second;
	}
	double dcg = DCG(b,len);
	sort(b,b+len,greater<double>());
	double idcg = DCG(b,len);
	return dcg/idcg;
}

int main()
{
	VectorXd actual(5), pred(5);
	actual<<1,2,3,4,5;
	pred<<5.1,4.1,3.1,2.1,1.1;
	cout<<NDCG(pred,actual,5);
	return 0;
}