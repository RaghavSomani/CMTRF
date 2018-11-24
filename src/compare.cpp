#include "init.h"
#include "sigma_L.cpp"
#include <sys/resource.h>

int main()
{
	const rlim_t kStackSize=512*1024*1024;   // min stack size = 512 MB
    struct rlimit rl;
    int result;

    result = getrlimit(RLIMIT_STACK, &rl);
    cout<<rl.rlim_cur/(1024*1024)<<endl;
    if (result == 0)
        if (rl.rlim_cur < kStackSize)
        {
            rl.rlim_cur = kStackSize;
            result = setrlimit(RLIMIT_STACK, &rl);
            if (result != 0)
                fprintf(stderr, "setrlimit returned result = %d\n", result);
        }
    int i,j,r,n,p,split,k=0,ndcg_thresh = 30, L = 5, r_hat;
	ifstream file, test_file;
	file.open("../../neural-net-matrix-factorization-master/data/ml-100k/split/u100k_full.hat");
	n = 751;
	p = 1616;
	split = 80000;

	set<pair<int,int> > known_ij, train_ij, test_ij, testset_ij;
	int known = 0, test = 0, train, testset;

	vector<T> tripletList_train;
	vector<T> tripletList_test;
	vector<T> tripletList_testset;

	vector<double> truth;

	while (file>>i>>j>>r>>r_hat)
	{
		i++;
		j++;
		known_ij.insert(make_pair(i-1,j-1));
		if(known<split)
		{
			tripletList_train.push_back(T(i-1,j-1,r_hat));
			train_ij.insert(make_pair(i-1,j-1));
		}
		else if(i<=n && j<=p && known>=split)
		{
			tripletList_test.push_back(T(i-1,j-1,r_hat));
			test_ij.insert(make_pair(i-1,j-1));
			test++;
		}
		known++;
	}
}