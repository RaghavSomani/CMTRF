#include "init.h"
#include "sigma.cpp"
#include <sys/resource.h>

int main()
{
	const rlim_t kStackSize = 512 * 1024 * 1024;   // min stack size = 16 MB
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
    
	// selecting dataset
	cout<<"0. movielens_small"<<endl;
	cout<<"1. movielens_medium"<<endl;
	cout<<"2. movielens_large"<<endl;
	cout<<"3. epinions"<<endl;
	cout<<"4. goodbooks"<<endl;
	cout<<"5. synthetic"<<endl;
	cout<<"6. yahoo"<<endl;
	cout<<"Dataset : ";
	int dataset;
	cin>>dataset;
	int i,j,r,n,p,split,k=0,ndcg_thresh = 30;
	ifstream file,test_file;
	if(dataset==0)
	{
		file.open("../datasets/u100k_mapped.data");
		n = 751;
		p = 1616;
		split = 80000;
	}
	else if(dataset==1)
	{
		file.open("../datasets/u1m_mapped.dat");
		n = 5301;
		p = 3682;
		split = 800000;
	}
	else if(dataset==2)
	{
		file.open("../datasets/u10m_mapped_subset.dat");
		n = 62007;
		p = 10586;
		split = 6400000;
	}
	else if(dataset==3)
	{
		file.open("../datasets/review_unique.dat");
		n = 77264;
		p = 150497;
		split = 760000;
	}
	else if(dataset==4)
	{
		file.open("../datasets/book_mapped.dat");
		n = 42813;
		p = 9403;
		split = 4000000;
	}
	else if(dataset==5)
	{
		file.open("../datasets/synthetic3.dat");
		n = 3000;
		p = 2000;
		split = 80000;
	}
	else if(dataset==6)
	{
		file.open("../datasets/ydata.txt");
		n = 15400;
		p = 1000;
		split = 311704;
	}
	else if(dataset==7)
	{
		file.open("../datasets/douban_mapped.dat");
		n = 2999;
		p = 3000;
		split = 123202;
	}
	else if(dataset==8)
	{
		file.open("../datasets/flixter_mapped.dat");
		n = 2307;
		p = 2945;
		split = 23556;
		int L = 10;
	}
	// getting all the optimal parametsrs into data object array stored in a seperate file params.txt
	int ND;
	ifstream param_file("params.txt");
	param_file>>ND;
	Dataset data[ND];
	get_opt_params(param_file,data,ND);

	// set if (i,j) indexes each for train and test set, known_ij is the union of both
	set<pair<int,int> > known_ij, train_ij, test_ij, testset_ij;
	int known = 0, test = 0, train, testset = 0;

	vector<T> tripletList_train;
	vector<T> tripletList_test;
	vector<T> tripletList_testset;

	cout<<n<<" "<<p<<endl;

	// reading data file in train/test split into triplets vectors in 80/20 ratio
	while (file>>i>>j>>r)
	{
		i++;
		j++;
		known_ij.insert(make_pair(i-1,j-1));
		if(known<split)
		{
			tripletList_train.push_back(T(i-1,j-1,r));
			train_ij.insert(make_pair(i-1,j-1));
		}
		else if(i<=n && j<=p && known>=split)
		{
			tripletList_test.push_back(T(i-1,j-1,r));
			test_ij.insert(make_pair(i-1,j-1));
			test++;
		}
		known++;
	}
	cout<<test<<endl;
	train = known-test;
	file.close();

	// loading testset
	// while (test_file>>i>>j>>r)
	// {
	// 	i++;
	// 	j++;
	// 	if(i<=n && j<=p)
	// 	{
	// 		testset_ij.insert(make_pair(i-1,j-1));
	// 		tripletList_testset.push_back(T(i-1,j-1,r));
	// 		testset++;
	// 	}
	// }
	// cout<<testset<<endl;
	// test_file.close();


	// declaring and initializing sparse matrices both in Column Major and Row Major form

	SpMatC M_trainC(n,p);
	SpMatR M_trainR(n,p);
	SpMatC M_testC(n,p);
	SpMatR M_testR(n,p);

	// SpMatC M_testsetC(n,p);
	// SpMatR M_testsetR(n,p);
	M_trainC.setFromTriplets(tripletList_train.begin(), tripletList_train.end());
	M_trainR.setFromTriplets(tripletList_train.begin(), tripletList_train.end());
	M_testC.setFromTriplets(tripletList_test.begin(), tripletList_test.end());
	M_testR.setFromTriplets(tripletList_test.begin(), tripletList_test.end());

	// M_testsetC.setFromTriplets(tripletList_testset.begin(), tripletList_testset.end());
	// M_testsetR.setFromTriplets(tripletList_testset.begin(), tripletList_testset.end());

	cout<<"Matrix M loaded"<<endl;
	cout<<"Total "<<known<<" ratings"<<endl;
	cout<<"Size of train_ij = "<<train_ij.size()<<endl;
	cout<<"Size of test_ij = "<<test_ij.size()<<endl;
	
	// Asking the user for the low rank k
	cout<<"k = ";
	cin>>k;

	/* Initializing the grid axis vectors of regularization hyperparameters
	 * To use optimal parameters set them to
	 * data[dataset].No_sigma.Kmap[k].KT.l1 and data[dataset].No_sigma.Kmap[k].KT.l2 for best Kendal Tau
	 * data[dataset].No_sigma.Kmap[k].TL.l1 and data[dataset].No_sigma.Kmap[k].TL.l2 for best Test Loss
	 */
	vector<double> L1 = {10,15,20,25,30,35,40,45,50};
	vector<double> L2 = {10,15,20,25,30,35,40,45,50};
	// the grid storing results (test loss and Kendall Tau respectively) of each pair of l1 and l2
	MatrixXd L12(L1.size(),L2.size());
	MatrixXd KTvalue(L1.size(),L2.size());
	MatrixXd NDCGvalue10(L1.size(),L2.size());
	MatrixXd NDCGmed10(L1.size(),L2.size());
	MatrixXd NDCGvalue20(L1.size(),L2.size());
	MatrixXd NDCGmed20(L1.size(),L2.size());
	MatrixXd NDCGvalue30(L1.size(),L2.size());
	MatrixXd NDCGmed30(L1.size(),L2.size());
	// declaring variables to be used further down the program
	MatrixXd U, V;
	pair<MatrixXd,MatrixXd>* UV;
	double train_loss, test_loss, tol;
	clock_t begin_time;
	int sz, Msz;
	if(dataset==0)
		tol = 1;
	else if(dataset==1)
		tol = 10;
	else if(dataset==2)
		tol = 100;
	else if(dataset==3)
		tol = 100;
	else if(dataset==4)
		tol = 50;
	else if(dataset==5)
		tol = 10;
	else if(dataset==6)
		tol = 10;
	// creating experiment directory
	string directory;
	create_directory(data,dataset,k,"No_sigma_/",directory);
	ofstream out(directory+"results.txt");
	// iteration over grid points
	int l1i = 0, l2j = 0;
	for(auto l1:L1)
	{
		for(auto l2:L2)
		{
			// creating directories for every grid point
			string dir_l12 = directory+tostring(l1)+"_"+tostring(l2)+"/";
			string cmd = "mkdir -p "+dir_l12;
			int dir_created = system(cmd.c_str());

			cout<<l1<<" "<<l2<<endl;

			// initialize factor matrices U and V
			UV = initialize_UV(n,p,k);
			U = UV->first;
			V = UV->second;
			// calculating initial train and test loss
			train_loss = loss(M_trainR,U,V,train_ij,l1,l2)/train;
			test_loss = loss(M_testR,U,V,test_ij,0,0)/test;
			// the vectors to store series of Train and Test losses
			vector<double> LTrain, LTest;

			// iteration till convergence
			for(int t=0;1;t++)
			{
				begin_time = clock();
				cout<<t<<" Train Loss = "<<train_loss<<" Test loss = "<<test_loss;
				// U and V convex optimization steps
				Ustep(M_trainC,M_trainR,U,V,l1);
				Vstep(M_trainC,M_trainR,U,V,l2);
				// calculeting train and test loss after each iteration and appending to the loss series
				train_loss = loss(M_trainR,U,V,train_ij,l1,l2)/train;
				test_loss = loss(M_testR,U,V,test_ij,0,0)/test;
				LTrain.push_back(train_loss);
				LTest.push_back(test_loss);

				// break if consecutive test loss difference falls below a data size dependent threshold - tol
				if(t>=5 && ( (LTest[t] > LTest[t-1]) || fabs(LTest[t] - LTest[t-1])<tol/test ) )
					break;
				// break if consecutive train loss difference falls below a data size dependent threshold - tol
				// if(t!=0 && (LTrain[t-1]-LTrain[t])<tol)
				// 	break;
				cout<<"  "<<float(clock () - begin_time )/CLOCKS_PER_SEC<<endl;
			}
			
			// writing results to files
			// out<<l1<<"\t"<<l2<<endl;
			// out<<LTest.back()<<endl;
			L12(l1i,l2j) = LTest.back();
			// KTvalue(l1i,l2j) = KT(M_testC,M_testR,U,V,"kt_No.txt");
			// cout<<"Test set error = "<<loss(M_testsetR,U,V,testset_ij,0,0)/testset;
			cout<<endl<<"Test MAE = "<<MAE(M_testR,U,V,test_ij)/test;
			cout<<endl;

			// pair<double,double> ndcg_stat = NDCG_Matrix(M_testC,M_testR,U,V,"ndcg_No_10.txt",ndcg_thresh,10);
			// NDCGvalue10(l1i,l2j) = ndcg_stat.first;
			// NDCGmed10(l1i,l2j) = ndcg_stat.second;

			// ndcg_stat = NDCG_Matrix(M_testC,M_testR,U,V,"ndcg_No_20.txt",ndcg_thresh,20);
			// NDCGvalue20(l1i,l2j) = ndcg_stat.first;
			// NDCGmed20(l1i,l2j) = ndcg_stat.second;

			// ndcg_stat = NDCG_Matrix(M_testC,M_testR,U,V,"ndcg_No_30.txt",ndcg_thresh,30);
			// NDCGvalue30(l1i,l2j) = ndcg_stat.first;
			// NDCGmed30(l1i,l2j) = ndcg_stat.second;

			// // cout<<KTvalue(l1i,l2j)<<endl;
			// cout<<endl<<NDCGvalue10(l1i,l2j)<<" "<<NDCGmed10(l1i,l2j)<<endl;
			// cout<<NDCGvalue20(l1i,l2j)<<" "<<NDCGmed20(l1i,l2j)<<endl;
			// cout<<NDCGvalue30(l1i,l2j)<<" "<<NDCGmed30(l1i,l2j)<<endl;
			// cout<<endl;
			
			// dumping U, V, Loss vectors, and KT to files
			// dumpMatrixToFile(dir_l12+"U.txt",U);
			// dumpMatrixToFile(dir_l12+"V.txt",V);
			// ofstream datafile(dir_l12+"losses.txt");
			// for(int i=0;i<LTrain.size();i++)
			// 	datafile<<LTrain[i]<<" "<<LTest[i]<<endl;
			// datafile.close();
			// datafile.clear();
			// datafile.open(dir_l12+"KT.txt");
			// datafile<<KTvalue(l1i,l2j);
			// datafile.close();

			l2j++;
		}
		l1i++;
		l2j = 0;
	}
	// writing and printing grid results to file and terminal respectively
	// out<<"LT"<<endl<<L12<<endl;
	// out<<"KT"<<endl<<KTvalue<<endl;
	cout<<"LT"<<endl<<L12<<endl;
	// cout<<"KT"<<endl<<KTvalue<<endl;
	// cout<<"NDCG_mean NDCG_median"<<endl;
	// cout<<NDCGvalue10<<" "<<NDCGmed10<<endl;
	// cout<<NDCGvalue20<<" "<<NDCGmed20<<endl;
	// cout<<NDCGvalue30<<" "<<NDCGmed30<<endl;

	// out.close();

	// playing a sound when program end :) #Note - only for linux OS
	string sound = "paplay /usr/share/sounds/freedesktop/stereo/complete.oga";
	int beep = system(sound.c_str());
	return 0;
}