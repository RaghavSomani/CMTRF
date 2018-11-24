#include "init.h"
#include "sigma_L.cpp"
#include "km.cpp"
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
	cout<<"6. movielens_large_full"<<endl;
	cout<<"7. douban"<<endl;
	cout<<"8. flixter"<<endl;
	cout<<"9. movielens_small_orig"<<endl;
	cout<<"Dataset : ";
	int dataset;
	cin>>dataset;
	int i,j,n,p,split,k=0,ndcg_thresh=30,L=5;
	double r;
	ifstream file, test_file;
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
		file.open("../datasets/review_unique.dat");
		n = 77264;
		p = 150497;
		split = 760000;
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
		file.open("../datasets/u10m_transformed.dat");
		n = 55790;
		p = 8598;
		split = 7200000;
		L = 10;
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
		L = 10;
	}
	else if(dataset==9)
	{
		file.open("../datasets/u100k_orig_mapped.dat");
		n = 943;
		p = 1650;
		split = 80000;
	}

	// getting all the optimal parametsrs into data object array stored in a seperate file params.txt
	int ND;
	ifstream param_file("params.txt");
	param_file>>ND;
	Dataset data[ND];
	get_opt_params(param_file,data,ND);

	// set if (i,j) indexes each for train and test set, known_ij is the union of both
	set<pair<int,int> > known_ij, train_ij, test_ij, testset_ij;
	int known = 0, test = 0, train, testset;

	vector<T> tripletList_train;
	vector<T> tripletList_test;
	vector<T> tripletList_testset;

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

	/* Assign the optimal parameters for the corresponding N-sigma experiment
	 * To use optimal parameters set them to
	 * data[dataset].sigma_N.Kmap[k].KT.l1 and data[dataset].sigma_N.Kmap[k].KT.l2 for best Kendal Tau
	 * data[dataset].sigma_N.Kmap[k].TL.l1 and data[dataset].sigma_N.Kmap[k].TL.l2 for best Test Loss
	 */
	// vector<double> L1 = {data[dataset].sigma_N.Kmap[k].TL.l1};
	// vector<double> L2 = {data[dataset].sigma_N.Kmap[k].TL.l2};
	// double opt_l1,opt_l2;
	// if(dataset==3)
	// {
	// 	if(k==1)
	// 	{
	// 		opt_l1 = 30;
	// 		opt_l2 = 20;
	// 	}
	// 	else if(k==5)
	// 	{
	// 		opt_l1 = 7;
	// 		opt_l2 = 18;
	// 	}
	// 	else if(k==10)
	// 	{
	// 		opt_l1 = 7;
	// 		opt_l2 = 30;
	// 	}
	// 	else if(k==25)
	// 	{
	// 		opt_l1 = 7;
	// 		opt_l2 = 17;
	// 	}
	// 	else if(k==50)
	// 	{
	// 		opt_l1 = 12;
	// 		opt_l2 = 10;
	// 	}
	// }
	vector<double> L1 = {2.5};
	vector<double> L2 = {5};

	MatrixXd L12(L1.size(),L2.size());
	MatrixXd KTvalue(L1.size(),L2.size());
	MatrixXd NDCGvalue10(L1.size(),L2.size());
	MatrixXd NDCGmed10(L1.size(),L2.size());
	MatrixXd NDCGvalue20(L1.size(),L2.size());
	MatrixXd NDCGmed20(L1.size(),L2.size());
	MatrixXd NDCGvalue30(L1.size(),L2.size());
	MatrixXd NDCGmed30(L1.size(),L2.size());

	// declaring variables to be used further down the program
	MatrixXd U(n,k), V(p,k);
	pair<MatrixXd,MatrixXd>* UV;
	int sz, Msz, dir_created;
	double train_loss, test_loss, tol = 1;
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

	// Asking the user for the number of clusters if optimal not there in params.txt
	int K;
	int opt_K = -1;
	if(opt_K==-1)
	{
		cout<<"1. Number of clusters = ";
		cin>>K;
	}
	else
	{
		K = opt_K;
		cout<<"2. Number of clusters = "<<K<<endl;
	}

	// creating experiment directory
	string directory;
	create_directory(data,dataset,k,"k_sigma_/",directory);
	ofstream out(directory+"results.txt");

	// Keep the frequency matrix, N[i][r-1] = Number of ratings r in M_train[i]
	int N[n][L];
	memset(N, 0, sizeof(N[0][0])*n*L);
	int N_total[n];
	fill_n(N_total, n, 0);

	// getting frequency of ratings in each row and the users who rated just one movie
	set<int> unique_ratings, users_with_single_rating;
	for(int i=0;i<n;i++)
	{
		for(SpMatR::InnerIterator it(M_trainR,i); it; ++it)
		{
			N[i][(int)it.value()-1]++;
			N_total[i]++;
			unique_ratings.insert((int)it.value());
		}
		if(unique_ratings.size()==1)
			users_with_single_rating.insert(i);
		unique_ratings.clear();
	}

	// converting frequency counts in W, eps is defined in init.h
	ofstream histfile("hist_medium.txt");
	double W[n][L];
	for(int i=0;i<n;i++)
	{
		transform(N[i], N[i]+L, W[i],bind2nd(divides<double>(),(double)N_total[i]));
		for(int j=0;j<L;j++)
			histfile<<W[i][j]<<" \n"[j==(L-1)];
	}
	histfile.close();

	// z will store the sigma function array for each row of M_train
	// defining margins such that z[i+1] - z[i] >= margin[i] for i in {0,1,2,3}
	double margins[L-1];// = {eps,eps,eps,eps};
	fill_n(margins, L-1, eps);
	double* z[n];

	// declaring sparse Matrices to store sigma_i(M_train[i]) and sigma_i(M_test[i])
	SpMatC M_C(n,p), M_test_C(n,p);
	SpMatR M_R(n,p), M_test_R(n,p);

	// declaring sigma inverse linear splines
	tk::spline sig_inv[n];

	// array to store histogram of clusters
	int cluster_count[K];
	fill_n (cluster_count,K,0);

	// weight arrays for each cluster, to be used in pooled PAV
	double* W_cluster[K];
	for(int k=0;k<K;k++)
		W_cluster[k] = new double[L];

	// Vectors of concatenated cluster M_train[cluster(k)] and M_test[cluster(k)]
	VectorXd M_expanded_cluster[K], M_test_expanded_cluster[K];

	// pair of train and test sizes for clusters, to be used during pooled PAV
	pair<int,int> pair_size;

	// cluster centers sigma function arrays for each cluster
	double* z_cluster[K];

	// cluster center sigma inverse linear splines for each cluster
	tk::spline sig_inv_cluster[K];


	// iteration over grid points
	clock_t begin_time;
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
			UV = initialize_UV(n,p,k,L);
			U = UV->first;
			V = UV->second;

			// assigning sigma(M_train) and sigma(M_test) to original train and test matrices
			M_C = M_trainC;
			M_test_C = M_testC;
			M_R = M_trainR;
			M_test_R = M_testR;

			// calculating initial train and test loss
			train_loss = loss(M_trainR,U,V,train_ij,l1,l2);
			test_loss = loss(M_testR,U,V,test_ij,0,0)/test;

			// the vectors to store series of Train and Test losses
			vector<double> LTest, LTrain;
			// variables used for moving average early stopping
			double last_few = 0, second_last_few = 0;

			// iteration till convergence
			for(int t=0;1;t++)
			{
				begin_time = clock();
				cout<<t<<" Train Loss = "<<train_loss<<" Test loss = "<<test_loss<<"  ";

				// U and V convex optimization steps
				Ustep(M_C,M_R,U,V,l1);
				Vstep(M_C,M_R,U,V,l2);

				// N sigma step
				for(int i=0;i<n;i++)
				{
					if(users_with_single_rating.find(i)!=users_with_single_rating.end())
					{
						z[i] = getSigma_i(M_trainC,M_trainR,U,V,margins,W[i],i,L);
						getSigmaInv(z[i],sig_inv[i],L);
						continue;
					}
					// obtain a sigma for each row
					z[i] = getSigma_i(M_trainC,M_trainR,U,V,margins,W[i],i,L);

					// compute corresponding sigma inverse into a linear spline
					getSigmaInv(z[i],sig_inv[i],L);

					// update sigma(M_train) and sigma(M_test)
					for(SpMatR::InnerIterator it(M_R,i); it; ++it)
						for(int rating=1;rating<=L;rating++)
							if(M_trainR.coeffRef(i,it.col())==rating)
							{
								it.valueRef() = z[i][rating-1];
								M_C.coeffRef(it.row(),it.col()) = z[i][rating-1];
							}
					for(SpMatR::InnerIterator it(M_test_R,i); it; ++it)
						for(int rating=1;rating<=L;rating++)
							if(M_testR.coeffRef(i,it.col())==rating)
							{
								it.valueRef() = z[i][rating-1];
								M_test_C.coeffRef(it.row(),it.col()) = z[i][rating-1];
							}
				}
				// calculating train and test loss after each iteration and appending to the loss series
				train_loss = loss(M_R,U,V,train_ij,l1,l2);
				test_loss = testlossN(M_testR,U,V,test_ij,sig_inv,users_with_single_rating);
				LTrain.push_back(train_loss);
				LTest.push_back(test_loss);

				// break if moving average of last 5 Test Losses cross a data dependent threshold - tol
				if(t==5)
				{
					last_few = LTest[1]+LTest[2]+LTest[3]+LTest[4]+LTest[5];
					second_last_few = LTest[0]+LTest[1]+LTest[2]+LTest[3]+LTest[4];
				}
				else if(t>5)
				{
					last_few += (LTest[t] - LTest[t-5]);
					second_last_few += (LTest[t-1] - LTest[t-6]);
					if(last_few>second_last_few || fabs(last_few - second_last_few) < (tol*5)/test )
						break;
				}
				// break if consecutive train loss difference falls below a data size dependent threshold - tol
				// if(t!=0 && (LTrain[t-1]-LTrain[t])<1)
				// 	break;
				cout<<float(clock () - begin_time )/CLOCKS_PER_SEC<<endl;
			}
			cout<<endl;

			// writing results to files
			out<<l1<<"\t"<<l2<<endl;
			out<<LTrain.back()<<"\t"<<LTest.back()<<endl;
			L12(l1i,l2j) = LTest.back();
			// KTvalue(l1i,l2j) = KT(M_testC,M_testR,U,V,"kt_N.txt");
			// cout<<KTvalue(l1i,l2j)<<endl;
			// cout<<endl;
			cout<<endl<<"Test MAE = "<<MAE_N(M_testR,U,V,test_ij,sig_inv,users_with_single_rating);
			cout<<endl;
			// pair<double,double> ndcg_stat = NDCG_Matrix_Nsigma(M_testC,M_testR,U,V,"ndcg_No.txt",sig_inv,users_with_single_rating,ndcg_thresh,10);
			// NDCGvalue10(l1i,l2j) = ndcg_stat.first;
			// NDCGmed10(l1i,l2j) = ndcg_stat.second;

			// ndcg_stat = NDCG_Matrix_Nsigma(M_testC,M_testR,U,V,"ndcg_No.txt",sig_inv,users_with_single_rating,ndcg_thresh,20);
			// NDCGvalue20(l1i,l2j) = ndcg_stat.first;
			// NDCGmed20(l1i,l2j) = ndcg_stat.second;

			// ndcg_stat = NDCG_Matrix_Nsigma(M_testC,M_testR,U,V,"ndcg_No.txt",sig_inv,users_with_single_rating,ndcg_thresh,30);
			// NDCGvalue30(l1i,l2j) = ndcg_stat.first;
			// NDCGmed30(l1i,l2j) = ndcg_stat.second;

			// cout<<endl<<NDCGvalue10(l1i,l2j)<<" "<<NDCGmed10(l1i,l2j)<<endl;
			// cout<<NDCGvalue20(l1i,l2j)<<" "<<NDCGmed20(l1i,l2j)<<endl;
			// cout<<NDCGvalue30(l1i,l2j)<<" "<<NDCGmed30(l1i,l2j)<<endl;
			// cout<<endl;

			// dumping U, V, Loss vectors, and KT to files
			// dumpMatrixToFile(dir_l12+"U.txt",U);
			// dumpMatrixToFile(dir_l12+"V.txt",V);
			ofstream datafile(dir_l12+"losses.txt");
			for(int i=0;i<LTrain.size();i++)
				datafile<<LTrain[i]<<" "<<LTest[i]<<endl;
			datafile.close();
			datafile.clear();
			// datafile.open(dir_l12+"sigma.txt");
			// for(int i=0;i<n;i++)
			// {
			// 	for(int j=0;j<5;j++)
			// 		datafile<<z[i][j]<<" ";
			// 	datafile<<endl;
			// }
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
	out<<"LT"<<endl<<L12<<endl;
	// out<<"KT"<<endl<<KTvalue<<endl;
	cout<<"LT"<<endl<<L12<<endl;
	// cout<<"KT"<<endl<<KTvalue<<endl;
	// cout<<"NDCG_mean NDCG_median"<<endl;
	// cout<<NDCGvalue10<<" "<<NDCGmed10<<endl;
	// cout<<NDCGvalue20<<" "<<NDCGmed20<<endl;
	// cout<<NDCGvalue30<<" "<<NDCGmed30<<endl;
	out<<"-----------------------\n";
	// N sigma algorithm completed, note that the latest U, V and z will be used in the further steps

	/* Initializing the grid axis vectors of regularization hyperparameters
	 * To use optimal parameters set them to
	 * data[dataset].sigma_K.Kmap[k].KT.l1 and data[dataset].sigma_K.Kmap[k].KT.l2 for best Kendal Tau
	 * data[dataset].sigma_K.Kmap[k].TL.l1 and data[dataset].sigma_K.Kmap[k].TL.l2 for best Test Loss
	 */
	// vector<double> L3 = {data[dataset].sigma_K.Kmap[k].TL.l1};
	// vector<double> L4 = {data[dataset].sigma_K.Kmap[k].TL.l2};

	vector<double> L3 = {2.8};
	vector<double> L4 = {4};

	MatrixXd L34(L3.size(),L4.size());
	MatrixXd KTvalue2(L3.size(),L4.size());
	MatrixXd NDCGvalue2_10(L3.size(),L4.size());
	MatrixXd NDCGmed2_10(L3.size(),L4.size());
	MatrixXd NDCGvalue2_20(L3.size(),L4.size());
	MatrixXd NDCGmed2_20(L3.size(),L4.size());
	MatrixXd NDCGvalue2_30(L3.size(),L4.size());
	MatrixXd NDCGmed2_30(L3.size(),L4.size());

	// temporarily storing U and V for reusability
	MatrixXd U_temp = U, V_temp = V;

	// histogram of non zero ratings in each cluster, to be used to resize M_expanded_cluster[k]
	int N_total_train[K], N_total_test[K];

	// vector of cluster numbers for each user;
	VectorXi cluster(n);

	// iteration over grid points
	l1i = 0;
	l2j = 0;
	for(auto l1:L3)
	{
		for(auto l2:L4)
		{
			// getting back the temporarily stored U and V for each grid point
			U = U_temp;
			V = V_temp;

			// creating directories for every grid point
			string dir_l12 = directory+"K_"+tostring(l1)+"_"+tostring(l2)+"/"+tostring(K)+"/";
			string cmd = "mkdir -p "+dir_l12;
			int dir_created = system(cmd.c_str());

			cout<<l1<<" "<<l2<<endl;

			// k means clustering step
			MatrixXd Z = MatrixXd::Zero(n,L);
			CMatToEigenMat(z,Z);
			MatrixXd Z_centers = MatrixXd::Zero(K,L);
			
			// performing k means
			kmeans(Z,Z_centers,cluster,users_with_single_rating);
			EigenMatToCMat(Z_centers,z_cluster);

			// getting histogram of clusters
			cluster_hist(cluster,cluster_count,K);
			
			// variables used for early stopping
			vector<double> LTest, LTrain;
			
			// looping till early stop
			int changes;
			for(int t=0;1;t++)
			{
				// assigning clusters to each row and getting the number of cluster changes
				changes = assign_sigma(M_trainR, U, V, z_cluster, cluster,K);
				cout<<"Cluster changes = "<<changes<<endl;

				// updating histogram of cluster counts
				cluster_hist(cluster,cluster_count,K);
				// cout<<"Cluster count : ";
				// copy(cluster_count,cluster_count+K,ostream_iterator<int>(cout," "));
				// cout<<endl;
				train_loss = loss(M_R,U,V,train_ij,l1,l2);
				cout<<"Assign clusters  : ";
				cout<<train_loss<<endl;

				// updating weight vector for each cluster for pooled PAV
				update_W(M_trainR,M_testR,W_cluster,N_total_train,N_total_test,cluster,K,users_with_single_rating,L);
				for(int k=0;k<K;k++)
				{
					// skip empty clusters if any
					if(cluster_count[k]==0)
						continue;
					M_expanded_cluster[k].resize(N_total_train[k]);
					M_test_expanded_cluster[k].resize(N_total_test[k]);
					// concatenate the non zero rating of each cluster into M_expanded_cluster[k] and M_test_expanded_cluster[k]
					concatM(M_trainC, M_trainR, M_expanded_cluster[k], M_testC, M_testR, M_test_expanded_cluster[k],cluster,k,pair_size);
					int cluster_train_size = pair_size.first, cluster_test_size = pair_size.second;
					VectorXd M_expanded_cluster_k = M_expanded_cluster[k].head(cluster_train_size);

					// Pooled PAV for each cluster
					z_cluster[k] = get_1_sigma(M_trainC,M_trainR,M_expanded_cluster_k,U,V,margins,W_cluster[k],cluster,k,L);
				}

				// update sigma_k(M_train) and sigma_k(M_test)
				for(int i=0;i<n;i++)
				{
					if(cluster(i)<0 || cluster(i)>=K)
						continue;
					for(SpMatR::InnerIterator it(M_R,i); it; ++it)
						for(int rating=1;rating<=L;rating++)
							if(M_trainR.coeffRef(i,it.col())==rating)
							{
								it.valueRef() = z_cluster[cluster(i)][rating-1];
								M_C.coeffRef(it.row(),it.col()) = z_cluster[cluster(i)][rating-1];
							}
					for(SpMatR::InnerIterator it(M_test_R,i); it; ++it)
						for(int rating=1;rating<=L;rating++)
							if(M_testR.coeffRef(i,it.col())==rating)
							{
								it.valueRef() = z_cluster[cluster(i)][rating-1];
								M_test_C.coeffRef(it.row(),it.col()) = z_cluster[cluster(i)][rating-1];
							}
				}

				// get sigma inverse linear spline for each non empty cluster
				for(int k=0;k<K;k++)
				{	
					if(cluster_count[k]==0)
						continue;
					getSigmaInv(z_cluster[k],sig_inv_cluster[k],L);
				}

				// Losses after Pooled PAV
				cout<<"After pooled PAV : ";
				train_loss = loss(M_R,U,V,train_ij,l1,l2);
				test_loss = testlossK(M_testR,U,V,test_ij,sig_inv_cluster,cluster,K);
				cout<<train_loss<<"\t"<<test_loss<<endl;

				// U, V updates after Pooled PAV
				Ustep(M_C,M_R,U,V,l1);
				Vstep(M_C,M_R,U,V,l2);

				// Losses after U, V updates
				cout<<t<<" After UV updates : ";
				train_loss = loss(M_R,U,V,train_ij,l1,l2);
				test_loss = testlossK(M_testR,U,V,test_ij,sig_inv_cluster,cluster,K);
				cout<<train_loss<<"\t"<<test_loss<<endl;

				// storing train and test loss in corresponding Loss vectors
				LTrain.push_back(train_loss);
				LTest.push_back(test_loss);
				// break if consecutive test loss difference falls below data dependent threshold - tol
				if(t!=0 && ( (LTest[t] > LTest[t-1]) || fabs(LTest[t] - LTest[t-1])<(tol/test) ) )
					break;
			}

			// writing results to files
			out<<l1<<"\t"<<l2<<endl;
			out<<LTrain.back()<<"\t"<<LTest.back()<<endl;
			L34(l1i,l2j) = LTest.back();
			// KTvalue2(l1i,l2j) = KT(M_testC,M_testR,U,V,"kt_k.txt");
			// cout<<KTvalue2(l1i,l2j)<<endl;
			cout<<"Test MAE = "<<MAE_K(M_testR,U,V,test_ij,sig_inv_cluster,cluster,K);
			cout<<endl;
			// pair<double,double> ndcg_stat2 = NDCG_Matrix_Ksigma(M_testC,M_testR,U,V,"ndcg_ksigma_10.txt",sig_inv_cluster,cluster,K,ndcg_thresh,10);
			// NDCGvalue2_10(l1i,l2j) = ndcg_stat2.first;
			// NDCGmed2_10(l1i,l2j) = ndcg_stat2.second;
			
			// ndcg_stat2 = NDCG_Matrix_Ksigma(M_testC,M_testR,U,V,"ndcg_ksigma_20.txt",sig_inv_cluster,cluster,K,ndcg_thresh,20);
			// NDCGvalue2_20(l1i,l2j) = ndcg_stat2.first;
			// NDCGmed2_20(l1i,l2j) = ndcg_stat2.second;

			// ndcg_stat2 = NDCG_Matrix_Ksigma(M_testC,M_testR,U,V,"ndcg_ksigma_30.txt",sig_inv_cluster,cluster,K,ndcg_thresh,30);
			// NDCGvalue2_30(l1i,l2j) = ndcg_stat2.first;
			// NDCGmed2_30(l1i,l2j) = ndcg_stat2.second;

			// cout<<endl<<NDCGvalue2_10(l1i,l2j)<<" "<<NDCGmed2_10(l1i,l2j)<<endl;
			// cout<<NDCGvalue2_20(l1i,l2j)<<" "<<NDCGmed2_20(l1i,l2j)<<endl;
			// cout<<NDCGvalue2_30(l1i,l2j)<<" "<<NDCGmed2_30(l1i,l2j)<<endl;
			// cout<<endl;

			// dumping U, V, Loss vectors, and KT to files
			// dumpMatrixToFile(dir_l12+"U.txt",U);
			// dumpMatrixToFile(dir_l12+"V.txt",V);
			ofstream datafile(dir_l12+"losses.txt");
			for(int i=0;i<LTrain.size();i++)
				datafile<<LTrain[i]<<" "<<LTest[i]<<endl;
			datafile.close();
			datafile.clear();
			// datafile.open(dir_l12+"KT.txt");
			// datafile<<KTvalue2(l1i,l2j);
			datafile.close();

			l2j++;
		}
		l1i++;
		l2j = 0;
	}

	// writing and printing grid results to file and terminal respectively
	out<<"LT"<<endl<<L34<<endl;
	// out<<"KT"<<endl<<KTvalue2<<endl;
	cout<<"LT"<<endl<<L34<<endl;
	// cout<<"KT"<<endl<<KTvalue2<<endl;
	// cout<<"NDCG_mean NDCG_median"<<endl;
	// cout<<NDCGvalue2_10<<" "<<NDCGmed2_10<<endl;
	// cout<<NDCGvalue2_20<<" "<<NDCGmed2_20<<endl;
	// cout<<NDCGvalue2_30<<" "<<NDCGmed2_30<<endl;
	out.close();


	ofstream lossfile("rowtrainloss.txt");
	double rloss;
	int count = 0;
	for(i=0;i<n;i++)
	{
		rloss = row_loss(M_trainR,U,V,z_cluster,i,cluster(i),count);
		lossfile<<i<<" "<<(count>0?(rloss/count):0)<<" "<<count<<endl;
	}
	lossfile.close();

	user_loss(M_testR,U,V,sig_inv_cluster,cluster);

	ofstream sgma("sigma_k_sigma.txt");
	for(int i=0;i<n;i++)
	{
		sgma<<i<<" ";
		if(users_with_single_rating.find(i)!=users_with_single_rating.end())
		{
			for(int j=0;j<L;j++)
				sgma<<j+1<<" ";
			sgma<<endl;
			continue;
		}
		for(int j=0;j<L;j++)
			sgma<<z_cluster[cluster(i)][j]<<" ";
		sgma<<endl;
	}
	sgma.close();
	sgma.clear();

	// playing a sound when program end :) #Note - only for linux OS
	// string sound = "paplay /usr/share/sounds/freedesktop/stereo/complete.oga";
	// int beep = system(sound.c_str());
	return 0;
}