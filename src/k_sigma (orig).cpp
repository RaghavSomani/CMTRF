#include "init.h"
#include "sigma.cpp"
#include "km.cpp"

int main()
{
	// selecting dataset
	cout<<"0. movielens_small"<<endl;
	cout<<"1. movielens_medium"<<endl;
	cout<<"Dataset : ";
	int dataset;
	cin>>dataset;
	int i,j,r,n,p,split,k=0;
	ifstream file;
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

	// getting all the optimal parametsrs into data object array stored in a seperate file params.txt
	int ND;
	ifstream param_file("params.txt");
	param_file>>ND;
	Dataset data[ND];
	get_opt_params(param_file,data,ND);

	// set if (i,j) indexes each for train and test set, known_ij is the union of both
	set<pair<int,int> > known_ij, train_ij, test_ij;
	int known = 0, test = 0, train;

	vector<T> tripletList_train;
	vector<T> tripletList_test;

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

	// declaring and initializing sparse matrices both in Column Major and Row Major form
	SpMatC M_trainC(n,p);
	SpMatR M_trainR(n,p);
	SpMatC M_testC(n,p);
	SpMatR M_testR(n,p);
	M_trainC.setFromTriplets(tripletList_train.begin(), tripletList_train.end());
	M_trainR.setFromTriplets(tripletList_train.begin(), tripletList_train.end());
	M_testC.setFromTriplets(tripletList_test.begin(), tripletList_test.end());
	M_testR.setFromTriplets(tripletList_test.begin(), tripletList_test.end());

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
	vector<double> L1 = {data[dataset].sigma_N.Kmap[k].TL.l1};
	vector<double> L2 = {data[dataset].sigma_N.Kmap[k].TL.l2};
	MatrixXd L12(L1.size(),L2.size());
	MatrixXd KTvalue(L1.size(),L2.size());

	// declaring variables to be used further down the program
	MatrixXd U(n,k), V(p,k);
	pair<MatrixXd,MatrixXd>* UV;
	int sz, Msz, dir_created;
	double train_loss, test_loss, tol;
	if(dataset==0)
		tol = 1;
	else if(dataset==1)
		tol = 10;

	// Asking the user for the number of clusters if optimal not there in params.txt
	int K;
	int opt_K = data[dataset].sigma_K.Kmap[k].TL.K;
	if(opt_K==-1)
	{
		cout<<"Number of clusters = ";
		cin>>K;
	}
	else
	{
		K = opt_K;
		cout<<"Number of clusters = "<<K<<endl;
	}

	// creating experiment directory
	string directory;
	create_directory(data,dataset,k,"k_sigma_/",directory);
	ofstream out(directory+"results.txt");

	// Keep the frequency matrix, N[i][r-1] = Number of ratings r in M_train[i]
	int N[n][5];
	memset(N, 0, sizeof(N[0][0])*n*5);
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
	double W[n][5];
	for(int i=0;i<n;i++)
		transform(N[i], N[i]+5, W[i],bind2nd(divides<double>(),(double)N_total[i]));

	// z will store the sigma function array for each row of M_train
	// defining margins such that z[i+1] - z[i] >= margin[i] for i in {0,1,2,3}
	double margins[4] = {eps,eps,eps,eps};
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
		W_cluster[k] = new double[5];

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
			UV = initialize_UV(n,p,k);
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
					// obtain a sigma for each row
					z[i] = getSigma_i(M_trainC,M_trainR,U,V,margins,W[i],i);

					// compute corresponding sigma inverse intoa linear spline
					getSigmaInv(z[i],sig_inv[i]);

					// update sigma(M_train) and sigma(M_test)
					for(SpMatR::InnerIterator it(M_R,i); it; ++it)
						for(int rating=1;rating<=5;rating++)
							if(M_trainR.coeffRef(i,it.col())==rating)
							{
								it.valueRef() = z[i][rating-1];
								M_C.coeffRef(it.row(),it.col()) = z[i][rating-1];
							}
					for(SpMatR::InnerIterator it(M_test_R,i); it; ++it)
						for(int rating=1;rating<=5;rating++)
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
			KTvalue(l1i,l2j) = KT(M_testC,M_testR,U,V);
			cout<<KTvalue(l1i,l2j)<<endl;

			// dumping U, V, Loss vectors, and KT to files
			dumpMatrixToFile(dir_l12+"U.txt",U);
			dumpMatrixToFile(dir_l12+"V.txt",V);
			ofstream datafile(dir_l12+"losses.txt");
			for(int i=0;i<LTrain.size();i++)
				datafile<<LTrain[i]<<" "<<LTest[i]<<endl;
			datafile.close();
			datafile.clear();
			datafile.open(dir_l12+"sigma.txt");
			for(int i=0;i<n;i++)
			{
				for(int j=0;j<5;j++)
					datafile<<z[i][j]<<" ";
				datafile<<endl;
			}
			datafile.clear();
			datafile.open(dir_l12+"KT.txt");
			datafile<<KTvalue(l1i,l2j);
			datafile.close();

			l2j++;
		}
		l1i++;
		l2j = 0;
	}

	// writing and printing grid results to file and terminal respectively
	out<<"LT"<<endl<<L12<<endl;
	out<<"KT"<<endl<<KTvalue<<endl;
	cout<<"LT"<<endl<<L12<<endl;
	cout<<"KT"<<endl<<KTvalue<<endl;
	out<<"-----------------------\n";
	// N sigma algorithm completed, note that the latest U, V and z will be used in the further steps

	/* Initializing the grid axis vectors of regularization hyperparameters
	 * To use optimal parameters set them to
	 * data[dataset].sigma_K.Kmap[k].KT.l1 and data[dataset].sigma_K.Kmap[k].KT.l2 for best Kendal Tau
	 * data[dataset].sigma_K.Kmap[k].TL.l1 and data[dataset].sigma_K.Kmap[k].TL.l2 for best Test Loss
	 */
	vector<double> L3 = {data[dataset].sigma_K.Kmap[k].TL.l1};
	vector<double> L4 = {data[dataset].sigma_K.Kmap[k].TL.l2};
	MatrixXd L34(L3.size(),L4.size());
	MatrixXd KTvalue2(L3.size(),L4.size());

	// temporarily storing U and V for reusability
	MatrixXd U_temp = U, V_temp = V;

	// histogram of non zero ratings in each cluster, to be used to resize M_expanded_cluster[k]
	int N_total_train[K], N_total_test[K];

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
			MatrixXd Z = MatrixXd::Zero(n,5);
			CMatToEigenMat(z,Z);
			MatrixXd Z_centers = MatrixXd::Zero(K,5);
			VectorXi cluster(n);
			
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
				update_W(M_trainR,M_testR,W_cluster,N_total_train,N_total_test,cluster,K,users_with_single_rating);
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
					z_cluster[k] = get_1_sigma(M_trainC,M_trainR,M_expanded_cluster_k,U,V,margins,W_cluster[k],cluster,k);
				}

				// update sigma_k(M_train) and sigma_k(M_test)
				for(int i=0;i<n;i++)
				{
					if(cluster(i)<0 || cluster(i)>=K)
						continue;
					for(SpMatR::InnerIterator it(M_R,i); it; ++it)
						for(int rating=1;rating<=5;rating++)
							if(M_trainR.coeffRef(i,it.col())==rating)
							{
								it.valueRef() = z_cluster[cluster(i)][rating-1];
								M_C.coeffRef(it.row(),it.col()) = z_cluster[cluster(i)][rating-1];
							}
					for(SpMatR::InnerIterator it(M_test_R,i); it; ++it)
						for(int rating=1;rating<=5;rating++)
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
					getSigmaInv(z_cluster[k],sig_inv_cluster[k]);
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
				cout<<"After UV updates : ";
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
			KTvalue2(l1i,l2j) = KT(M_testC,M_testR,U,V);
			cout<<KTvalue2(l1i,l2j)<<endl;
			
			// dumping U, V, Loss vectors, and KT to files
			dumpMatrixToFile(dir_l12+"U.txt",U);
			dumpMatrixToFile(dir_l12+"V.txt",V);
			ofstream datafile(dir_l12+"losses.txt");
			for(int i=0;i<LTrain.size();i++)
				datafile<<LTrain[i]<<" "<<LTest[i]<<endl;
			datafile.close();
			datafile.clear();
			datafile.open(dir_l12+"KT.txt");
			datafile<<KTvalue2(l1i,l2j);
			datafile.close();

			l2j++;
		}
		l1i++;
		l2j = 0;
	}

	// writing and printing grid results to file and terminal respectively
	out<<"LT"<<endl<<L34<<endl;
	out<<"KT"<<endl<<KTvalue2<<endl;
	cout<<"LT"<<endl<<L34<<endl;
	cout<<"KT"<<endl<<KTvalue2<<endl;
	out.close();

	// playing a sound when program end :) #Note - only for linux OS
	string sound = "paplay /usr/share/sounds/freedesktop/stereo/complete.oga";
	int beep = system(sound.c_str());
	return 0;
}