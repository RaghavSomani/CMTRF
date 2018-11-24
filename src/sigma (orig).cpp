/* A function to get the optimal parameters kept in a file.
 * There are assumptions in the format of the file.
 * The ifstream object must point to this file, having read the first line as ND,
 * that stores the number of datasets to follow so that the dataset array of size ND can be filled.
 * The first line after the beginning of a dataset chunk is the name of the dataset
 * 		The next line is the number of low ranks, followed by chunks describing each low rank
 * 			Each low rank chunk has 8 lines,
 * 			The first 2 lines are the optimal <l1><space><l2> for best KT and TL respectively for No sigma
 * 			The next 2 lines are the optimal <l1><space><l2> for best KT and TL respectively for 1 sigma
 * 			The next 2 lines are the optimal <l1><space><l2> for best KT and TL respectively for N sigma
 * 			The next 2 lines are the optimal <l1><space><l2><space><clusters> for best KT and TL respectively for K sigma
 * Refer params.txt for the format.
 */
void get_opt_params(ifstream& file, Dataset data[], int ND)
{
	low_rank lr;
	int clusters, rank, Nk;
	double L1, L2;
	for(int d=0;d<ND;d++)
	{
		file>>data[d].name;
		file>>Nk;
		for(int nk=0;nk<Nk;nk++)
		{
			file>>rank;
			file>>L1>>L2;
			lr.KT.set_params(L1,L2);
			file>>L1>>L2;
			lr.TL.set_params(L1,L2);
			data[d].No_sigma.Kmap[rank] = lr;

			file>>L1>>L2;
			lr.KT.set_params(L1,L2);
			file>>L1>>L2;
			lr.TL.set_params(L1,L2);
			data[d].sigma_1.Kmap[rank] = lr;

			file>>L1>>L2;
			lr.KT.set_params(L1,L2);
			file>>L1>>L2;
			lr.TL.set_params(L1,L2);
			data[d].sigma_N.Kmap[rank] = lr;

			file>>L1>>L2>>clusters;
			lr.KT.set_params(L1,L2,clusters);
			file>>L1>>L2>>clusters;
			lr.TL.set_params(L1,L2,clusters);
			data[d].sigma_K.Kmap[rank] = lr;
		}
	}
}

/* Takes in a double value
 * Returns a string using stringstream conversion
 */
string tostring(double val)
{
	ostringstream convert;
	convert<<val;
	return convert.str();
}

/* Creates directory for experiments
 * Takes inputs
 * dataset = dataset ID
 * 				1 for movielens small
 * 				2 for movielens medium
 * algo = The algorithm used
 * 			No_sigma_/ for w/o sigma algorithm
 * 			1_sigma_/ for 1-sigma algorithm
 * 			N_sigma_/ for N-sigma algorithm
 * 			k_sigma_/ for k-sigma algorithm
 * directory = the return referenced path of the directory created
 */																																										
void create_directory(Dataset data[], int dataset, int k, const string& algo, string& directory)
{
	string parent_dir = "../data/";
	// if(dataset==1)
		parent_dir = parent_dir+data[dataset].name+"/";
	// else if(dataset==2)
		// parent_dir = parent_dir+"movielens_medium/";
	string K = "k"+tostring(k)+"/";
	directory = parent_dir+K+algo;
	string cmd = "mkdir -p "+directory;
	int dir_created = system(cmd.c_str());
}

/* Initializes U (n x k) and V (p x k) low rank factor matrices such that
 * M (n x p) = U.V^T is initialized with iid random variables with mean m = 3
 * and standard deviation s = 1.
 * The values in U and V are ~ N(mu,sigma)
 * mu = sqrt(/k)
 * sigma = sqrt( sqrt(m^4 + s/k) - mu^2 )
 * Input arguments are assumed to be positive integers
 * n = number of rows of M
 * p = number of colums of M
 * k = number of columns of U and V
 * Returns a pointer to a pair of Matrices, U and V.
 */
pair<MatrixXd,MatrixXd>* initialize_UV(int n, int p, int k)
{
	// Mean M and standard deviation set to 1
	double M = 3.0, S = 1.0;
	pair<MatrixXd,MatrixXd>* UV = new pair<MatrixXd,MatrixXd>;
	MatrixXd* U = new MatrixXd(n,k);
	MatrixXd* V = new MatrixXd(p,k);

	// the mean and variance of elements of U and V are set
	double mu = sqrt(M/k);
	double sigma = sqrt( sqrt(pow(mu,4)+S/k) -mu*mu);
	default_random_engine generator;
	normal_distribution<double> distribution(mu,sigma);
	
	// filling in U and V with iid N(mu,sigma)
	for(int i = 0; i < U->size(); i++)
		*(U->data() + i) = distribution(generator);
	for(int i = 0; i < V->size(); i++)
		*(V->data() + i) = distribution(generator);
	
	// returing a pair pointer to matrices U and V
	UV->first = *U;
	UV->second = *V;
	return UV;
}

/* Computes the train loss as
 * (1/|set_ij|)*sum_(set_ij){ (M_ij - <U_i,V_j>)^2 } + l1*||U||_F^2 + l2*||V||_F^2
 * Takes inputs
 * M (nxp) = training rating matrix
 * U (nxk), V (pxk) = Factor matrices
 * set_ij = set of pairs of integer index values (i,j) over which the loss has to be computed.
 * 		Assuming i is in {0,1,...,n-1} and j is in {0,1,...,p-1}
 * l1, l2 = non negative regularization hyperparameters for U and V respectively
 * 		# Note - For computing test loss, set l1 and l2 to 0.
 * Outputs loss
 */
double loss(SpMatR& M, MatrixXd& U, MatrixXd& V,set<pair<int,int> >& set_ij,double l1,double l2)
{
	double reg_loss = l1*pow(U.norm(),2) + l2*pow(V.norm(),2), temp, loss = 0, total_loss = 0;
	int i,j;
	for(auto ij:set_ij)
	{
		i = ij.first;
		j = ij.second;
		temp = M.coeffRef(i,j) - U.row(i).dot(V.row(j));
		loss += temp*temp;
	}
	total_loss = loss + reg_loss;
	return total_loss;
}

/* Computes 1_sigma test loss as
 * (1/|set_ij|)*sum_(set_ij){ ( M_ij - sigma_inv(<U_i,V_j>) )^2 }
 * Takes inputs
 * M (nxp) = training rating matrix
 * U (nxk), V (pxk) = Factor matrices
 * set_ij = set of pairs if integer index values (i,j) over which the loss has to be computed.
 * 		Assuming i is in {0,1,...,n-1} and j is in {0,1,...,p-1}
 * sig_inv = spline object as sigma_inv
 * 		Assuming - the points of sig_inv are set to increasing x values, and y values
 * 				 - there are atleast 3 points set in the spline
 */
double testloss(SpMatR& M, MatrixXd& U, MatrixXd& V,set<pair<int,int> >& set_ij,tk::spline& sig_inv)
{
	double loss = 0, temp;
	int i,j;
	for(auto ij:set_ij)
	{
		i = ij.first;
		j = ij.second;
		temp = M.coeffRef(i,j) - sig_inv(U.row(i).dot(V.row(j)));
		loss += temp*temp;
	}
	return loss/set_ij.size();
}

/* Computes 1_sigma test loss as
 * (1/|set_ij|)*sum_(set_ij){ ( M_ij - sigma_inv_i(<U_i,V_j>) )^2 }
 * Takes inputs
 * M (nxp) = training rating matrix
 * U (nxk), V (pxk) = Factor matrices
 * set_ij = set of pairs if integer index values (i,j) over which the loss has to be computed.
 * 		Assuming i is in {0,1,...,n-1} and j is in {0,1,...,p-1}
 * sig_inv = array spline object as sigma_inv of length n
 * 		Assuming - the points of sig_inv[i] are set to increasing x values, and y values for every i
 * 				 - there are atleast 3 points set in every spline
 * users_with_single_rating = set of integer indexes, not to include in computing loss
 * 		Assuming the set has integers in {0,1,...,n-1}
 */
double testlossN(SpMatR& M, MatrixXd& U, MatrixXd& V,set<pair<int,int> >& set_ij,tk::spline sig_inv[],set<int> users_with_single_rating)
{
	double loss = 0, temp;
	int i,j;
	for(auto ij:set_ij)
	{
		i = ij.first;
		j = ij.second;
		if(users_with_single_rating.find(i)!=users_with_single_rating.end())
			continue;
		temp = M.coeffRef(i,j) - sig_inv[i](U.row(i).dot(V.row(j)));
		loss += temp*temp;
	}
	return loss/set_ij.size();
}

/* Computes k_sigma test loss as
 * (1/|set_ij|)*sum_(set_ij){ ( M_ij - sigma_inv_k(<U_i,V_j>) )^2 }
 * Takes inputs
 * M (nxp) = training rating matrix
 * U (nxk), V (pxk) = Factor matrices
 * set_ij = set of pairs if integer index values (i,j) over which the loss has to be computed.
 * 		Assuming i is in {0,1,...,n-1} and j is in {0,1,...,p-1}
 * sig_inv = array spline object as sigma_inv of length k
 * 		Assuming - the points of sig_inv[i] are set to increasing x values, and y values for every i
 * 				 - there are atleast 3 points set in every spline
 * cluster = Vector of length n containing integers in {-1,0,1,...,k-1} representing user i belongs to cluster(i)
 * 		CLuster -1 beongs to the users who are not there in
 * 		any of the k clusters, and for which the loss is not to be computed.
 * K = number of clusters
 * 		Assuming K to be a positive integer
 * 
 */
double testlossK(SpMatR& M, MatrixXd& U, MatrixXd& V,set<pair<int,int> >& set_ij,tk::spline sig_inv[], VectorXi& cluster, int K)
{
	double loss = 0, temp;
	int i,j;
	for(auto ij:set_ij)
	{
		i = ij.first;
		j = ij.second;
		if(cluster(i)<0 || cluster(i)>=K)
			continue;
		temp = M.coeffRef(i,j) - sig_inv[cluster(i)](U.row(i).dot(V.row(j)));
		loss += temp*temp;
	}
	return loss/set_ij.size();
}

/* Mask non-zeros from each row/column (1/0) of M and fill Xsub with masked rows/columns of X
 * This is a function created to do subsetting of a matrix X
 * using some other matrix M based on non zero values of M.
 * Takes inputs
 * MC and MR = Sparse matrices containing non zero values in column and row major forms respectively
 * X = Matrix from which subsetting has to be done
 * axis = dimension along which subsetting has to be done
 * 		1 - rows
 * 		0 - columns
 * Xsub = a pointer to the containeer matrix which will contain the subset rows.
 * 		Assuming it to be allocated a space of (n x X.cols()), n >= X.rows()
 * Eg :- subset(M,V,1,i,&Vsub)
 * 			M is (nxp), V is (pxk) and Vsub is (>=p,k)
 * 			This will fill Vsub matrix with rows V(j) if M(i,j)!=0 maintaining order
 * Returns the number of rows filled in so that the subset can be accessed by Xsub->topRows(return_value)
 */
int subset(SpMatC& MC, SpMatR& MR, MatrixXd& X, int axis, int index, MatrixXd* Xsub)
{
	Array<bool,Dynamic,1> mask;
	VectorXd temp;
	if(axis==0)
	{
		if(MC.rows()!=X.rows())
			return 0;
		mask.resize(MC.rows(),1);
		temp = MC.col(index);
		mask = (temp.array()!=0);
	}
	else if(axis==1)
	{
		if(MR.cols()!=X.rows())
			return 0;
		mask.resize(MR.cols(),1);
		temp = MR.row(index);
		mask = (temp.array()!=0);	
	}
	for(int i=0,j=0;i<mask.size();++i)
		if(mask(i))
			Xsub->row(j++) = X.row(i);
	return mask.count();
}

/* This is a function created to do subsetting of a matrix M based
 * on the non zero values of matrix M_mask and filling it to the vector pointed by Msub contiguously
 * Takes inputs
 * MC and MR = Sparse Matrix from which values have to be taken, both in column and row major form respectively
 * M_maskC and M_maskR = Sparse Matrices from where corresponding non zeros values have to be observed,
 * 							both in column and row major form respectively
 * axis = dimension along which subsetting has to be done continguosly
 * 		1 - rows
 * 		0 - columns
 * index = row of column index (depending on axis = 1 or 0) of M_mask
 * Msub = pointer to a vector in which the values of M are filled.
 * 		Assuming the pointer is allocated memory of size atleast
 * 		equal to the number of non-zero values of M_mask.row(index)/M_mask.col(index) depending on axis = 1/0
 * Eg :- subvec(M,M,1,i,&MsubV)
 * 			M is (nxp) and MsubV is of size atleast number of non-zero values of M_mask.row(index)
 * 			This will fill in MsubV with elements of M.row(index) such that M_mask.row(index)
 * 			is non zero, in a contiguous manner.
 */
int subvec(SpMatC& MC, SpMatR& MR, SpMatC& M_maskC, SpMatR& M_maskR, int axis, int index, VectorXd* Msub)
{
	Array<bool,Dynamic,1> mask;
	VectorXd vec, temp;
	if(axis==0)
	{
		mask.resize(MC.rows());
		temp = M_maskC.col(index);
		mask = (temp.array()!=0);
		vec = MC.col(index);
	}
	else if(axis==1)
	{
		mask.resize(MR.cols());
		temp = M_maskR.row(index);
		mask = (temp.array()!=0);
		vec = MR.row(index);
	}
	for(int i=0,j=0;i<mask.size();i++)	
		if(mask(i))
			(*Msub)(j++)=vec(i);
	return mask.count();
}

/* Optimizes over U keeping V constant such that the train loss
 * (1/|known_ij|)*sum_(known_ij){ (M_ij - <U_i,V_j>)^2 } + l1*||U||_F^2 + l2*||V||_F^2
 * minimizes.
 * MC and MR - Sparse training rating matrices, both in column and row major form respectively
 */
void Ustep(SpMatC& MC, SpMatR& MR, MatrixXd& U, MatrixXd& V, double l1)
{
	int n = MC.rows(), k = U.cols(), p = MC.cols(), sz, Msz;
	MatrixXd I = MatrixXd::Identity(k,k);
	MatrixXd Vsub(p,k);
	VectorXd MsubV(p);
	for(int i=0;i<n;i++)
	{
		sz = subset(MC,MR,V,1,i,&Vsub);
		Msz = subvec(MC,MR,MC,MR,1,i,&MsubV);
		U.row(i) = (((Vsub.topRows(sz)).transpose())*(Vsub.topRows(sz))+l1*I).ldlt().solve((Vsub.topRows(sz)).transpose()*(MsubV.head(Msz)));
	}
}

/* Optimizes over V keeping U constant such that the train loss
 * (1/|known_ij|)*sum_(known_ij){ (M_ij - <U_i,V_j>)^2 } + l1*||U||_F^2 + l2*||V||_F^2
 * minimizes.
 * MC and MR - Sparse training rating matrices, both in column and row major form respectively
 */
void Vstep(SpMatC& MC, SpMatR& MR, MatrixXd& U, MatrixXd& V, double l2)
{
	int n = MC.rows(), k = U.cols(), p = MC.cols(), sz, Msz;
	MatrixXd I = MatrixXd::Identity(k,k);
	MatrixXd Usub(p,k);
	VectorXd MsubU(p);
	for(int j=0;j<p;j++)
	{
		sz = subset(MC,MR,U,0,j,&Usub);
		Msz = subvec(MC,MR,MC,MR,0,j,&MsubU);
		V.row(j) = (((Usub.topRows(sz)).transpose())*(Usub.topRows(sz))+l2*I).ldlt().solve((Usub.topRows(sz)).transpose()*(MsubU.head(Msz)));
	}
}

/* Reads a matrix from a file
 * Takes inputs
 * M (n x p) = matrix into whhich the content has to be stored
 * in = ifstream object with open writable file
 * This function is not tested properly.
 * Since this is not used yet in any of the programs, its not a worry.
 * Please check this function while of before using it.
 */
void readEigenFromFile(MatrixXd& M, ifstream& in)
{
	double item;
	if (in.is_open())
	    for (int row = 0; row < M.rows(); row++)
	        for (int col = 0; col < M.cols(); col++)
	        {
	            in>>item;
	            M(row, col) = item;
	        }
}

/* Write a Matrix to a file
 * Takes inputs
 * file = string cvariable containg the file name
 * M = Matrix to be written
 */
void dumpMatrixToFile(const string& file, MatrixXd M)
{
	ofstream datafile(file.c_str());
	datafile<<M;
	datafile.close();
}

/* Calculates the Kendall Tau-b correlation between two input vectors arr1 and arr2 of length len
 * Assuming the size of both arr1 and arr2 = len
 * For all pairs indexes (i,j), i and j in {0,1,...,len-1}
 * Refer https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient#Tau-b
 */
double KendalTau(const VectorXd& arr1, const VectorXd& arr2, int len)
{
    int m1 = 0, m2 = 0, s = 0, nPair;
    double cor ;
    for(int i=0; i < len; i++)
        for(int j=i+1;j<len;j++)
            if(arr2(i) > arr2(j))
            {
                if (arr1(i) > arr1(j))
                    s++;
                else if(arr1(i) < arr1(j))
                    s--;
                else
                    m1++;
            }
            else if(arr2(i) < arr2(j))
            {
                if (arr1(i) > arr1(j))
                    s--;
                else if(arr1(i) < arr1(j))
                    s++;
                else
                    m1++;
            }
            else
            {
                m2++;
                if(arr1(i) == arr1(j))
                    m1++;
            }
    nPair = len*(len - 1)/2;
    if(m1 < nPair && m2 < nPair)
		cor = s / ( sqrtf((double)(nPair-m1)) * sqrtf((double)(nPair-m2)) );
    else
		cor = 0.0;
    return cor ;
}

/* Calculates the average Kendall Tau-b correlation values between corresponding row vectors of
 * M_test(C/R for Column/Row major forms) and
 * M_pred (calculated, and not stored, element wise based on non-zero elements of M_test <U_i,V_j>).
 * Assumes M_test(C/R) (n x p) with U (n x k) and V (p x k)
 */
double KT(SpMatC& M_testC, SpMatR& M_testR, MatrixXd& U, MatrixXd& V)
{
	int Msz=0, M_predsz=0, p = M_testR.cols();
	VectorXd Msub(p);
	VectorXd M_predsub(p);
	double sum = 0.0;
	for(int i=0;i<M_testR.rows();i++)
	{
		M_predsz = 0;
		Msz = subvec(M_testC,M_testR,M_testC,M_testR,1,i,&Msub);
		for (SpMatR::InnerIterator it(M_testR,i); it; ++it)
			M_predsub(M_predsz++) = U.row(i).dot(V.row(it.col()));
		sum += KendalTau(Msub.head(Msz),M_predsub.head(M_predsz),Msz);
	}
	return sum/M_testR.rows();
}

/* Concatenates both M_train and M_test to contiguous vectors only the rows which are there in the cluster
 * Takes Inputs
 * M_train(C/R) (n x p) and M_test(C/R) (n x p) = Sparse Train and test matrices to be concatenated,
 * 		both in Row and Column major forms respectively
 * M_expanded and M_test_expanded = Concatenated vector containeers of sufficient size
 * cluster = Vector of length n storing the cluster number a user row belongs to, in {-1,0,1,...,K-1}
 * 				-1 represents that the user has been alotted no cluster
 * k = The cluster number, 0 <= k < Number of clusters
 * pair_size = pair of integers storing the number of elements of train and test concatenated
 * so that they can be acessed using head call
 */
void concatM(SpMatC& M_trainC, SpMatR& M_trainR, VectorXd& M_expanded, SpMatC& M_testC, SpMatR& M_testR, VectorXd& M_test_expanded, VectorXi& cluster, int k, pair<int,int>& pair_size)
{
	int M_start = 0, M_test_start = 0, Msz, n = M_trainR.rows(), p = M_trainR.cols();
	VectorXd MsubV(p), temp_vec;
	for(int i=0;i<n;i++)
	{
		if(cluster(i)!=k)
			continue;
		Msz = subvec(M_trainC,M_trainR,M_trainC,M_trainR,1,i,&MsubV);
		temp_vec = MsubV.head(Msz);
		M_expanded.segment(M_start,Msz) = temp_vec;
		M_start += Msz;

		Msz = subvec(M_testC,M_testR,M_testC,M_testR,1,i,&MsubV);
		temp_vec = MsubV.head(Msz);
		M_test_expanded.segment(M_test_start,Msz) = temp_vec;
		M_test_start += Msz;
	}
	pair_size.first = M_start;
	pair_size.second = M_test_start;
}

/* Concatenates V_subset.U_i vectors to a single vector used in 1-sigma algorithm.
 * Takes inputs
 * M_train(C/R) (n x p) = Sparse Training Rating Matrix, both in Row and Column major forms respectively
 * U (n x k) and V (p x k) = factor matrices
 * Vui_expanded = The concatenated vector containeer of sufficient length
 * cluster = Vector of length n storing the cluster number a user row belongs to, in {-1,0,1,...,K-1}
 * 				-1 represents that the user has been alotted no cluster
 * k = number of clusters, assumed to be greater than 0
 */
int concatVui(SpMatC& M_trainC, SpMatR& M_trainR, MatrixXd& U, MatrixXd& V, VectorXd& Vui_expanded, VectorXi& cluster, int k)
{
	int n = M_trainR.rows(), sz, Vui_start=0;
	MatrixXd Vsub(V.rows(),V.cols());
	VectorXd temp_vec;
	for(int i=0;i<n;i++)
	{
		if(cluster(i)!=k)
			continue;
		sz = subset(M_trainC,M_trainR,V,1,i,&Vsub);
		temp_vec = (Vsub.topRows(sz))*(U.row(i).transpose());
		Vui_expanded.segment(Vui_start,sz) = temp_vec;
		Vui_start += sz;
	}
	return Vui_start;	
}

/* Computes the mean Vui vector for each rating in {1,2,3,4,5}
 * target_y[r-1] = Vui_expanded[M_expanded==r].mean()
 * Takes input
 * M_expanded = Vector containing concatenated rating values from the rating matrix
 * Vui_expanded = Vector of Vui on which rating depended averaging has to be done
 * target_y = double array of size 5 strong the averages for each rating {1,2,3,4,5}
 */
int get_Vui_mean(VectorXd& M_expanded, const Ref<VectorXd>& Vui_expanded, double target_y[])
{	
	Array<bool,Dynamic,1> mask;
	mask.resize(M_expanded.size());
	int target_y_count = 0;
	VectorXd Exp(M_expanded.size());
	for(int rating=1;rating<=5;rating++)
	{
		mask = (M_expanded.array()==rating);
		if(mask.count())
		{
			for(int i=0,j=0;i<mask.size();i++)	
				if(mask(i))
					Exp(j++)=Vui_expanded(i);
			target_y[rating-1] = (Exp.head(mask.count())).mean();
			target_y_count++;
		}
	}
	return target_y_count;
}

/* Weighted isotonic regression.
 * minimizing over y, sum_i { w_i(y_i - a_i)^2 }
 * such that the constrains y_0<=y_1<=...<=y_(n-1) is maintained.
 * Takes inputs
 * a and w are non empty double arrays of length n>0
*/
double* isotonic(double a[], double w[], int n)
{
	double a_[n], w_[n];
	int S[n], j = 0;
	double* y = new double[n];
	a_[0] = a[0];
	w_[0] = w[0];
	S[0] = -1;
	S[1] = 0;
	for(int i=1;i<=n-1;i++)
	{
		j++;
		a_[j] = a[i];
		w_[j] = w[i];
		while(j>0 && a_[j]<a_[j-1])
		{
			a_[j-1] = (w_[j]*a_[j] + w_[j-1]*a_[j-1])/(w_[j] + w_[j-1]);
			w_[j-1] = w_[j] + w_[j-1];
			j--;
		}
		S[j+1] = i;
	}
	for(int k=0;k<=j;k++)
		for(int l=S[k]+1;l<=S[k+1];l++)
			y[l] = a_[k];
	return y;
}

/* weighted isotonic regression maintaining margin constrains.
 * minimizing over y , sum_i { w_i(y_i - a_i)^2 }
 * such that the constrains
 * y_(i+1) - y_i >= margin(i) for i in {0,1,...,n-2}
 * Takes inputs
 * a and w are non empty double arrays of size n>1
 * margin is a non empty double array of size n-1>0
 */
double* margin_isotonic(double a[], double w[], double margin[], int n)
{
	double* y;
	double t_[n];
	t_[0] = 0;
	copy(margin,margin+n-1,t_+1);
	partial_sum (t_, t_+n, t_);
	reverse(t_,t_+n);
	transform(t_, t_+n, t_,bind1st(multiplies<double>(),-1));
	for(int i=0;i<n;i++)
		a[i] -= t_[i];
	y = isotonic(a,w,n);
	for(int i=0;i<n;i++)
		y[i] += t_[i];
	return y;
}

/* Linearly interpolates and returns sigma function array {1,2,3,4,5} -> {z_0,z_1,...,z_4}
 * using margin isotonic regression.
 * Minimizes sum_{known_i} {w_i(z_i-target_y_i)^2} such that
 * z_(i+1)-z_(i) >= summed margin in the gap of i and (i+1)
 * Takes inputs
 * target_y_count = number of elements known out of 5 in target_y
 * target_y = may be incomplete double array of length 5, not known is assumed to be represented by 0.
 * margins = double array of size 4
 * W = weight double array of size 5.
 */
double* get_interp_Sigma_i(int target_y_count, double target_y[], double margins[], double W[])
{
	vector<double> temp_x,temp_y;
	double* z;
	// if target_y is full, do normal margin isotonic regression
	if(target_y_count==5)
		z = margin_isotonic(target_y,W,margins,5);

	//if target_y has 2, 3 or 4 number of ratings averages, compute spline using only those
	if(target_y_count!=5 && target_y_count!=1)
	{
		for(int rating=1;rating<=5;rating++)
			if(target_y[rating-1]!=0)
			{
				temp_x.push_back(rating);
				temp_y.push_back(target_y[rating-1]);
			}
		double* target_x_temp = &temp_x[0];
		double* target_y_temp = &temp_y[0];

		// constructing the equvalent margin_temp and w_temp array from margins and W to add on the gaps
		double margins_temp[temp_y.size()-1], w_temp[temp_y.size()];
		for(int ii=0;ii<temp_y.size()-1;ii++)
		{
			margins_temp[ii] = 0.0;
			for(int j=temp_x[ii];j<temp_x[ii+1];j++)
				margins_temp[ii] += margins[j-1];
		}
		for(int ii=0;ii<temp_y.size();ii++)
			w_temp[ii] = W[(int)temp_x[ii]-1];
		
		// perform margin isotonic regression on the known values packed in target_y_temp
		double* z_temp = margin_isotonic(target_y_temp,w_temp,margins_temp,temp_x.size());
		// here size of z_temp is 2, 3 or 4
		for(int j=0;j<temp_y.size();j++)
			temp_y[j] = z_temp[j];
		delete z_temp;

		// if size of known target_y values were 2, we extend it to 3 by manually linear interpolaiting
		if(temp_y.size()==2)
		{
			temp_x.push_back(2*temp_x[1]-temp_x[0]);
			temp_y.push_back(2*temp_y[1]-temp_y[0]);
		}

		// here size of z is 3 or 4
		// using spline to construct the complete sigma function array represented by z
		z = new double[5];
		tk::spline s;
		s.set_points(temp_x,temp_y,false);
		for(int rating=1;rating<=5;rating++)
			z[rating-1] = s(rating);
	}

	//if target_y has just one averaged value, return identity function (garbage and will be taken care not to be used)
	else if(target_y_count==1)
	{
		z = new double[5];
		for(int rating=1;rating<=5;rating++)
			z[rating-1] = rating;
	}
	return z;
}

/* Pooled PAV for a particular cluster
 * M_train(C/R) (n x p) = Sparse Rating matrix, both in Row and Column major forms respectively
 * M_expanded = Vector of sufficient length corresponding to the k^th cluster
 * margins = margin array of length 4 used in the margin isotonic regression
 * W = weights array of length 5 used in  the margin isotonic regression
 * cluster = Vector of length n storing the cluster number a user row belongs to, in {-1,0,1,...,K-1}
 * 				-1 represents that the user has been alotted no cluster
 * k = cluster number for which pooled PAV has to be done, in {0,1,...,K}
 */
double* get_1_sigma(SpMatC& M_trainC, SpMatR& M_trainR, VectorXd& M_expanded, MatrixXd& U, MatrixXd& V, double margins[], double W[], VectorXi& cluster, int k)
{
	double target_y[5];
	int train = M_expanded.size();
	VectorXd Vui_expanded(train);
	int Vui_expanded_size = concatVui(M_trainC,M_trainR,U,V,Vui_expanded,cluster,k);
	int target_y_count = get_Vui_mean(M_expanded,Vui_expanded.head(Vui_expanded_size),target_y);
	return get_interp_Sigma_i(target_y_count,target_y,margins,W);
}

/* Computed the inverse of sigma function array into sig_inv
 * Takes inputs
 * z = double array of size 5 containing sigma function array, z[rating-1] = sigma(rating)
 */
void getSigmaInv(double z[], tk::spline& sig_inv)
{
	vector<double> sigma_inv_x, sigma_inv_y;
	for(int rating=1;rating<=5;rating++)
	{
		sigma_inv_x.push_back(z[rating-1]);
		sigma_inv_y.push_back(rating);
	}
	// false for linear interpolation, true for cubic spline
	sig_inv.set_points(sigma_inv_x,sigma_inv_y,false);
}

/* Computes the ith sigma function array
 * Takes inputs
 * M_train(C/R) = Sparse training rating matrix (n x p), both in Row and Column major forms respectively
 * U (n x k) and V (p x k) are the factor matrices
 * margins = non empty double array of margins of size 4
 * W = non empty double array of weights of size 5
 * i = row number of M in {0,1,...,n-1}
 */
double* getSigma_i(SpMatC& M_trainC, SpMatR& M_trainR, MatrixXd& U, MatrixXd& V, double margins[], double W[], int i)
{
	double* z;
	double target_x[5] = {1,2,3,4,5};
	double target_y[5] = {0,0,0,0,0};
	int target_y_count = 0, n = M_trainR.rows(), p = M_trainC.cols(), k = U.cols();
	Array<bool,Dynamic,1> mask;
	mask.resize(p,1);
	MatrixXd Vsub(p,k), Usub(n,k);
	VectorXd MsubV(p), MsubU(n), Exp(n), temp_vec, temp;
	vector<double> temp_x, temp_y;

	// Computing the average of V_subset_r.u_i for each rating r in row i for every i
	for(int rating=1;rating<=5;rating++)
	{
		// mask = (M_train.row(i).array()==rating);
		temp = M_trainR.row(i);
		mask = (temp.array()==rating);
		for(int j=0,jj=0;j<mask.size();++j)
			if(mask(j))
				Vsub.row(jj++) = V.row(j);
		temp_vec = Vsub.topRows(mask.count())*U.row(i).transpose();
		if(mask.count())
		{
			target_y[rating-1] = temp_vec.mean();
			target_y_count++;
		}
	}
	// Computing the linearly interpolated sigma function array
	// as target_y can be incomplete due to sparsity
	z = get_interp_Sigma_i(target_y_count,target_y,margins,W);
	return z;
}

/* Computes the loss from the i^th row of a matrix
 * Row loss for i^th row = sum{j such that M(i,j)!=0} { ( sigma(M(i,j)) - <U_i,V_j> )^2 }
 * Takes inputs
 * M_trainR (n x p) = Sparse Matrix for computing row loss, in Row MAjor form
 * U (n x k), V (p x k) = factored matrices
 * sigma = array of double pointers (assuming allocated)of sigma functions of size (n x 5)
 * i = the row index of M_train for which row loss has to be calculated
 */
double row_loss(SpMatR& M_trainR, MatrixXd& U, MatrixXd& V, double* sigma[], int i, int k)
{
	double temp = 0, loss = 0;
	for(SpMatR::InnerIterator it(M_trainR,i); it; ++it)
	{
		temp = U.row(i).dot(V.row(it.col())) - sigma[k][(int)it.value()-1];
		loss += temp*temp;
	}
	return loss;
}

/* Assigns rows to one of the sigma functions based on minimum row loss
 * Takes inputs
 * M_trainR (n x p) = Sparse Matrix of ratings, in Row Major form
 * U (n x k) and V (p x k) = factor matrices
 * z_cluster = array of K double pointers to sigma function arrays each allocated and of size 5, K>=1
 * cluster = Vector of length n storing the cluster number a user row belongs to, in {-1,0,1,...,K-1}
 * 				-1 represents that the user has been alotted no cluster
 * K = number of clusters, assumed to be greater than 0
 * Returns the number of cluster changes
 */
int assign_sigma(SpMatR& M_trainR, MatrixXd& U, MatrixXd& V, double* z_cluster[], VectorXi& cluster, int K)
{
	int changes = 0;
	for(int i=0;i<M_trainR.rows();i++)
	{
		double min_row_loss = numeric_limits<double>::max(), temp;
		int initial_cluster = cluster(i);
		if(cluster(i)==-1)
			continue;
		for(int k=0;k<K;k++)
		{
			temp = row_loss(M_trainR,U,V,z_cluster,i,k);
			if(temp<min_row_loss)
			{
				min_row_loss = temp;
				cluster(i) = k;
			}
		}
		if(initial_cluster!=cluster(i))
			changes++;
	}
	return changes;
}

/* After change in clusters, the pooled weights for the Pooled PAV needs to  be recalculated
 * Also updates Train and Test cluster histograms
 * Takes inputs
 * M_trainR (n x p) = training rating matrix, in Row Major form
 * M_testR (n x p) = testing rating matrix, in Row Major form
 * W_cluster = array of size K (number of clusters) of double pointers,
 * 				each pointing to doube array of size 5 storing the pooled weights
 * 				of each cluster
 * N_total_train = cluster histogram of train ratings
 * N_total_test = cluster histogram of test ratings
 * cluster = Vector of length n storing the cluster number a user row belongs to, in {-1,0,1,...,K-1}
 * 				-1 represents that the user has been alotted no cluster
 * k = cluster number for which pooled PAV has to be done, in {0,1,...,K}
 * users_with_single_rating = set of indexes in {0,1,...,n-1} to be ignored
 */
void update_W(SpMatR& M_trainR, SpMatR& M_testR, double* W_cluster[], int N_total_train[], int N_total_test[], VectorXi& cluster, int k, set<int> users_with_single_rating)
{
	int N[k][5];
	memset(N, 0, sizeof(N[0][0])*k*5);
	// int N_total[k];
	fill_n(N_total_train, k, 0);
	fill_n(N_total_test, k, 0);
	int n = M_trainR.rows(), p = M_trainR.cols();
	for(int i=0;i<n;i++)
	{
		if(users_with_single_rating.find(i)!=users_with_single_rating.end())
			continue;
		for(SpMatR::InnerIterator it(M_trainR,i); it; ++it)
			if(cluster(i)>=0 && cluster(i)<k)
			{
				N[cluster(i)][(int)it.value()-1]++;
				N_total_train[cluster(i)]++;
			}
		for(SpMatR::InnerIterator it(M_testR,i); it; ++it)
			if(cluster(i)>=0 && cluster(i)<k)
				N_total_test[cluster(i)]++;
	}
	for(int i=0;i<k;i++)
		transform(N[i], N[i]+5, W_cluster[i],bind2nd(divides<double>(),(double)N_total_train[i]));
}