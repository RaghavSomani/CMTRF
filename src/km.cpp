/* Converts a C matrix (pointer to double arrays) to Eigen matrix
 * Assuming the size of both the matrix are same
 */
void CMatToEigenMat(double* z[], MatrixXd& Z)
{
	int n = Z.rows(), p = Z.cols();
	for(int i=0;i<n;i++)
		for(int j=0;j<p;j++)
			Z(i,j) = z[i][j];
}

/* Converts Eigen matrix to C matrix (array of double pointers)
 * Allocates memory to each double pointer equal to the number of columns in the Eigen Matrix
 * Assumed that size of double pointer array is atleast equal to the number of rows of Eigen matrix
 */
void EigenMatToCMat(MatrixXd& Z_centers,double* z_cluster[])
{
	int n = Z_centers.rows(), p = Z_centers.cols();
	for(int i=0;i<n;i++)
	{
		z_cluster[i] = new double[p];
		for(int j=0;j<p;j++)
			z_cluster[i][j] = Z_centers(i,j);
	}
}

/* Selects k distinct random integers from {0,1,...,n-1}
 * Assumed that k<=n
 * Returns a pointer to an integer array of size k
 */
int* selectKItems(int n, int k)
{
    int i;
    int* reservoir = new int[k];
    for (i = 0; i < k; i++)
        reservoir[i] = i;
    srand(1);
    for (;i<n;i++)
    {
        int j = rand() % (i+1);
        if (j < k)
          reservoir[j] = i;
    }
    return reservoir;
}

/* Chooses random rows from Matrix z and fills in z_centers
 * Takes inputs
 * z = Matrix with n>=k rows
 * z_centers = matrix with k rows and same number of columns as of z
 */
void lloyds_initialize(MatrixXd& z, MatrixXd& z_centers)
{
	int* center_indexes = selectKItems(z.rows(),z_centers.rows());
	for(int i=0;i<z_centers.rows();i++)
		z_centers.row(i) = z.row(center_indexes[i]);
	delete center_indexes;
}

/* Assigns centers to each cluster as a part of a step of lloyd's algorithm
 * Takes inputs
 * Z = Matrix of size (n x p), all the data points as row vectors
 * Z_centers = matrix of size (k x p) which is to be filled,
 * 				Z_centers.row(k) = Z[cluster==k].mean()
 * cluster = Vector of length n storing the cluster number a user row belongs to, in {-1,0,1,...,K-1}
 * 				-1 represents that the user has been alotted no cluster
 * users_with_single_rating = set of integer row indexes in {0,1,...,n-1} which are to be ignored while cluster center averaging
 */
void cluster_centers(MatrixXd& Z, MatrixXd& Z_centers, VectorXi& cluster, set<int>& users_with_single_rating)
{
	int cluster_count[Z_centers.rows()];
	memset(cluster_count, 0, sizeof(cluster_count[0])*Z_centers.rows());
	Z_centers = MatrixXd::Zero(Z_centers.rows(),Z_centers.cols());
	for(int i=0;i<Z.rows();i++)
	{
		if(users_with_single_rating.find(i)!=users_with_single_rating.end())
			continue;
		Z_centers.row(cluster(i)) += Z.row(i);
		cluster_count[cluster(i)]++;
	}
	for(int k=0;k<Z_centers.rows();k++)
		Z_centers.row(k) = Z_centers.row(k)/cluster_count[k];
}

/* Computes the within cluster variation of the clustered dataset
 * Z = matrix of data points size (n x p)
 * Z_centers = Matrix of centers, size (k x p), assumed k<=n
 * cluster = Vector of length n storing the cluster number a user row belongs to, in {-1,0,1,...,K-1}
 * 				-1 represents that the user has been alotted no cluster
 * users_with_single_rating = set of integer row indexes in {0,1,...,n-1} which are to be ignored
 */
double WCV(MatrixXd& Z, MatrixXd& Z_centers, VectorXi& cluster)
{
	double var = 0;
	for(int k=0;k<Z_centers.rows();k++)
		for(int i=0;i<Z.rows();i++)
			if(cluster(i)==k)
				var += (Z.row(i)-Z_centers.row(k)).squaredNorm();
	return var;
}

/* Assigns cluster numbers to each row as a part of a step of lloyd's algorithm
 * Z = matrix of data points of size (n x p)
 * Z_centers = Matrix of centers, size (k x p), assumed k<=n
 * cluster = cluster = Vector of length n to be filled with cluster number for each user row, in {-1,0,1,...,K-1}
 * 				-1 represents that the user has been alotted no cluster
 * users_with_single_rating = set of integer row indexes in {0,1,...,n-1} which are to be ignored and assign a cluster number = -1
 */
void assign_cluster(MatrixXd& Z, MatrixXd& Z_centers, VectorXi& cluster, set<int>& users_with_single_rating)
{
	int n = Z.rows(), p = Z.cols();
	double d = 0;
	int min_k;
	for(int i=0;i<n;i++)
	{
		if(users_with_single_rating.find(i)!=users_with_single_rating.end())
		{
			cluster(i) = -1;
			continue;
		}
		double min_d = numeric_limits<double>::max();
		for(int k=0;k<Z_centers.rows();k++)
		{
			d = (Z.row(i) - Z_centers.row(k)).norm();
			if(d<min_d)
			{
				min_d = d;
				min_k = k;
			}
		}
		cluster(i) = min_k;
	}
}

/* Performs iteration of the 2 steps of k means in the lloyd's algorithm
 * Stopping criteria is when relative change in within cluster variance goes below 1e-5
 * Takes inputs
 * Z = Matrix of data points of size (n x p)
 * Z_centers = Matrix of cluster centers of size (k x p)
 * cluster = Vector of length n storing the cluster number a user row belongs to, in {-1,0,1,...,K-1}
 * 				-1 represents that the user has been alotted no cluster
 * users_with_single_rating = set of row indexes in {0,1,...,n-1} which are to be ignored
 */
void kmeansiterate(MatrixXd& Z, MatrixXd& Z_centers, VectorXi& cluster, set<int>& users_with_single_rating)
{
	int k = Z_centers.rows();
	assign_cluster(Z,Z_centers,cluster,users_with_single_rating);
	vector<double> TWCV;
	TWCV.push_back(WCV(Z,Z_centers,cluster));
	int iterations = 0;
	while(true)
	{
		cluster_centers(Z,Z_centers,cluster,users_with_single_rating);
		TWCV.push_back(WCV(Z,Z_centers,cluster));
		assign_cluster(Z,Z_centers,cluster,users_with_single_rating);
		TWCV.push_back(WCV(Z,Z_centers,cluster));
		iterations++;
		cout<<TWCV.back()<<endl;
		if( fabs(TWCV[iterations*2] - TWCV[iterations*2-1])/TWCV[iterations*2-1] < 1e-5 )
			break;
	}
}

/* k means function, implements lloyd's algorithm
 * Z = Matrix of data points of size (n x p)
 * Z_centers = Matrix of cluster centers of size (k x p)
 * cluster = Vector of length n storing the cluster number a user row belongs to, in {-1,0,1,...,K-1}
 * 				-1 represents that the user has been alotted no cluster
 * users_with_single_rating = set of row indexes in {0,1,...,n-1} which are to be ignored
 */
void kmeans(MatrixXd& Z, MatrixXd& Z_centers, VectorXi& cluster, set<int>& users_with_single_rating)
{
	lloyds_initialize(Z,Z_centers);
	kmeansiterate(Z,Z_centers,cluster,users_with_single_rating);
}

/* calculates the histogram of cluster, the number of data points in each cluster
 * Takes inputs
 * cluster = Vector of length n storing the cluster number a user row belongs to, in {-1,0,1,...,K-1}
 * 				-1 represents that the user has been alotted no cluster
 * cluster_count = integer array of size K,
 					cluster_count[k] = number of data points in k^th cluster
 * K = number of clusters, assumed >=1
 */
void cluster_hist(VectorXi& cluster, int cluster_count[], int K)
{
	fill_n (cluster_count,K,0);
	for(int i=0;i<cluster.size();i++)
		if(cluster(i)>=0 && cluster(i)<K)
			cluster_count[cluster(i)]++;
}

void furthest_centers(MatrixXd& Z, MatrixXd& Z_centers, VectorXi& cluster, set<int>& users_with_single_rating)
{
	int n = Z.rows(), p = Z.cols(), k = Z_centers.rows();
	set<int> remaining_data_indexes, center_indexes;
	for(int i=0;i<n;i++)
		remaining_data_indexes.insert(i);
	int* a = selectKItems(n,1);
	center_indexes.insert(a[0]);
	remaining_data_indexes.erase(a[0]);
	Z_centers.row(0) = Z.row(a[0]);
	cluster(a[0]) = 0;
	int to_fill = 1;
	double d;
	while(center_indexes.size()!=k)
	{
		double D[n];
		int new_index;
		for(int i=0;i<n;i++)
		{
			if(users_with_single_rating.find(i)!=users_with_single_rating.end())
			{
				cluster(i) = -1;
				continue;
			}
			if(center_indexes.find(i)!=center_indexes.end())
			{
				D[i] = 0;
				continue;
			}
			D[i] = numeric_limits<double>::max();
			for(auto center_i:center_indexes)
			{
				d = (Z.row(i) - Z.row(center_i)).norm();
				if(d<D[i])
				{
					D[i] = d;
					new_index = i;
				}
			}
		}
		center_indexes.insert(new_index);
		remaining_data_indexes.erase(new_index);
		Z_centers.row(to_fill) = Z.row(new_index);
	}
}