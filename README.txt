				Reccomendation Systems using order preserving monotonic transformations
				=======================================================================

The project consists of 4 algorithms :-
	1. No-sigma or the naive Matrix factorization method,
	2. 1-sigma where a universal monotonic function is trained,
	3. N-sigma where each user is provided with its own monotonic function, and
	4. k-sigma where the functions obtained in N-sigma are partitioned into k clusters each cluster having its own monotonic transformation.

Source code
===========

The corresponding files for each of the experiments are :-
	1. MF.cpp
	2. 1_sigma.cpp
	3. N_sigma.cpp
	4. k_sigma.cpp

Eigen library is been used to work with matrices and linear algebra - /eigen
All the general function definitions are inside - sigma.cpp
All the function definitions for k-means clustering are inside - km.cpp
A header file for spline type object is used in the project - spline.h
A header file for including other necessary header files and defines - init.h
Shell script to run each of the 4 experiments - run.sh

Project Directory Structure
===========================

RecSys+
	  |--data+
	  :		 |--<described in the below section>
	  |--datasets+
	  :			 |-- u100k.data (movielens_small dataset)
	  :			 |-- u1m.dat (movielens_medium dataset)
	  :			 |-- u10m_mapped_subset.dt (movielens_large dataset)
	  |--src+
	  :		|--eigen (library for linear algebra)
	  :		|-- MF.cpp
	  :		|-- 1_sigms.cpp
	  :		|-- N_sigma.cpp
	  :		|-- k_sigma.cpp
	  :		|-- init.h
	  :		|-- spline.h
	  :		|-- sigma.cpp
	  :		|-- km.cpp
	  :		|-- run.sh
	  :		|-- params.txt (raw text file storing optimal parameters)
	  :		|-- opt_params.cpp (a demp file showing how to extract optimal parameters)
	  |-- README.txt (this file)
	  +-- Comparison.ods (A libre office file storing all the grid search work with summaries)

Data directory structure
========================

All the data is stored in the directory "data" which has a defined structure
data+
	|--movielens_small
	:
	:
	|--movielens_medium
		|--k1
		|--k5
		:
		:
		+--k<max> (here "k50")
			|--1_sigma_
			|--N_sigma_
			|--No_sigma_
				results.txt
				|--30_3
				|--5_25
					|--KT.txt
					|--losses.txt
					|--U.txt
					|--V.txt
			+--k_sigma_
				|--0.01_0.01
					|--KT.txt
					|--losses.txt
					|--U.txt
					|--V.txt
					|--sigma.txt
				|--K_1_0.06
					|--50
						|--KT.txt
						|--losses.txt
						|--U.txt
						|--V.txt
					:
					+
				:
				+
	:
	+

The data folder contains subfolders for each kind of dataset.
	Each dataset folder contains subfolders for each low rank k
		Each low rank k folder has subfolders for the 4 algorithms
			Algorithms No_sigma, 1_sigma and N_sigma have subfolders for l1 and l2 hyperparameters as "<l1>_<l2>"
				Each hyperparameter folder has 4 (5 for N_sigma) files
					KT.txt storing the Kendal Tau correlation in that experiment
					losses.txt storing the train and test loss series
					U.txt has the trained U matrix
					V.txt has the trained V matrix
					sigma.txt has the n different sigma functions (only in N_sigma)
				result.txt has the information regarding the last conducted experiment (overwritten always)
			Algorithm k_sigma has 2 kind of directories, one prefixed with "K" and the other not prefixed with "K"
				The directory not prefixed with "K" corresponds to the N_sigma step with subfolder named "<l1>_<l2>"
					This again has 4 files KT.txt, losses.txt, U.txt and V.txt as described above
				The directory prefixed with "K" corresponds to the hyperparameters of the pooled 1-sigma hyperparametsr as "K<l1>_<l2>". It has subfolders for each K (# of clusters) used.
					Each folder corresponding to the number of clusters used again have the 4 files KT.txt, losses.txt, U.txt and V.txt as described above
				result.txt has the information regarding the last conducted experiment (overwritten always)

The format of the result.txt files in No_sigma, 1_sigma and N_sigma is
	<l1>\t<l2>
	<Train loss at the end of iterations>\t<Test loss at the end of iterations>
	...
	...
	LT
	<Grid of Test losses>
	KT
	<Grid of Kendall Tau corelations>

The format of the result.txt files in k_sigma is same as that of others with appended results of
	<same as the result.txt of other 3 algorithms>
	-----------------------
	<l1>\t<l2>
	<Train loss at the end of iterations>\t<Test loss at the end of iterations>
	...
	...
	LT
	<Grid of Test losses>
	KT
	<Grid of Kendall Tau corelations>

Running an experiment
=====================
A shell script with arguments is used for simplicity for compiling the code corresponding to each experiment.

	run.sh
	------
	g++ -std=c++11 -O3 -Ieigen -fno-math-errno -funsafe-math-optimizations -fno-rounding-math -fcx-limited-range -fno-signaling-nans "$1"

	Compile
	-------
	./run.sh 1_sigma.cpp

	Run
	---
	./a.out

The run.sh can be changed in order to have different or extra flags or having differently named compiled files.

Obtaining the optimal hyperparameters
=====================================

A class heirarchy is created for obtaining the optimal hyperparameters from a file.
The file storing the optimal hyperparameters is - params.txt, and has a specific format which must be strictly followed.
First line is the number of dataset chunks
	For each dataset chunk the first line is the name of the dataset
	The next line for each dataset chhunk, is the number of low ranks tuned
		For each low rank the first line is the low rank number itself followed by 8 lines
			The first 2 lines are the optimal <l1><space><l2> for best KT and TL respectively for No sigma
			The next 2 lines are the optimal <l1><space><l2> for best KT and TL respectively for 1 sigma
			The next 2 lines are the optimal <l1><space><l2> for best KT and TL respectively for N sigma
			The next 2 lines are the optimal <l1><space><l2><space><clusters> for best KT and TL respectively for K sigma

-1 is used to denote the hyperparameters not tuned yet
A demo file for obtaining these optimal parameters from the file is - opt_params.cpp

These optimal parameters are stored in a class based hierarchy.
Dataset
	|-- string name
	|-- algo No_sigma
	|-- algo sigma_1
	|-- algo sigma_N
	|-- algo sigma_K
		  |-- map<int,low_rank> Kmap
		  		   |-- low rank integer, low_rank object
		  		   							|-- measure KT
		  		   							|-- measure TL
		  		   									|-- l1
		  		   									|-- l2
		  		   									|-- K (default to 0 for algos other than k-sigma)
		  		   									|-- set_params function to set the above 2 (or 3 for k-sigma)

Datasets
========

The experiments have been done on 2 datasets.
	1. Movie lens small (943 users, 1682 movies, 100000 ratings) - u100k.data
	2. Movie lens medium (6040 users, 3952 movies, 1000209 ratings) - u1m.dat

It is expected that the rating data has a format of
<user_id> <movie_id> <rating>
in each of its lines
user_id range from 1, continuously upto the number of users
movie_id range from 1, continuously upto the number of movies
ratings are from the set {1,2,3,4,5} only

P.S. :- Kindly edit this README section and params.txt if the work is done for more datasets than the above mentioned :)
