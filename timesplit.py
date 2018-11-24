# f = open('ratings_10m.dat','r')
# g = open('u10m.dat','w')
# rat_list = []
# for line in f:
# 	lst = line.split("::")
# 	lst = lst[-1:] + lst[:-1]
# 	rat_list.append(" ".join(lst))
# rat_list.sort()
# for line in rat_list:
# 	lst = line.split()
# 	g.write(lst[1]+" "+lst[2]+" "+lst[3]+"\n")
# f.close()
# g.close()

count = 0
ui = 0
mj = 0
itr = 1
user_map = dict()
movie_map = dict()
g = open('review_unique.dat','r')
h = open('review_mapped.dat','w')
for line in g:
	lst = line.split()
	user = int(lst[0])
	movie = int(lst[1])
	# if "." in lst[2]:
	# 	continue
	if user not in user_map:
		user_map[user] = ui
		new_ui = ui
		ui = ui + 1
	else:
		new_ui = user_map[user]
	if movie not in movie_map:
		movie_map[movie] = mj
		new_mj = mj
		mj = mj + 1
	else:
		new_mj = movie_map[movie]
	h.write(str(new_ui)+" "+str(new_mj)+" "+lst[2]+"\n")
	if(itr==760000):
		print ui, mj
	# if(new_ui<=62007 and new_mj<=10586 and itr>6400000):
	# 	count = count + 1
	itr = itr + 1
# print count
g.close()
h.close()

g = open('user_map_10m.txt','w')
h = open('movie_map_10m.txt','w')
for (k,v) in user_map.items():
	g.write(str(k)+" "+str(v)+"\n")
for (k,v) in movie_map.items():
	h.write(str(k)+" "+str(v)+"\n")
g.close()
h.close()