mapping = dict()
g = open('review.dat','r')
h = open('review_unique.dat','w')
for line in g:
	lst = line.split()
	user = int(lst[0])
	movie = int(lst[1])
	if (user,movie) not in mapping:
		mapping[(user,movie)] = lst[2]
		h.write(str(user)+" "+str(movie)+" "+lst[2]+"\n")
	else:
		mapping[(user,movie)] = lst[2]

g.close()
h.close()