#assignment 1
#Jacqueline Coates

import math as m

#takes a list or lists of lists and puts all items into a single list on the same level.
def flatten(oldlist):
	if all(isinstance(y, int) for y in oldlist):
		return oldlist
	else:
		newlist = []
		#iterate thru old list
		for x in range(0, len(oldlist)):
			#print(isinstance(oldlist[x], list))
			#if there is still a list type in oldlist
			if isinstance(oldlist[x], list):
				i = 0
				while i < len(oldlist[x]):
					newlist.append(oldlist[x][i])
					i += 1
			else:
				newlist.append(oldlist[x])
			#print(newlist)
		return flatten(newlist)
	
	
#returns the set of all possible subsets for a set
def powerset(list):
	#base case
	if len(list) == 1:
		return list
	elif len(list) == 2:
		newlist = []
		newlist.append([ list[0], list[1] ])
		newlist.append( powerset( [list[0]] ) )
		newlist.append( powerset( [list[1]] ) )
		return newlist
	else:
		newlist = [list]
		for x in range(1, len(list)):
			newlist.append([list[0], list[x]])
		newlist.append( powerset(list[1::]) )
		newlist.append([list[0]])
		newlist.append([])
		return newlist

	
#produce all permutations of a list as follows
def all_perms(list):
	newlist = []
	if len(list) == 0:
		return []
	if len(list) == 1:
		return [list]
	else:
		for x in range(len(list)):	
			temp = list[:x] + list[x+1:]
			for p in all_perms(temp):
			 	newlist.append([list[x]] + p)
	return newlist


def spiral(n, end_corner):	
	endnum = n**2
	#create list of list temp value 0
	spiral = [[-1]* n for j in range(n)]
	#find which corner to start at
	x = 0
	y = 0	
	dx = 1
	dy = 0		
	if end_corner == 2:
		x,y = (n-1), 0
		dx = 0
		dy = 1
	elif end_corner == 3:	
		x,y = 0, (n-1)	
		dx = -1
		dy = 0
	elif end_corner == 4:
		x,y = (n-1), (n-1)
		dx = -1
		dy = 0
		
	all = range(endnum)
	all.reverse()	
	for t in range(0,endnum):
		spiral[x][y] = all[t]
		potx, poty = x+dx, y+dy
		if (potx >= 0 and potx < n) and (poty >= 0 and poty < n) and spiral[potx][poty] == -1:
			x = potx
			y = poty
		else:
			dx, dy = -dy, dx
			x =  x+dx
			y =  y+dy
		#print('DX:'+ str(dx)+ " DY: "+ str(dy))
			
	#print out the spiral
	s = ''
	for p in range(len(spiral)):
		for l in range(len(spiral)):
			if spiral[p][l] < 10:
				s += str(spiral[p][l]) + '  '
			else:
				s += str(spiral[p][l]) + ' '
		print(s)
		s = ''
		
			
			
	
		
		
		

