#basic function
#from funtools import *

def add_2(x, y):
	return x + y

def my_range(start, end, by=1):
	numbers = []
	i = start
	while i < end:
		numbers.append(i)
		i += by
	return numbers

#rewrite my range to use a for loop rather than restoring to pyton builitn range function

def histogram(list):
    count = 0
    dictionary = {}
    output = []
    for x in list:
        if x not in output:
            output.append(x)
    print(output)
    for x in output:
        for j in list:
            if x == j:
                count += 1
        dictionary.update({x: count})
        count = 0
    return dictionary
    
#read in a file and gets the word counts as a histogram
def word_counts(file_path, case_sensitive=True):
	text = open(file_path, 'r').read().replace('\n', ' ')
	text = text.replace('.', '')
	text = text.replace(' ', '')
	print(text)
	return len(text) - text.count(' ')
	
def word_c(file_path, case_sensitive =True, punct=['!', '.', ',', '"', '?', '~']):
	text = open(file_path, 'r').read()
	if not(case_sensitive):
		text = text.lower()
	#todo: add code to count each punctuation
	#for time being, remove punctuation characters
	count = 0
	for p in punct:
		text = text.replace(p, '')
		count += text.count(p)
	words = text.split(' ')
	print(count)
	cleaned_words =[]
	for w in words:
		if len(w) > 0:
			cleaned_words.append(w.strip()) 
	return histogram(cleaned_words)
		
def print_triangle(n, full=False):
	pos = 1
	while pos <= n:
		print('*' * pos)
		pos += 1
	if full:
		pos = n - 1
		while pos >= 1:
			print('*' * pos)
			pos -= 1

def variable_number_ofinputs(a, b, *test):
	print("A is " + str(a))
	print("B is " + str(b))
	for e in test:
		print(" Next Optional Input: "+ str(e))
	
#return maximum element in a list			
#def my_max(elements):
#	tempmax = elements[0]
#	for x in range(1, len(elements)):
#		if(elements[x] > tempmax):
#			tempmax = elements[x]
#	return tempmax	
	
def fzip(f, *element):
	return list(map(lambda tup: f(*tup), zip(*element) ) )
	
def sum_range(a, b):
	if a == b:
		return a
	else:
		return sum_range(a, b-1) + b

#returns a reversed list
def rrev(list):
	if len(list) == 1:
		return list[0]
	else:
		return str(rrev(list[1:])) +''+str(list[0])
		
def fib(first, second, n):
	if n == 1:
		return first;
	if n == 2:
		return second;
	else:
		return fib(first, second, n-1) + fib(first, second, n-2)

def mfib(first, second, n, cache={}):	
	if n == 1:
		return first
	if n == 2:
		return second
	elif n in cache:
	#if this has been found
		return cache[n]
	else:
		v = mfib(first, second, n-1, cache) + mfib(first, second, n-2, cache)
		cache[n] = v
		return v

def cp(a,b):
	cps = []
	for i in a:
		for x in b:
			cps.append((i,x))
	return cps
	
def cartesian(*sets):
	if len(sets) == 1:
		return map(lambda x: (x), set[0])
	else:
		cps = []
		rest = cartesian(*sets[1:])
		for x in a:
			for y in b:
				cps.append(tuple[:] + list(j))
		return cps

#computer the cartesian product of the given sets
def cartesian_product(*sets):
	if len(sets) == 1:
		return map(lambda x: [x], sets[0])
	else:
		rest = cartesian_product(*sets[1:])
		combine = lambda x: map(lambda y: [x] + y, rest)
		return reduce(lambda x,y: x+y, map(combine, sets[0]))

#find all distinct combos of k elements taken from elts		
def kcombos(elts, k):
	if len(elts) == k:
		return [elts]
	if k == 1:
		return map(lambda x: [x], elts)
	else:
		partials  = kcombos(elts[1:], k-1)
		return map(lambda x: [elts[0]]+x, partials) + kcombos(elts[1:], k)

def combinations(iterable, r):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = range(r)
    print (tuple(pool[i] for i in indices))
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        print( tuple(pool[i] for i in indices))	

#build a compositional pope function
def pipe(*function_sequence):
	def applier(input):
		output = input
		for f in function_sequence:
			output = f(input)
		return output
	return applier

		