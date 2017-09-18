#basic function
def add_2(x, y):
	return x + y

def my_range(start, end, by=1):
	numbers = []
	i = 0
	for x in range(start, end,1):
		numbers[i] = x
		i+=1
	return numbers

#rewrite my range to use a for loop rather than restoring to pyton builitn range function

