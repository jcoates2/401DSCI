import function as bs

print(bs.add_2(3,5))
print(bs.add_2(6,9))
print(bs.add_2(2,3))


print(bs.my_range(1,50))
print(bs.my_range(1,11,3))
print(bs.my_range(1,50,3))


#print(bs.print_triangle(3))
#print(bs.print_triangle(5))
#print(bs.print_triangle(7))


print(bs.histogram([1,1,1,1,2,2,2,2,4]))
print('\n')

#print(bs.classhist(['a', 'x', '2', 'x', '2', '3']))

print(bs.word_counts('data/text_data.txt'))

print(bs.word_c('data/text_data.txt'))

#print(bs. my_max([4,23, 1, 9, 4, 9]))

#print(bs. my_max([4,23, 1, 900, 4, 9]))

#print(bs.max_m([4,23, 1, 9, 4, 9]))

print(bs.fzip(lambda x: x+x, ([1,2,3], [4,5,6])))
print(bs.fzip(max, ([1,2,3], [4,5,6], [7,8,9]) ) )

print(bs.sum_range(1,5))

print(bs.rrev([1,1,1,1,2,2,2,2,4]))

print(bs.mfib(1,1,100))

print(bs.cartesian_product( [1,2], [3,4] ))

print(bs.kcombos([1,2,3,4], 2))

#test the pipe function
f1 = lambda x: x+3
f2 = lambda x: x* x
f3 = lambda x: x/ 2.3
f4 = lambda x: x ** 0.5
#construct a new function that popes the above in sequence 
my_pipe = bs.pipe(f1, f2, f3, f4)
#apply my_pipe to the numbers 1 thru 20
print(map(my_pipe, range(1,21)))




