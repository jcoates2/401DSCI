print("Hello World")

e1 = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
e2 = [(1,'a'), (2,'b'), (3,'c'), (4,'d'), (5, 'e')]

#print out list
for char in e1:
	print("Next Character" + char)

print('\n\n')

#print out elemtns in list of truples
for (number, letter) in e2:
	print("The number is "+ str(number) + "\nThe Character is "+ letter)
	print('-----------------------')
