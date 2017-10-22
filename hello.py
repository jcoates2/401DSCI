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


dolphins = ['pink', 'blue', 'green', 'magenta']

print('\nif and else loop')
for dolphin in dolphins:
	#one will definatly will run
	if(dolphin == 'blue'):
		print('Dolphin '+dolphin+' is sad')
	else:
		print(dolphin+" dolphin  is happy")

print('\nif, elif, and else loop')
#one will defininatly run
for dolphin in dolphins:
	if(dolphin == 'blue'):
        	print('Dolphin '+dolphin+' is sad')
	elif(dolphin == 'pink'):
		print("Dolphin "+dolphin+" likes cupcakes")
	else:
        	print(dolphin+" dolphin  is happy")

print('\nif and elif')
for dolphin in dolphins:
	#one or none will run
	if(dolphin == 'blue'):
        	print('Dolphin is sad')
	elif(dolphin == 'pink'):
        	print("Dolphin likes cupcakes")


