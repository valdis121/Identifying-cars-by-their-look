from collections import defaultdict
import random

def def_value1():
    return "Nothing"
def def_value2():
    return []

minCost = 0
maxCost = 100
pathToLabels = '../VehicleID_V1.0/train_test_split/train_list.txt'
nameOfResult = 'result.csv'
numberOfSamplesOnOneCar = 10

labels = open(pathToLabels, 'r')
result = open(nameOfResult, 'w')
result.write("car1,car2,distance\n")

carsId = defaultdict(def_value1)
idCars = defaultdict(def_value2)

for line in labels:
    strings = line.split(' ')
    car = ''.join(e for e in strings[0] if e.isalnum())
    id = ''.join(e for e in strings[1] if e.isalnum())
    carsId[car] = id
    idCars[id].append(car)

n = 0
cars = list(carsId.keys())
ids = list(idCars.keys())
for i in ids:
    car = idCars[i][0]
    for x in idCars[i]:
        if car == x:
            continue
        n += 1
        dis = minCost
        result.write("{},{},{}\n".format(car, x, dis))


for x in range(len(cars)):
    if x + 1 >= len(cars):
        break
    randomList = random.sample(range(0, len(cars)), numberOfSamplesOnOneCar)
    for y in randomList:
        car1 = cars[x]
        car2 = cars[y]

        if car1 == car2:
            continue

        dis = maxCost
        n += 1
        if carsId[car1] == carsId[car2]:
            continue
        result.write("{},{},{}\n".format(car1, car2, dis))

labels.close()
result.close()
print("Num of samples = {}".format(n))

