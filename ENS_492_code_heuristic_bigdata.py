import pandas as pd
import numpy as np
from math import sqrt
import geopy.distance
import matplotlib.pyplot as plt
import math
import pqdict
from pqdict import pqdict

#Calculates the euiclidian distance
def euiclidian(x1,y1,x2,y2):
    return sqrt((x1-x2)**2+(y1-y2)**2)


#Reading the TXT File,
df = pd.read_csv("r112_21.txt", sep='\s+', header=None, skiprows=[0], names= ["StringID", "Type", "x", "y", "demand", "ReadyTime", "DueDate", "ServiceTime"])
df['x'] = df.iloc[0:103, 2].apply(pd.to_numeric, errors='coerce')
df['y'] = df.iloc[0:103, 3].apply(pd.to_numeric, errors='coerce')
df['demand'] = df.iloc[0:103, 4].apply(pd.to_numeric, errors='coerce')
df['ServiceTime'] = df.iloc[0:103, 7].apply(pd.to_numeric, errors='coerce')


demandf = df.iloc[0:103, 4:5]

#Read earliest and latest start times
earliest_start = df.iloc[0:103, 5].apply(pd.to_numeric, errors='coerce').values.tolist()
latest_start = df.iloc[0:103, 6].apply(pd.to_numeric, errors='coerce').values.tolist()


#Combine them into a single list of tuples
time_window = list(zip(earliest_start, latest_start))
 
df = df[["x","y"]]
coorddf=df
cols_dist = []
values = []
for i in range(101): 
  cols_dist.append(i)

for i in range(len(df)):
  val = []
  for j in range(len(df)):
    val.append(euiclidian(df.loc[i, "x"],df.loc[i, "y"],df.loc[j, "x"],df.loc[j, "y"]))
  values.append(val)

df = pd.DataFrame(values, columns = cols_dist)
demandf = pd.DataFrame(demandf.values, columns = ["demand"])
nodes = list(range(df.shape[0]))

#Distance matrix is created
d = [[round(float(df[j][i]), 2) for j in nodes] for i in nodes]

#Demand matrix is created
demand = [round(float(demandf['demand'][j]), 2) for j in nodes]

origin = 0  # origin for the VRP routes, origin has selected
vehicle_capacity = 200 #vehicle capacity is selected C

#Set of customer nodes (i.e. nodes other than the origin)
customers = {i for i in nodes if i != origin} 

#Initialize out-and-back tours from the origin to every other node
tours = {(i, i, demand[i]): [origin, i, origin] for i in customers}

#Compute the savings
savings = {(i, j): round(d[i][origin] + d[origin][j] - d[i][j], 2)
           for i in customers for j in customers if j != i}


#Defining the battery capacity
battery_capacity = 65.48 # sample battery capacity, in kilometers Q

#Defining the energy consumption rate
h = 1  # R # Energy consumed to traverse arc(i,j),it is the consumption per km, 1260 (Robot Battery Capacity) / Robot consumption rate per km: 2.4 km (it is the range of the robot the diameter area which means that it can travel at most 2.4 km with fully charged) = 525

#Define a priority queue dictionary to get a pair of nodes (i,j) which yields the maximum savings.
#Priority queue sorts from low to high, but we make reverse = True, it becomes highest low 
#For example: savings = {(1,2): 44, (1,3) : 38 .... } when we make it reverse it becomes list again to make it dictionary again we used pqdict

pq = pqdict(savings, reverse=True)
arcs = []
# Execute the algorithm until all saving values are evaluated
while len(pq) > 0:
    merged = False
    i, j = pq.pop()
    print("Selected customer pair:", (i, j))
    break_outer = False
    for t1 in list(tours):
        for t2 in list(tours.keys() - {t1}):
            if t1[1] == i and t2[0] == j and t1[2] + t2[2] <= vehicle_capacity and tours[t1][-2] != origin and tours[t2][1] != origin:
                
                #Check if the merged tours meet time window constraints
                time_violation = False
                tour1_time = time_window[t1[0]][0]
                tour2_time = time_window[t2[0]][0]
                for index in range(1, len(tours[t1])):
                    tour1_time += d[tours[t1][index - 1]][tours[t1][index]]
                    if not time_window[tours[t1][index]][0] <= tour1_time <= time_window[tours[t1][index]][1]:
                        time_violation = True
                        break
                if not time_violation:
                    for index in range(1, len(tours[t2])):
                        tour2_time += d[tours[t2][index - 1]][tours[t2][index]]
                        if not time_window[tours[t2][index]][0] <= tour2_time <= time_window[tours[t2][index]][1]:
                            time_violation = True
                            break
                    if not time_violation:
                        totalDemand = t1[2] + t2[2]
                        tours[(t1[0], t2[1], totalDemand)] = tours[t1][:-1] + tours[t2][1:]
                        tour_time = [time_window[tours[(t1[0], t2[1], totalDemand)][0]][0]]
                        for index in range(1, len(tours[(t1[0], t2[1], totalDemand)])):
                            tour_time.append(max(time_window[tours[(t1[0], t2[1], totalDemand)][index]][0], tour_time[-1] + d[tours[(t1[0], t2[1], totalDemand)][index - 1]][tours[(t1[0], t2[1], totalDemand)][index]]))
                            if not time_window[tours[(t1[0], t2[1], totalDemand)][index]][0] <= tour_time[-1] <= time_window[tours[(t1[0], t2[1], totalDemand)][index]][1]:
                                time_violation = True
                                break
                        if not time_violation:
                            del tours[t1], tours[t2]
                            break_outer = True
                            break
                if break_outer:
                    break
                #Check if the merged tours meet battery capacity constraints
                battery_violation = False
                battery_used = 0
                for index in range(len(tours[t1]) - 1):
                    battery_used += d[tours[t1][index]][tours[t1][index + 1]] * h
                    if battery_used > battery_capacity:
                        battery_violation = True
                        break
                if not battery_violation:
                    for index in range(len(tours[t2]) - 1):
                        battery_used += d[tours[t2][index]][tours[t2][index + 1]]
                        if battery_used > battery_capacity:
                            battery_violation = True
                            break
            
                if not time_violation and not battery_violation:
                    totalDemand = t1[2] + t2[2]
                    print('Merging', tours[t1], 'and', tours[t2])
                    new_tour = tours[t1][:-1] + tours[t2][1:]
                    tours[(t1[0], t2[1], totalDemand)] = new_tour
                    del tours[t1], tours[t2]
                    break_outer = True  # Set break_outer to True when a merge is successful

                    # Update the savings priority queue
                    new_savings = {}
                    for (ii, jj), sav in savings.items():
                        if ii == t1[0] or ii == t2[0]:
                            ii = t1[0], t2[1]
                        if jj == t1[1] or jj == t2[1]:
                            jj = t1[0], t2[1]
                        if ii != jj:
                            new_savings[(ii, jj)] = sav
                    pq = pqdict(new_savings, reverse=True)
                    if break_outer:
                        break
            if break_outer:
                break

    else:
        print('No merging opportunities can be found for', (i, j))
       
                                         
#Compute the total traveling distance
vrp_solution_length = 0    #Total length initially determined as 0 
for tour in tours.values():   #For all tour lists, we used .values() and we reached to the list
    for i in range(len(tour) - 1):       #We included the one previous index of the tour
        vrp_solution_length += d[int(tour[i])][int(tour[i + 1])]     #We calculated the distances between the customer nodes in that tour, and we added to the vrp_solution_length 

# Round the result to 2 decimals to avoid floating point representation errors, it is not really necessary, because excel file contains flat values but with doing this we can ensure it
vrp_solution_length = round(vrp_solution_length, 2)

# Print the tour
print('VRP solution found with savings heuristic starting from', origin, 'is: ')
for t in tours:
    innerTour = []
    for i in tours[t]:
        innerTour.append(i)
    print(innerTour)
    for index in range(len(tours[t])-1):
        arcs.append((int(tours[t][index]), int(tours[t][index+1])))

print('With a Total Traveling Distance of: ', vrp_solution_length)

# 2-opt Algorithm Part
def calculate_total_distance(route, d):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += d[int(route[i])][int(route[i + 1])]
    return total_distance

def two_opt(route, d):
    best_distance = calculate_total_distance(route, d)
    while True:
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1:
                    continue
                new_route = route[:]
                new_route[i:j] = route[j - 1:i - 1:-1]
                new_distance = calculate_total_distance(new_route, d)
                if new_distance < best_distance:
                    route = new_route
                    best_distance = new_distance      
                    break  # improvement found, restart the outer loop
            else:
                continue  # no improvement found, continue to next pair
            break  # improvement found, break the inner loop
        else:
            break  # no improvement found, break the outer loop
    return route


#Apply 2-opt optimization to each tour
for t in tours:
    tours[t] = two_opt(tours[t], d)
    
# Print the final tours
for t in tours:
    print("Tour:",tours[t])
    
arcs = []
for t in tours:
    for index in range(len(tours[t])-1):
        arcs.append((int(tours[t][index]), int(tours[t][index+1])))

#Compute the total traveling distance
vrp_solution_length = 0    #Total length initially determined as 0 
for tour in tours.values():   #For all tour lists, we used .values() and we reached to the list
    for i in range(len(tour) - 1):       #We included the one previous index of the tour
        vrp_solution_length += d[int(tour[i])][int(tour[i + 1])]     #We calculated the distances between the customer nodes in that tour, and we added to the vrp_solution_length 

# Round the result to 2 decimals to avoid floating point representation errors, it is not really necessary, because excel file contains flat values but with doing this we can ensure it
vrp_solution_length = round(vrp_solution_length, 2)

print('With a Total Traveling Distance of After 2-opt Algorithm: ', vrp_solution_length)


#Visualization Part
plt.figure(figsize=(15, 10)) #Size has been selected for visualization figure

for i in arcs:
  plt.plot([coorddf["x"][i[0]],coorddf["x"][i[1]]],[coorddf["y"][i[0]],coorddf["y"][i[1]]],c="g") #Arcs has been plotted by green color as you can see
for i in nodes:
  if i != 0:
    plt.text(coorddf["x"][i],coorddf["y"][i],i,fontdict=dict(color='black', alpha=0.5, size=16)) 
plt.text(coorddf["x"][0],coorddf["y"][0],'depot',fontdict=dict(color='black', alpha=0.5, size=16))
plt.plot(coorddf["x"][0],coorddf["y"][0],c="r",marker='s')
plt.scatter(coorddf["x"][1:],coorddf["y"][1:],c="b") #Coordinates x and y of the customer nodes has marked with the blue color
plt.title("Savings Algorithm Solution")



