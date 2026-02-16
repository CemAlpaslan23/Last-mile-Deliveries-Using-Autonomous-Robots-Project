import pandas as pd
import numpy as np
from math import sqrt
import gurobipy
from gurobipy import GRB, Model, quicksum
import geopy.distance
import matplotlib.pyplot as plt
import math

# Function of Haversine distance calculation formula    
def geodesic(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers

    # convert decimal degrees to radians
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    # Apply Haversine Formula
    a = math.sin(dLat / 2)**2 + math.sin(dLon / 2)**2 * math.cos(lat1) * math.cos(lat2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    return distance


df = pd.read_excel("Robot Test Data.xlsx",index_col=0) #Reading the dataset

xcoord = list(df['Latitude']) #Reading X-coordinate from the dataset
ycoord = list(df['Longitude']) #Reading Y-coordinate from the dataset
q = list(df['demand']) #Reading Demand from the dataset
e = list(df['Earliest']) #Reading Earliest start time from the dataset
l = list(df['Latest']) #Reading Latest start time from the dataset
s = list(df['Service Time']) #Reading Service time from the dataset

#Haversine distance calculation, Dij parameter
C_geo = [[geodesic(xcoord[i],ycoord[i],xcoord[j],ycoord[j]) for j in range(len(xcoord))] for i in range(len(xcoord))]

M = 1e10  #Big M
C = 10  #Vehicle, cargo capacity parameter
# A = len(xcoord)  #Number of nodes, the set
A = 16
F = 2752  #Cost of each robot, setup cost + maintanence cost + delivery cost per robot, 2250 + 500 + 2
Q = 1260  #Battery capacity of each robot
h = 525  #Energy consumed to traverse arc(i,j),it is the consumption per km, 1260 (Robot Battery Capacity) / Robot consumption rate per km: 2.4 km (it is the range of the robot the diameter area which means that it can travel at most 2.4 km with fully charged) = 525
t = [[C_geo[i][j] / 6 for j in range(A)] for i in range(A)] #Time calculation,Tij parameter, distance / robot speed(6 km/h)


EVRPTW = gurobipy.Model()


EVRPTW.setParam('MIPFocus', 3)
EVRPTW.setParam('MIPGapAbs', 100)
EVRPTW.setParam('TimeLimit', 60)
 

#Create the decision variables
x = EVRPTW.addVars(A,A,vtype=GRB.BINARY,name=['x'+str(i)+','+str(j) for i in range(A) for j in range(A)])
b = EVRPTW.addVars(A,lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name=['b'+str(i) for i in range(A)])
u = EVRPTW.addVars(A,lb=0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name=['u'+str(i) for i in range(A)])
y = EVRPTW.addVars(A,lb=0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name=['y'+str(i) for i in range(A)])


#Create the objective function
EVRPTW.setObjective(quicksum(C_geo[i][j] * x[i,j] for i in range(A) for j in range(A)) + F * (quicksum(x[0,i] for i in range(A))),GRB.MINIMIZE)

#Add the constraints
EVRPTW.addConstrs(quicksum(x[i,j] for j in range(A) if i != j) == 1 for i in range(1,A))

EVRPTW.addConstrs(quicksum(x[i,j] for i in range(A) if i != j) == 1 for j in range(1,A))

EVRPTW.addConstrs(b[i] + s[i] + t[i][j] - M * (1-x[i,j]) <= b[j] for i in range(A) for j in range(1,A))

EVRPTW.addConstrs(b[j] >=e[j] for j in range(A)) #A

EVRPTW.addConstrs(b[j] <=l[j] for j in range(A)) #A

EVRPTW.addConstrs(u[j] >= q[j] for j in range(A)) #A

EVRPTW.addConstrs(u[j] <= C for j in range(A)) #A

EVRPTW.addConstrs(u[j] <= (u[i] - (q[i] * x[i,j]) + (C * (1-x[i,j]))) for i in range(A) for j in range(1,A) if i!=j) #A, 1,A

EVRPTW.addConstr(u[0] >= 0) 

EVRPTW.addConstr(u[0] <= C) #Çok gerekli değil

EVRPTW.addConstr(quicksum(x[0,j] for j in range(1,A)) >= 1) #Depodan çıkan araç sayısı 1 den büyük olsun

EVRPTW.addConstrs(y[j] >= 0 for j in range(1,A)) #Çok gerekli değil #A

EVRPTW.addConstrs(y[j] <= (y[i] - (h * C_geo[i][j] * x[i,j]) + Q * (1 - x[i,j])) for i in range(A) for j in range(1,A) if i!=j)

EVRPTW.addConstr(y[0] == Q)

EVRPTW.addConstrs(y[i] >= quicksum(h * C_geo[i][j] * x[i,j] for j in range(1,A)) for i in range(A)) # A, (1,A)


#Finding the solution
EVRPTW.update()
EVRPTW.optimize()


status = EVRPTW.status
object_Value = EVRPTW.objVal

print()
print("Model status is: ", status)
print()
print("Objective Function value is: ", object_Value)


#Print decision varibales which are not zeros
if status !=3 and status != 4:
    for v in EVRPTW.getVars():
        if EVRPTW.objVal < 1e+99 and v.x!=0:
            print('%s %f'%(v.Varname,v.x))
            
            
        
#For visualization part
xcoord = list(df['Latitude']) #Reading X-coordinate from the dataset
ycoord = list(df['Longitude']) #Reading Y-coordinate from the dataset


nodes = list(range(df.shape[0]))
arcs = []
for i in range(A):
    for j in range(A):
        if x[i,j].X == 1:
            arcs.append((i,j))
          
print(arcs)

plt.figure(figsize=(15, 10)) #Size has been selected for visualization figure

for i in arcs:
  plt.plot([xcoord[i[0]],xcoord[i[1]]],[ycoord[i[0]],ycoord[i[1]]],c="g") #Arcs has been plotted by green color as you can see
for i in nodes:
  if i != 0:
    plt.text(xcoord[i],ycoord[i],i,fontdict=dict(color='black', alpha=0.5, size=16)) 
plt.text(xcoord[0],ycoord[0],'depot',fontdict=dict(color='black', alpha=0.5, size=16))
plt.plot(xcoord[0],ycoord[0],c="r",marker='s')
plt.scatter(xcoord[1:],ycoord[1:],c="b") #Coordinates x and y of the customer nodes have marked with the blue color
plt.title("Mathematical Formulation Solution")
