# Last-mile-Deliveries-Using-Autonomous-Robots-Project
-The main objective of this project is to develop an optimisation-based solution methodology for Coffy, a coffee company operating on the Sabancı University campus, utilising autonomous robots for last-mile delivery operations.

-Diverse types of companies which are using electrical autonomous robots in their delivery process were investigated and some parameter values such as robot cargo capacity, cost of each robot, battery capacity of each robot, and energy consumption rate per kilometer were determined and calculated.

-The problem was looked from a routing viewpoint and deal with the problem as an extension of the well-known Vehicle Routing Problem (VRP) and integrated in the Sabanci University map.

-Provided dataset which includes 15 customer locations and depot, Latitude, Longitude, Demand, and Time Window values of those locations evaluated, and Haversine Distance calculation method was used to determine the distances between customer locations.

-Mixed integer linear programming model was developed by identifying the sets, parameters, decision variables, objective function and constraints with a goal of minimizing the total distance traveled and the total cost.

-Developed constraints ensured that each customer is visited exactly once, each route starts and ends at the depot, flow/balance, time window requirements, ensuring the vehicle capacity limitation, meeting customer demands and battery capacity of the electrical vehicles are not violated.

-Established mathematical model was coded by using Python as a programming language and Gurobi was used as an optimization solver tool in the project.

-The problem was modeled and solved within the Sabanci University map, and formed routes were visualized to provide better outcomes for the company.

-Additionally, Savings Heuristic was adjusted by considering the vehicle capacity, battery capacity, and time window restrictions of the clients.

-The developed Savings Heuristic was coded using the Python programming language.

-Adjusted Savings Heuristic Algorithm was modeled within the Sabanci University map and formed routes were visualized.

-The solution obtained by Mathematical Modelling and the solution obtained by Savings Heuristic were compared.

-Lastly, Savings Heuristic was tested on 10 large instances by using the Euclidean Distance calculation method.
