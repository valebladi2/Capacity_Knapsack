import sys
import math	
import numpy as np
from  collections.abc import Iterable	#flatten lists
import os								# access operationg system dependent file paths
import matplotlib.pyplot as plt					# plotting solution process and graphs

''' 
Dynamic Knapsack Problem, returns the maximum value that can be put in a knapsack of capacity W given weight and value
'''

# part 0: settings
filename = 'moabit_test4'						# select vrp file
show_gui = True
# part 1: MURMEL energy and time
## part 1.1: murmel energy
energy_murmel_loc = 0.095						# energy consumption mobility of MURMEL
energy_murmel_bin = 0.017						# energy consumption per bin
weight_murmel_cst = 0							#(energy_murmel_loc + energy_murmel_bin)*30/70, energy consumption of constant varibles (equipment)	
speed_murmel= 1/3.24							# time per distance in h per km (MURMEL) based on 0.9 m/s maximum speed in Urbanek bachelor thesis
murmel_capacity = 100							# on average, a mothership visit is necessary every 'murmel_capacity's waypoint
## part 1.2: murmel time
time_adjust_bin = 120/3600							#seconds-hr, whole process of opening and closing trash can 		
time_compress  = 300/3600 						#seconds-hr
#time_emptying = 42.5/3600						# time to empty a dustbin in hr (MURMEL) based on 42.5s in Merle Simulation

# part 2: MOTHERSHIP energy and time
## part 2.1: MOTHERSHIP energy
energy_mothership = 0.27						# energy consumption kWh per km of MOTHERSHIP
#energy_swapping_battery = 0.01					# energy consumption per battery swap between MS and MM TODO: get this from Abhi
#energy_unloading_trash_battery = 0.01			# energy consumption per complete unload between MM and MS TODO: get this from Abhi
speed_mothership = 1/50							# time per distance in h per km (mothership)
## part 2.2: MOTHERSHIP time
time_dropoff = 1/60								# time to empty Murmels load into mothership in hr
#time_swapping_battery = 0.01					# energy consumption per battery swap between MS and MM TODO: get this from Abhi
#time_unloading_trash_battery = 0.01			# energy consumption per complete unload between MM and MS TODO: get this from Abhi

## part 0.1: global vairables 
nodes = []
value = []
dist = []
num_visited = [0]
num_visited_cap = [[0]]
fsize=10
params = {'legend.fontsize': fsize*0.8,
          'axes.labelsize': fsize*0.9,
          'axes.titlesize': fsize,
          'xtick.labelsize': fsize*0.8,
          'ytick.labelsize': fsize*0.8,
          'axes.titlepad': fsize*1.5,
          'font.family': 'serif',
          'font.serif': ['cmr10']}
plt.rcParams.update(params)

### part 1: open file and get waypoints
def file_oppening(filename):
	try:
		with open(filename + '.tsp', 'r') as tsp_file:
			tsp_file_data = tsp_file.readlines()
	except Exception as e:
		print('error!\nExiting..') # more exception details: str(e)
		sys.exit()
	# possible entries in specification part:
	specification_list = ['NAME', 'TYPE', 'COMMENT', 'DIMENSION', 'CAPACITY', 'GRAPH_TYPE', 'EDGE_TYPE', 'EDGE_WEIGHT_TYPE', 'EDGE_WEIGHT_FORMAT', 'EDGE_DATA_FORMAT', 'NODE_TYPE', 'NODE_COORD_TYPE', 'COORD1_OFFSET', 'COORD1_SCALE', 'COORD2_OFFSET', 'COORD2_SCALE', 'COORD3_OFFSET', 'COORD3_SCALE', 'DISPLAY_DATA_TYPE']
	specification = [None] * len(specification_list)
	node_data = False
	for data_line in tsp_file_data:
		data_line = data_line.replace('\n', '')
		if node_data:
			node = data_line.split()
			if len(node) == 4:
				try:
					node[0], node[1], node[2] ,node[3] = int(node[0]), float(node[1]), float(node[2]),int(node[3])
					nodes.append(node)
				except Exception as e: # not expected data format; try to continue parsing
					node_data = False
			else:
				node_data = False
		for i in range(len(specification_list)):
			if data_line.find(specification_list[i] + ': ') == 0:
				specification[i] = data_line.replace(specification_list[i] + ': ', '')
		if (data_line.find('NODE_COORD_SECTION') == 0):
			node_data = True

def flatten(x):
    if isinstance(x, Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]
		
def geographic_2d_distance(i, j,nodes_coor):
	R = 6371  # Earth radius in kilometers
	dLat = math.radians(nodes_coor[j,1] - nodes_coor[i,1])
	dLon = math.radians(nodes_coor[j,0] - nodes_coor[i,0])
	lat1 = math.radians(nodes_coor[i,1])
	lat2 = math.radians(nodes_coor[j,1])
	return 2 * R * math.asin(math.sqrt(math.sin(dLat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dLon / 2)**2)) #converted in kilometers

# part 1.2: MM energy and time cost
def cost_murmel_distance(dist):
	energy_cost = dist*energy_murmel_loc
	time_cost = dist*speed_murmel
	return (energy_cost, time_cost)

def cost_murmel_compressing(bins_empied):
	#add cost of ms and murmel traveling to get together (decide on how will it be) 
	energy_cost = energy_murmel_bin*bins_empied
	time_cost = 0# (time_unload + time_compress + time_emptying)*bins_empied
	return (energy_cost,time_cost)

# part 1.2: MS energy and time cost
def cost_mothership(dist):
	energy_cost = dist*energy_mothership
	time_cost = dist*speed_mothership+time_dropoff
	return (energy_cost+time_cost,energy_cost,time_cost) 

# part 2.1: update listed after node is visited
def update(t_val, t_wt, t_coor,t_nod,t_point):
	l_val = []
	l_wt = []
	l_coor = []
	l_point = []
	#flatten the irregular list 
	t_nod = flatten(t_nod)
	#remove the visited nodes, so next search is based on the unvisted nodes 
	for i in range (0,len(t_nod),2):
		if [t_nod[i],t_nod[i+1]] in t_coor:
			n = t_coor.index([t_nod[i],t_nod[i+1]])
			#have them in the visited lists
			l_val.append(t_val[n])
			l_wt.append(t_wt[n])
			l_coor.append(t_coor[n])
			l_point.append(t_point[n])
			#pop elements from the lists
			t_val.pop(n)
			t_wt.pop(n)
			t_coor.pop(n)
			t_point.pop(n)
	#sort coor from max to min value
	l_point = flatten(l_point)
	values_t = []
	if l_val and l_point:
		num_1 = []
		sort_val = np.argsort(l_val)
		for i in range (0,len(sort_val)):
			num_1.append([sort_val[i],l_point[i]])
		num_1= sorted(num_1, key=lambda x:x[0],reverse=True)
		a = []
		for num in num_1:
			a.append(int(num[1]))
			num_visited.append(int(num[1]))
		num_visited_cap.append(a)
		values_t = np.delete(value, num_visited, 1) # last arg colum 1 / row 0
	#return the value list of the visited node
	return (num_visited,values_t)

# part 2: coptimize path between value and cost
def knapSack(W, wt, val, n,coor, point_num):
	# build matrices for dynamic program
	K = [[0 for x in range(W + 1)] for x in range(n + 1)]
	coor_k = [[[0,0] for x in range(W + 1)] for x in range(n + 1)]
	# max capacity W given the weight and added value of the bins 
	for i in range(n+1):
		for w in range(W + 1):
			if i == 0 or w == 0:
				K[i][w] = 0
			elif wt[i-1] <= w:
				max_opt = (val[i-1] + K[i-1][w-wt[i-1]],  K[i-1][w])
				max_opt2 = ([coor[i-1],coor_k[i-1][w-wt[i-1]]], coor_k[i-1][w])
				K[i][w] = max(max_opt)
				coor_k[i][w]= max_opt2[max_opt.index(max(max_opt))]
			else:
				K[i][w] = K[i-1][w]
				coor_k[i][w]= (coor_k[i-1][w])
		n_point, values = update(val, wt, coor,coor_k[n][W],point_num)
	return (n_point,values) 

# part 3: calculate final cost and energy of selected path
def calculate_final(points_cap,points,capacity):
	# calculate the final cost energy, time and distance 
	final_route = []
	f_dist_cost = 0
	d_energy_cost_loc = 0
	d_time_cost_loc = 0
	f_cap = len(points_cap)-1
	t_nodes_coor = nodes_coor.tolist()
	for i in points:
		final_route.append(t_nodes_coor[i])
	for i in range (0,len(points)-1):
		f_dist_cost +=  (dist[points[i], points[i+1]])
		a = dist[points[i], points[i+1]]
		b,c = cost_murmel_distance(a)
		d_energy_cost_loc += b
		d_time_cost_loc += c
	energy_cost_bin, time_cost_bin = cost_murmel_compressing(f_cap)
	print ('---------')
	print (d_energy_cost_loc,energy_cost_bin,d_time_cost_loc,time_cost_bin)
	weight_murmel_cst = (d_energy_cost_loc + energy_cost_bin)*30/70
	f_energy_cost_bins = d_energy_cost_loc+energy_cost_bin+weight_murmel_cst
	f_time_cost_bins = d_time_cost_loc+time_cost_bin
	#f_energy_cost_bins += (energy_unload + energy_compress + energy_emptying) * (len(points_cap)-1) #TODO compress tie and energy is times the acutal garbage collected
	#f_time_cost_bins   += (time_unload + time_compress + time_emptying)       * (len(points_cap)-1)
	return (f_cap,f_energy_cost_bins,f_time_cost_bins,f_dist_cost,final_route)

# part 4: draw path planning
def gui(f_route):
	f_route = np.array(f_route)
	print ( f_route[:,1])
	print ( f_route[:,0])
	x = f_route[:,1] #lon
	y = f_route[:,0] #lat
	plt.xlabel("Longitud")
	plt.ylabel("Latitud")
	plt.scatter(x, y)
	plt.plot(x,y)
	plt.show()

#if __name__ == "__main__":
file_oppening(filename)
print('#Debug info: input file nodes part:')
for node in nodes:
	print ('#   ' + str(node))

# get nodes specifications in different arrays
nodes_num = np.array(nodes)[:,[0]]
nodes_coor = np.array(nodes)[:,[1,2]]
nodes_bins_cap = np.array(nodes)[:,[3]].astype(int)
# generate cost and value functions for murmel
dist = np.zeros((len(nodes_coor), len(nodes_coor)))
#costs = np.zeros((len(nodes_coor), len(nodes_coor)))
value = np.zeros((len(nodes_coor), len(nodes_coor)))
for i in range(len(nodes_coor)):
	for j in range(i):
		dist[i,j] = geographic_2d_distance(i,j,nodes_coor)
		dist[j,i] = dist[i,j]
		#costs[j,i] = cost_murmel(dist[j,i])
		#costs[i,j] = costs[j,i]
		value[i,j] = ((nodes_bins_cap[i]+nodes_bins_cap[j])/sum(nodes_bins_cap))/dist[i,j] #the value it is added given its collection
		value[j,i] = ((nodes_bins_cap[i]+nodes_bins_cap[j])/sum(nodes_bins_cap))/dist[j,i]  #TODO think if it should it be dist or cost

# change from np.array to lists 
nodes_num_2 = nodes_num.tolist()
nodes_coor_2 = nodes_coor.tolist()
nodes_bins_cap_2= flatten(nodes_bins_cap)
#eliminate origin point of MURMEL, I could call update function
nodes_coor_2 = nodes_coor_2[1:]
nodes_num_2 = nodes_num_2[1:]
nodes_bins_cap_2 = nodes_bins_cap_2[1:]
#get the value function from initial point
value_current =value[0].tolist()
value_current = value_current[1:]
'''
print (nodes_coor_2)
print (nodes_num_2)
print (nodes_bins_cap_2)
print (value)
print (nodes_coor)
input()
'''

#make that if the limit of apcity is lower than the min cost, throw error
while nodes_coor_2: #while there is capcity or there is battery.
	n = len(nodes_bins_cap_2)
	#have a condition that if they are unequal that we stop the test
	#print ( murmel_capacity, nodes_bins_cap_2,value_current,n,nodes_coor_2, nodes_num_2)
	#print (murmel_capacity,len(nodes_bins_cap_2),len(value_current),n,len(nodes_coor_2),len(nodes_num_2))
	num_visited,c_values = knapSack( murmel_capacity, nodes_bins_cap_2,value_current,n,nodes_coor_2, nodes_num_2)
	#chnage value list to last point
	value_current = c_values[num_visited[-1]].tolist()
#calculate final path on energy time and distance
f_cap, f_energy,f_time,f_distance, f_route = calculate_final(num_visited_cap,num_visited,nodes_bins_cap)

print('Visited Ndes: ',num_visited)
print ('Visited Ndes with cap: ',num_visited_cap)
print ('Number of dustbins:', len(num_visited))
print ('Emptying times: ',f_cap)
print ('Energy in KWhs: ',f_energy)
print ('Time in hrs: ', f_time)
print ('Distance in km: ',f_distance) #, f_route)
print ('Finished')

if show_gui:
	gui(f_route)


#TODO Next
#1 complete the total cost of the path DONE, find a better cost function which will help optimize the path
#1.1 add battery restriction and time of changing, choose threshold gor this 10% min battery.
#2 add mother ship, notes (do three cases and see which is better)
#3 calibrate cost function to get a good cost-benefit trade off done , good results
# make a conection between the price we have for energy and time so we can correlate energy and time 
#4 have the base case, which is having 50% and stops at every 2nd bin 