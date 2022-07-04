import sys
import math	
import numpy as np
from  collections.abc import Iterable	#flatten lists
import os								# access operationg system dependent file paths
import matplotlib.pyplot as plt					# plotting solution process and graphs
import matplotlib							# plotting solution process and graphs

''' 
Dynamic Knapsack Problem, returns the maximum value that can be put in a knapsack of capacity W given weight and value
'''

## part 0: settings
filename = 'moabit_test4'						# select vrp file
show_gui = True
### MURMEL
weight_murmel_loc = 0.095						# energy consumption mobility of MURMEL
weight_murmel_bin = 0.017						# energy consumption per bin
weight_murmel_cst = 0							#(weight_murmel_loc + weight_murmel_bin)*30/70, energy consumption of constant varibles (equipment)				
speed_murmel= 1/3.24							# time per distance in h per km (MURMEL) based on 0.9 m/s maximum speed in Urbanek bachelor thesis
murmel_capacity = 50						# on average, a mothership visit is necessary every 'murmel_capacity's waypoint
#### Time
time_unload = 120/3600							#seconds-hr, whole process of opening and closing trash can 		
time_compress  = 300/3600 						#seconds-hr
time_emptying = 42.5/3600						# time to empty a dustbin in hr (MURMEL) based on 42.5s in Merle Simulation
##### Energy
energy_unload = 0.5								# energy consumption for dropping off load at mothership in kWh TODO
energy_compress  = 0.5							# TODO
energy_emptying = 0.0513 * time_emptying		# energy consumption for emptying a dustbin in kWh (MURMEL)
### MOTHERSHIP
weight_mothership = 0.27						# energy consumption kWh per km (mothership)
speed_mothership = 1/50							# time per distance in h per km (mothership)
time_dropoff = 1/60								# time to empty Murmels load into mothership in hr
energy_dropoff = 0								# TODO
battery_capacity = 960							# capacity of battery in kWh

## part 0.1: globla vairables 
nodes = []
value = []
dist = []
num_visited = [0]
num_visited_cap = [[0]]
column_width = 3.30

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


### part 1: get waypoints and draw them
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

def cost_murmel_distance(dist):
	energy_cost = dist*weight_murmel_loc
	time_cost = dist*speed_murmel
	return (energy_cost, time_cost)

def cost_murmel_compressing(bins_empied):
	#add cost of ms and murmel traveling to get together (decide on how will it be) 
	energy_cost = weight_murmel_bin*bins_empied
	time_cost =  (time_unload + time_compress + time_emptying)*bins_empied
	return (energy_cost,time_cost)

def cost_murmel_swap_battery(dist):
	return dist

def cost_mothership(dist):
	energy_cost = dist*weight_mothership+energy_dropoff
	time_cost = dist*speed_mothership+time_dropoff
	return (energy_cost+time_cost,energy_cost,time_cost) 

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

def calculate_final(points,capacity):
	# calculate the final cost energy, time and distance 
	final_route = []
	f_dist_cost = 0
	d_energy_cost_loc = 0
	d_time_cost_loc = 0
	t_nodes_coor = nodes_coor.tolist()
	# if SD did not exists
	t_cap = []
	temp_cap = 0
	temp_cap_2 = 0
	for i in points:
		final_route.append(t_nodes_coor[i])
		#if SD did not exists
		t_cap.append(capacity[i])
	for i in range (0,len(points)-1):
		f_dist_cost +=  (dist[points[i], points[i+1]])
		a = dist[points[i], points[i+1]]
		b,c = cost_murmel_distance(a)
		d_energy_cost_loc += b
		d_time_cost_loc += c
		#print (t_cap)
		#print (points[i], points[i+1])
		#if SD did not exists
		#print (t_cap[i], t_cap[i+1])
		temp_cap += t_cap[i]  + t_cap[i+1]
		if temp_cap>=100:
			temp_cap_2 += 1
			temp_cap = 0
	#if SD did not exists
	#given that the at needs to be emptied 
	temp_cap_2 = temp_cap_2+1
	energy_cost_bin, time_cost_bin = cost_murmel_compressing(temp_cap_2)
	print ('---------')
	print (d_energy_cost_loc,energy_cost_bin,d_time_cost_loc,time_cost_bin)
	weight_murmel_cst = (d_energy_cost_loc + energy_cost_bin)*30/70
	f_energy_cost_bins = d_energy_cost_loc+energy_cost_bin+weight_murmel_cst
	f_time_cost_bins = d_time_cost_loc+time_cost_bin
	#f_energy_cost_bins += (energy_unload + energy_compress + energy_emptying) * (temp_cap_2) #TODO compress tie and energy is times the acutal garbage collected
	#f_time_cost_bins   += (time_unload + time_compress + time_emptying)       * (temp_cap_2)
	return (temp_cap_2,f_energy_cost_bins,f_time_cost_bins,f_dist_cost,final_route)

def gui(f_route):
	f_route = np.array(f_route)
	x = f_route[:,1] #lon
	y = f_route[:,0] #lat
	plt.xlabel("Longitud")
	plt.ylabel("Latitud")
	plt.scatter(x, y)
	plt.plot(x,y)
	plt.show()

if __name__ == "__main__":
	file_oppening(filename)
	print('#Debug info: input file nodes part:')
	for node in nodes:
		print ('#   ' + str(node))
	# get nodes specifications in different arrays
	nodes_num = np.array(nodes)[:,[0]]
	nodes_coor = np.array(nodes)[:,[1,2]]
	t_nodes_bins_cap = np.array(nodes)[:,[3]].astype(int)
	nodes_bins_cap = np.full((len(t_nodes_bins_cap),1), 50, dtype=int)
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
			value[i,j] = ((nodes_bins_cap[i] +nodes_bins_cap[j])/sum(nodes_bins_cap))/dist[i,j] #the value it is added given its collection
			value[j,i] = ((nodes_bins_cap[i] +nodes_bins_cap[j])/sum(nodes_bins_cap))/dist[j,i] 

	# change from np.array to lists 
	nodes_num_2 = nodes_num.tolist()
	nodes_coor_2 = nodes_coor.tolist()
	t_nodes_bins_cap = flatten(t_nodes_bins_cap)
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
	#num_visited = [{0,3,2,4,6,5,8,7,10,9,11,12,13,14,15,16,17,18,19,20,21,23,22,24,123,124,25,125,27,26,28,29,30,34,35,36,37,38,39,33,32,31,47,48,44,42,40,41,43,45,46,120,122,121,50,49,53,51,52,54,55,59,57,58,56,60,63,64,65,61,66,67,62,68,71,72,73,75,76,77,78,80,79,81,74,70,69,86,85,87,97,88,96,84,89,94,90,93,91,92,100,98,101,99,102,103,104,106,105,107,108,110,111,109,113,112,114,116,118,115,117,119,179,181,180,182,183,184,185,186,187,189,188,190,178,177,176,175,174,173,171,172,170,169,168,167,166,165,163,164,143,142,141,144,145,146,152,153,151,148,150,147,149,161,160,159,155,158,162,157,156,154,133,139,132,131,130,129,136,135,137,134,138,140,82,83,95,128,127,126,1,191,192,210,211,212,213,209,208,207,204,202,206,201,200,199,198,203,196,197,195,194,193,205]
	#t_nodes_bins_cap = [0,64,18,14,70,78,72,95,81,67,96,69,84,88,65,79,43,30,22,14,53,52,57,19,6,2,29,75,74,45,46,39,8,90,92,14,14,58,55,96,93,52,41,5,11,84,18,37,20,7,56,100,14,15,48,12,100,23,61,7,60,61,46,94,36,33,49,5,2,70,18,43,78,72,51,91,78,77,93,50,64,6,59,58,83,70,5,6,65,51,74,25,72,78,28,79,69,55,3,21,48,11,59,64,71,12,54,4,65,20,4,6,83,58,62,100,4,31,87,8,3,29,38,25,28,33,47,31,89,25,31,69,6,18,65,71,6,15,32,23,24,56,87,10,64,98,27,21,34,5,66,50,5,95,19,63,72,57,66,40,90,93,84,5,52,56,5,64,56,47,100,19,70,97,96,32,27,76,84,24,92,42,6,31,39,5,69,46,38,88,11,59,97,77,41,9,1,53,19,98,99,40,4,99,56,8,47,84,53,64,95,11,63,82]
	f_cap, f_energy,f_time,f_distance, f_route = calculate_final(num_visited,t_nodes_bins_cap)

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
#1 complete the total cost of the path
#2 add mother ship
#3 calibrate cost function to get a good cost-benefit trade off
# make a conection between the price we have for energy and time so we can correlate energy and time 
#4 add a cost function for the battery

