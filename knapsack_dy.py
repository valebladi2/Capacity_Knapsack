''' 
A Dynamic Programming based Python Program for 0-1 Knapsack problem Returns the maximum value that can be put in a knapsack of capacity W
'''
import sys
import math	
import numpy as np
from  collections.abc import Iterable
import os								# access operationg system dependent file paths

nodes = []
nodes_2 = []
### part 0: settings
filename = 'moabit_test4'							# select vrp file
weight_murmel = 0.17							# energy consumption kWh per km (MURMEL)
weight_mothership = 0.27						# energy consumption kWh per km (mothership)
energy_unload = 0							# energy consumption for dropping off load at mothership in kWh TODO
energy_compress  = 0
time_unload = 120/3600		#seconds-hr, whole process of opening and closing trash can 		
time_compress  = 300/3600 	#seconds-hr
speed_murmel= 1/3.24							# time per distance in h per km (MURMEL) based on 0.9 m/s maximum speed in Urbanek bachelor thesis
speed_mothership = 1/50						# time per distance in h per km (mothership)
time_emptying = 42.5/3600						# time to empty a dustbin in h (MURMEL) based on 42.5s in Merle Simulation
time_dropoff = 1/60							# time to empty Murmels load into mothership in h
energy_emptying = 0.0513 * time_emptying				# energy consumption for emptying a dustbin in kWh (MURMEL)
murmel_capacity = 100							# on average, a mothership visit is necessary every 'murmel_capacity's waypoint

def cost_murmel(dist):
	energy_cost = dist*weight_murmel + energy_unload + energy_compress
	time_cost = dist*speed_murmel + time_unload + time_compress
	return (energy_cost+time_cost,energy_cost,time_cost)

def file_oppening(filename):
	### part 1: get waypoints and draw them
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

def geographic_2d_distance(i, j):
	R = 6371  # Earth radius in kilometers
	dLat = math.radians(nodes[j,0] - nodes[i,0])
	dLon = math.radians(nodes[j,1] - nodes[i,1])
	lat1 = math.radians(nodes[i,0])
	lat2 = math.radians(nodes[j,0])
	return 2 * R * math.asin(math.sqrt(math.sin(dLat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dLon / 2)**2)) #converted in kilometers

def flatten(x):
    if isinstance(x, Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

def remove(t_val, t_wt, t_coor,t_nod):
	l_val = []
	l_wt = []
	l_coor = []
	t_nod = flatten(t_nod)#flatten the irregular list 
	for i in range (0,len(t_nod),2): #remove the visited nodes, so next search is based on the unvisted nodes 
		if [t_nod[i],t_nod[i+1]] in t_coor:
			n = t_coor.index([t_nod[i],t_nod[i+1]])
			l_val.append(t_val[n])
			l_wt.append(t_wt[n])
			l_coor.append(t_coor[n])
			t_coor.pop(n)
			t_val.pop(n)
			t_wt.pop(n)
	return (l_val,l_wt, l_coor)

def knapSack(W, wt, val, n,coor):
	K = [[0 for x in range(W + 1)] for x in range(n + 1)]
	node_k = [[[0,0] for x in range(W + 1)] for x in range(n + 1)]
	#point_m = [[[0,0] for x in range(W + 1)] for x in range(n + 1)]
	# Build table K[][] in bottom up manner
	for i in range(n+1):
		for w in range(W + 1):
			if i == 0 or w == 0:
				K[i][w] = 0
			elif wt[i-1] <= w:
				max_opt = (val[i-1] + K[i-1][w-wt[i-1]],  K[i-1][w])
				max_opt2 = ([coor[i-1],node_k[i-1][w-wt[i-1]]], node_k[i-1][w])
				#max_opt3 = ([point[i-1],point_m[i-1][w-wt[i-1]]], point_m[i-1][w])
				wt_temp = (wt[i-1], wt[i-i])
				K[i][w] = max(max_opt)
				node_k[i][w]= max_opt2[max_opt.index(max(max_opt))]
				#point_m[i][w] = max_opt3[max_opt.index(max(max_opt))]
			else:
				K[i][w] = K[i-1][w]
				node_k[i][w]= (node_k[i-1][w])
		l_val, l_wt, l_coor = remove(val, wt, coor,node_k[n][W])
	return (K[n][W], node_k[n][W]) #point_m[n][W],


if __name__ == "__main__":
	file_oppening(filename)
	print('#Debug info: input file nodes part:')
	for node in nodes:
		print ('#   ' + str(node))

	nodes_bins = np.array(nodes)[:,[3]].astype(int)
	nodes_points = np.array(nodes)[:,[0]]
	nodes = np.array(nodes)[:,[1,2]]
	dist = np.zeros((len(nodes), len(nodes)))
	costs = np.zeros((len(nodes), len(nodes)))
	costs_energy = np.zeros((len(nodes), len(nodes)))
	costs_time = np.zeros((len(nodes), len(nodes)))
	value = np.zeros((len(nodes), len(nodes)))
	for i in range(len(nodes)):
		for j in range(i):
			dist[i,j] = geographic_2d_distance(i,j)
			dist[j,i] = dist[i,j]
			costs[i,j], costs_energy[i,j],costs_time = cost_murmel(dist[i,j])
			costs[j,i] = costs[i,j]
			#think of a higher penalty than the fullness % since the greater 
			value[i,j] = ((nodes_bins[i]+nodes_bins[j])/sum(nodes_bins))/costs[i,j] #how much weight should I give istance over fullness
			value[j,i] = value[i,j]

	#nodes_2
	#nodes_3
	#print('####')
	#no1, no2 = nodes[0]
	#print(nodes[0], no1,no2)

	#
	# 
	# inital point 
	print (dist)
	print (costs)
	print (value)
	print (nodes_bins)
	print('-------------- start')
	input()
	inital_point = nodes[0]
	inital_cap = nodes_bins[0]
	inital_cost = costs[0]
	inital_value = value[0]
	print (murmel_capacity)
	print (inital_point, inital_cap, inital_cost, inital_value)
	print('-------------- initial')
	input()
	nodes =np.delete(nodes,0, axis=0)
	nodes_bins =np.delete(nodes_bins,0, axis=0)
	costs=np.delete(costs,0, axis=0)
	value=np.delete(value,0, axis=0)

	print(nodes)
	print (nodes_bins)
	print (costs)
	print (value)
	print('-------------- all')
	input()
	'''
	input()
	K = [[0 for x in range(murmel_capacity + 1)] for x in range(len(nodes)+ 1)]
	while inital_cap<=murmel_capacity:
		inital_cap = inital_cap+101
		print ('there is still capt:',inital_cap)
	input()
	'''

	nodes_bins= flatten(nodes_bins)
	print (nodes_bins)
	costs_current =costs[0].tolist()
	print (value)
	value =value[0].tolist()
	print (value)
	nodes_2 = nodes.tolist()
	print(nodes_2)
	print('-------------- initial')
	input()

	#run while there are som points left
	#make that if the limit of apcity is lower than the min cost, throw error
	while nodes_2: #while there is capcity or there is battery.
		n = len(nodes_bins)
		#l_val, l_wt, l_coor = remove(value, nodes_bins, nodes_2,inital_point)

		print(knapSack( murmel_capacity, nodes_bins,value,n,nodes_2))
		#with in knpasack we find the order of closest
		#()
		'''
		inital_point = nodes_2[0]
		inital_cap = nodes_bins[0]
		inital_cost = costs[0]
		inital_value = value[0]
		print (murmel_capacity)
		print (inital_point, inital_cap, inital_cost, inital_value)
		print('-------------- initial')
		input()
		nodes_2 =np.delete(nodes_2,0, axis=0)
		nodes_bins =np.delete(nodes_bins,0, axis=0)
		costs=np.delete(costs,0, axis=0)
		value=np.delete(value,0, axis=0)
		'''
		input()


  
