''' 
A Dynamic Programming based Python Program for 0-1 Knapsack problem Returns the maximum value that can be put in a knapsack of capacity W
'''

import math	
import numpy as np
from  collections.abc import Iterable


'''
if use_linear_distance:
        costs = np.zeros((len(nodes), len(nodes)))
        for i in range(len(nodes)):
            for j in range(i):
                if edge_weight == EdgeWeight.EUCLIDEAN:
                    costs[i,j] = euclidean_2d_distance(i,j)
                if edge_weight == EdgeWeight.GEOGRAPHIC:
                    costs[i,j] = geographic_2d_distance(i,j)
                costs[j,i] = costs[i,j]
        print (costs)
        input()
        #######
'''

def geographic_2d_distance(i, j):
	R = 6371  # Earth radius in kilometers

	dLat = math.radians(nodes[j,0] - nodes[i,0])
	dLon = math.radians(nodes[j,1] - nodes[i,1])
	lat1 = math.radians(nodes[i,0])
	lat2 = math.radians(nodes[j,0])
	return 2 * R * math.asin(math.sqrt(math.sin(dLat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dLon / 2)**2))

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
    node = [[[0,0] for x in range(W + 1)] for x in range(n + 1)]
    # Build table K[][] in bottom up manner
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i-1] <= w:
                max_opt = (val[i-1] + K[i-1][w-wt[i-1]],  K[i-1][w])
                max_opt2 = ([coor[i-1],node[i-1][w-wt[i-1]]], node[i-1][w])
                K[i][w] = max(max_opt)
                #print (val[i-1],K[i-1][w-wt[i-1]])
                #print (coor[i-1],node[i-1][w-wt[i-1]])
                node[i][w]= max_opt2[max_opt.index(max(max_opt))]
            else:
                K[i][w] = K[i-1][w]
                node[i][w]= (node[i-1][w])
    print (node)
    #l_val, l_wt, l_coor = remove(val, wt, coor,node[n][W])
    return (K[n][W], node[n][W])

def multi_knapSack(n_murmels, W, wt, val,coor):
    tour_global = []
    visited_nodes = []
    murmel_capacity = W
    current = 0
    val_n  = 0
    for i in range (0,n_murmels):
        tour_local = []
        #current = point
        val_n, nod = (knapSack(W, wt, val, len(coor),coor))
        l_val, l_wt, l_coor = remove(val, wt, coor,nod)
        tour_local.append(l_coor)
        tour_global.extend(tour_local)
    print (tour_global)
    return (val_n, nod)

if __name__ == "__main__":
    # Driver program to test above function
    val = [1,2,3,1,2,3,1,2,3] #I need to create a new cost function that involves the
    wt =  [1,1,1,1,1,1,1,1,1]
    coor = [[0,1],[1,1],[1,0],[0,2],[2,3],[2,0],[0,3],[3,3],[3,0]]
    #calculate the cost and 
    nodes_2 = [[0,1],[1,1],[1,0],[0,3],[2,3],[2,0],[0,3],[3,3],[3,0]] #need to replicate since pop, removes it from all instances
    W =5
    n = len(val)
    n_murmels = 3
    print(knapSack( W, wt, val,n,coor))
    #print (multi_knapSack(n_murmels, W, wt, val,coor))

  
