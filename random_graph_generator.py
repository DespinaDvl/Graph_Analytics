# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 13:57:58 2018

@author: vikkas
"""
##Program to generate random graph for testing

import networkx as nx
import random
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

def graph_generator(n,m):
    """
    n is the number of nodes and m is the number of edges connected to each new node
    """
    if m < 1 or  m >=n:
        print("Network must have m>=1 and m<n"%(m,n))
        return None
    
    G =nx.barabasi_albert_graph(n, m, seed=24)
    G = G.to_directed()
    
    for (u,v,w) in G.edges(data=True):
        w['weight'] = random.randint(1,20)
    
    nx.draw(G,with_labels = True, with_edges=True)
    plt.axis('off')
    plt.show()
    nx.write_weighted_edgelist(G, 'test.weighted.edgelist')
    
    with open('test.weighted.edgelist', 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write('A'+'\n'+str(G.order())+ '\n'+ content)
    import os; os.system('cls')
    print("A graph with nodes = ",n," and edges = ",G.number_of_edges()," has been downloaded to file with the name test.weighted.edgelist as arc matrics")
    return G

# main program starts here    
import os; os.system('cls')
print("This Program will generate a random directed Graph using Barbasi Albert Method with weights ranging between 1-20\n\n")
print("Please enter the Number of nodes required: ",end="")
n=int(input())
print("\n\nPlease enter the Number of Edges connected to each new node: ",end="")
m=int(input())


graph_generator(n,m)


