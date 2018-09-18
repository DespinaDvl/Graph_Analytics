# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 07:38:49 2018

@author: vikkas
"""
import os.path
import os
import warnings
warnings.filterwarnings("ignore")
import networkx as nx
import collections
import matplotlib.pyplot as plt
import itertools

def file_input():
    '''Function to take graph file input from user'''
    i = 1
    while i == 1:
        import os; os.system('cls' if os.name == 'nt' else 'clear')
        print("Please enter the graph file path to load or 1 to exit: ",end="")
        path=input()
        # checking if the given file exists in the directory given by user    
        try:
            if int(path) == 1:
          #      print('i am in if')
                os._exit(1)
        except:
            pass
    
        print("\nYou entered: ",path)
        
        if os.path.isfile(path) :
            pass
            i = 2  
        else:
            print("\nfile doesn't exists, please try again")
            file_input()

    return path

def take_input(sparse_value):
    '''Function takes the input from user for the root node of min spanning tree 
    and for the method to be used'''
    
    l = 1 # variable for while loop
    # taking the root node input from user and checking if the given node is present in the loaded graph        

    while l == 1:
        print('Enter the Start/root node to find the Minimum Spanning Tree: ',end="")
        start_node = input()
        if start_node in K.nodes():
            l=2
        else:
            print('Node {} not present on graph loaded please enter again'.format(start_node))
        m = 1 # variable for next while loop
    # asking user to select a method or use our recommendation 
    while m ==1:
        print('\n\nEnter the method to use for finding Minimum Spanning Tree :\n')
        print('1. An array method (Dijkstra method)')
        print('2. A Heap Method')
        if sparse_value == "dense":
            print('3. As the given graph is dense we recommend Heap Method (Recommendation)')
        else:
            print('3. As the given graph is Sparse we recommend Array Method (Recommendation)')
        print('Enter your choice :',end="")
        choice = input()
        try:
            choice = int(choice)
            if choice <= 3:
                m=2
                if choice == 1:
                    method = "array"
                elif choice == 2:
                    method = "heap"
                else:
                    if sparse_value =="dense":
                        method = "heap"
                    else:
                        method = "array"
            else:
                print('\nPlease choose correct option, Try again')
        except:
            print('\nPlease enter correct value, Try again')
            
    #m = 1
    #while m == 1:
    #    print('Enter the End node to find the shortest path: ',end="")
    #    end_node = input()
    #    if end_node in K.nodes():
    #        m=2
    #    else:
    #        print('Node {} not present on graph loaded please enter again'.format(end_node))
    return start_node, method


def _siftdown(heap_list, strt, pos):
    newitem = heap_list[pos]
    # Follow the path to the root, moving parents down until finding a place
    # newitem fits.
    while pos > strt:
        prntpos = (pos - 1) >> 1
        prnt = heap_list[prntpos]
        if newitem < prnt:
            heap_list[pos] = prnt
            pos = prntpos
            continue
        break
    heap_list[pos] = newitem

def _siftup(heap_list, pos):
    endpos = len(heap_list)
    strtpos = pos
    newitem = heap_list[pos]
    # Bubble up the smaller child until hitting a leaf.
    childpos = 2*pos + 1    # leftmost child position
    while childpos < endpos:
        # Set childpos to index of smaller child.
        rightpos = childpos + 1
        if rightpos < endpos and not heap_list[childpos] < heap_list[rightpos]:
            childpos = rightpos
        # Move the smaller child up.
        heap_list[pos] = heap_list[childpos]
        pos = childpos
        childpos = 2*pos + 1
    # The leaf at pos is empty now.  Put newitem there, and bubble it up
    # to its final resting place (by sifting its parents down).
    heap_list[pos] = newitem
    _siftdown(heap_list, strtpos, pos)
    
def heapify(x_list):
    """Transform list into a heap, in-place, in O(len(x)) time."""
    n = len(x_list)
    # Transform bottom-up.  The largest index there's any point to looking at
    # is the largest with a child index in-range, for i in reversed(range(n//2)):
    for i in reversed(range(n//2)):
        _siftup(x_list, i)

def heappop(heap_list):
    """Pop the smallest item off the heap, maintaining the heap invariant."""
    lastelt = heap_list.pop()    # raises appropriate IndexError if heap is empty
    if heap_list:
        returnitem = heap_list[0]
        heap_list[0] = lastelt
        _siftup(heap_list, 0)
        return returnitem
    return lastelt

def heappush(heap_list, item):
    """Push item onto heap, maintaining the heap invariant."""
    heap_list.append(item)
    _siftdown(heap_list, 0, len(heap_list)-1)


def heap_shortestPath(edges, source, end):
    # create a weighted DAG - {node:[(cost,neighbour), ...]}
#    import time
#    start_time = time.time()
    graph = collections.defaultdict(list)
    for l, r, c in edges:
        graph[l].append((c,r))
    # create a priority queue (FIFO) practically a list of the type [weight, node, path]
    queue = [(0, source, [])]
    visited = set()
    heapify(queue)  # Transform list queue into a heap
    # print(queue)
    # print(visited)

    # traverse the heap and give me the triplets in descending order
    while queue:
        (cost, node, path) = heappop(queue)
        # print((cost, node, path))

        if node not in visited:
            visited.add(node)
            path = path + [node]
 #           print(path)

            # when the end is reached
            if node == end:
                return (cost, path)

            # check the neighbours
            for c, neighbour in graph[node]:

                if neighbour not in visited:
                    heappush(queue, (cost+c['weight'], neighbour, path))
#    print("-Heap-- %s seconds ---" % (time.time() - start_time))
    return float("inf")

def dijkstras_shortestpath(graph, start, end):
#    import time
#    start_time = time.time()
    graph = graph.copy()
    dist = collections.defaultdict()
    predecessor = collections.defaultdict()
    infinity = float("inf")
    visited = []

# first step: set all nodes equal to infinity, except the start node which will be zero
    for node in graph:
        dist[node] = infinity
        predecessor[node] = -1
    dist[start] = 0

# While there are nodes i haven't visited
    while graph:
        parentNode = None
        for node in graph:  # traverse the nodes and check their shortest distances

            if parentNode is None:  # this is just for the first iteration
                parentNode = node

            elif dist[node] < dist[parentNode]:   # we have already set as node our start node which has the smallest shortest distance assigned
                parentNode = node  # so this is just to assure that we correctly set our start node

# now we have a focus node and we need to check all of its child nodes and its corresponding weights

        for neighbour, weight in graph[parentNode].items():   # this will give me pairs of nodes & their relevant weight / ex. startNode= "a" childNode = "b" weight = 10

# now we have already set the the start node and we know that it currently has the lowest shortest distance(zero)
# so what happens here is that we repeatedly check pairs of adjacent nodes and we relax the shortest_distance
# (meaning initially  turn it from infinity to a number and later on to a lower number if possible)

            if weight['weight'] + dist[parentNode] < dist[neighbour]:
                dist[neighbour] = weight['weight'] + dist[parentNode]
                predecessor[neighbour] = parentNode # we need this to find the path eventually
        graph.remove_node(parentNode)
    # print(dist_so_far)
    # print(predecessor)

# now it is time to write the path out
# we will move in the path backwards(start from the last node) and we will write down each indivindual step we take

    currentNode = end
    while currentNode != start:   # keep running until we reach the start node
        try:
            visited.append(currentNode) # insert in position 0 my current node (first iteration is the end node as defined by the user
            currentNode = predecessor[currentNode]  # update the currentNode with its predecessor (at every iteration go one step back and save each none so we end up with a list of nodes visited)
        except KeyError:   # just in case the path is not reachable
#            print ("\nPath is not reachable")
            break
# i eventually have reached the startNode so i exit the while loop but remember that i haven't saved it, so it is time to do so
    visited.append(start)

# Just a final check to check that we reached our endNode

#    if dist[end] != infinity:
#        print("\nThe path is " + str(visited))
#        print("\nThe shortest distance is " + str(dist[end]))
    
#    visited[0], visited[-1] = visited[-1], visited[0]
    visited = visited[::-1]
#    print("-Array-- %s seconds ---" % (time.time() - start_time))
    return visited,dist[end]

def min_span_tree(graph, result):
    '''function to find the min spanning tree '''
    K = graph.copy()
    for i in range(len(result)):
        n=[]
        for j in range(1,len(result[i][2])):
            m=[]
            m.append(result[i][2][j-1])
            m.append(result[i][2][j])
            n.append(m)
        
        result[i][2]=n
        
    for i in range(1,len(result)):
        n=result[0][2]
        for j in range(len(result[i][2])):
            if result[i][2][j] not in n:
                n.append(result[i][2][j])
        
    X = graph.copy()
    A = nx.to_edgelist(X)
    
    l=[]
    
    for i in range(len(nx.to_edgelist(K))):
        if list(nx.to_edgelist(K)[i][0:2]) not in n:
            l.append(i) 
    
    l=list(set(l))
                
    for i in sorted(l, reverse=True):
        del A[i]
    
#    print(nx.to_edgelist(nx.from_edgelist(A)))
#    nx.draw(nx.from_edgelist(A),with_labels = True, with_edges=True)
    return nx.from_edgelist(A)

def read_file():
    '''Function to take graph file input from user, read the file and check for sparsity of the graph'''
    path = file_input()

    f = open(path, 'r+')

    f.seek(0)
# checking for the input file from user
    if (f.readline().rstrip() == 'A'):
        print('Correct file to read, its an Arc Matrix')
        order_graph = int(f.readline().rstrip())
#        print(order_graph)
    else:
        print('Program is not capable of handling this graph file')
        j=1 #variable for for loop
        while j ==  1:
            print('Please enter \'Yes\' to Enter the file path again or else enter \'No\' to exit: ',end="")
            option=input()
            if option.lower() == 'yes':
                j = 2
                path = file_input()
            elif option.lower() == 'no':
                j = 2
                os._exit(1)
            else:
                print('Invalid Option please try again....')
        
        f.close()

# writing a temperory file from user's file after removing the first two lines 
# as after that we have a graph matrix and we will use that as an input to create
# graph variable
    with open(path) as f:
        with open('test.edgelist', 'w') as b:
            for i,line in enumerate(f):
                if i >=2:
                    b.write(line)
    f.close()
    b.close()

# creating a graph variable using the temp file we created
    K = nx.read_weighted_edgelist('test.edgelist',create_using =nx.DiGraph())
#    print(K.order() == order_graph)
# matching the order of graph with the order of graph given in the file
    if K.order() == order_graph:
        print("Order of the given graph is {}".format(K.order()))
    else:
        print("Order of the given graph and order on file doesn't matches ... Try again.. will exit now")
        os._exit(1)
# calculating the edge ratio for the graph to know the sparsity of the graph    
    N = (K.number_of_edges()/(K.number_of_nodes()*(K.number_of_nodes()-1)))
# checking if the graph is sparse or dense    
    if N <= 0.1 and K.number_of_edges()<=600:
        print("Given graph is a Sparse Graph")
        sparse_value ="sparse"
    else:
        print("Given graph is a Dense Graph")
        sparse_value = "dense"
#display the graph to user without weights    
    graph_darw(K)
    return K, sparse_value

      
def shortest_path_conv(graph,selection,start_node):
    '''Function finds the shortest path using the given method, from the user defined
    root node to all other nodes of the graph'''    
    K=graph.copy() # graph copy is created to avoid any changes in the global graph variable
    
    if selection == "array":
        
        result = []
        
        for end_node in sorted(K.nodes()):
            temp = []
            if end_node == start_node:
                pass
            else:
                temp.append(start_node)
                temp.append(end_node)
                dijk = list(dijkstras_shortestpath(K, start_node, end_node))
                temp.append(dijk[0])
                temp.append(dijk[1])
                result.append(temp)
                del dijk
    
    if selection == "heap":
        result = []
        
        for end_node in sorted(K.nodes()):
            temp = []
            if end_node == start_node:
                pass
            else:
                temp.append(start_node)
                temp.append(end_node)
                dijk = list(heap_shortestPath(nx.to_edgelist(K), start_node, end_node))
                temp.append(dijk[1])
                temp.append(dijk[0])
                result.append(temp)
                del dijk
    
    return result
                
def graph_darw(graph,start_node="none"):
    '''Function to darw any graph given on screen'''
    pos=nx.fruchterman_reingold_layout(graph)
    if start_node != "none":
        node_list=min_span.nodes()
        node_list.remove(start_node)
        nx.draw_networkx_nodes(graph,pos,nodelist=node_list,node_color='b',alpha=0.7)
        nx.draw_networkx_nodes(graph,pos,nodelist=[start_node],node_color='r',alpha=0.7)
        nx.draw_networkx_labels(graph,pos,font_color='w')
        nx.draw_networkx_edges(graph,pos)
        plt.title("Minimum Spanning Tree rooted at Node {}".format(start_node))
    else:
        nx.draw_networkx_nodes(graph,pos,node_color='b',alpha=0.7)
        nx.draw_networkx_labels(graph,pos,font_color='w')
        nx.draw_networkx_edges(graph,pos)
        plt.title("The input Graph")
    
    plt.axis('off')
    plt.show()

def closeness_centrality(K):
    # Closeness centrality for each node for graph K

    '''function for calculating closeness centrality for each node of a graph using dijkstra method'''
    j=1 #variable for for loop
    while j ==  1:
        print('\nPlease enter \'Yes\' to Calculate Closeness centrality for each node or else enter \'No\' for next option: ',end="")
        option=input()
        if option.lower() == 'yes':
            j = 2
        elif option.lower() == 'no':
            j = 2
            return None
        else:
            print('Invalid Option please try again....')

    for m in sorted(K.nodes()):
        result_m = shortest_path_conv(K,"array",m)
        d=0
        for a in result_m:
           # print(a[3])
            d=d+a[3]
            
        
        print("Closeness centrality measures for node {} is {}".format(m,(K.number_of_nodes()/d)))
    
    
def betweenness_centrality(K):
    # computing betweenness centrality for graph K
    
    
    j=1 #variable for for loop
    while j ==  1:
        print('\nPlease enter \'Yes\' to Calculate Betweenness centrality for each node or else enter \'No\' to exit: ',end="")
        option=input()
        if option.lower() == 'yes':
            j = 2
        elif option.lower() == 'no':
            j = 2
            os._exit(1)
        else:
            print('Invalid Option please try again....')
            
                
    # making permulation of nodes
    comb=itertools.permutations(K.nodes(),2)
    
    comb_list=[]
    
    for j in comb:
        comb_list.append(j)
    
    path_list=[]
    # finding the shorest paths between each pair of nodes and putting it into 
    # a list without the start and end node.
    
    for m,n in comb_list:
        temp_list=[]
        temp_list.append(m)
        temp_list.append(n)
        temp=[]
        temp.append(list(filter(lambda a: a != m, dijkstras_shortestpath(K, m, n)[0])))
        temp.append(list(filter(lambda a: a != n, temp[0])))
        temp_list.append(temp[1])
        path_list.append(temp_list)
        del temp_list,temp
    
    
    betweenness=[]
    
    # Node for which betweenness is to be calculated
    
    for find_node in sorted(K.nodes()):
        
        # a is the count of shortest paths in which the node in question is found.
        a=0
        
        for i in range(1,len(path_list)):
            
            if find_node in path_list[i][2]:
                a=a+1
        betweenness.append(a)
        print("Node {} is found in {} number of shortest paths".format(find_node,a))
    
    
    print('\nNode {} has highest Betweenness Centrality of {} '.format(sorted(K.nodes())[betweenness.index(max(betweenness))],max(betweenness)))


############# Program Strats here #################

#### taking input for graph file from user#########
K,sparse_value = read_file()

# Take input from user for the root node to create the min spanning tree
# also take input for the method to be used
start_node, method_g = take_input(sparse_value)

# finding shortest path from given root node to each node of the graph and 
# storing it as list 
#import time
#start_time = time.time()
result_g = shortest_path_conv(K,method_g,start_node)

# finding minimum spanning tree using the list of all shortest paths
min_span=min_span_tree(K,result_g)
#print("--- %s seconds ---" % (time.time() - start_time))
# drawing the min spanning tree
graph_darw(min_span,start_node)

print('\nMinimum Spanning Tree as a list is :')
print(nx.to_edgelist(min_span))

# Closeness centrality for each node for graph K
closeness_centrality(K)


# computing betweenness centrality for graph K
betweenness_centrality(K)

######################End#########################
    
    
    