import heapq
import numpy as np


class Graph:
    def __init__(self, vertices):
        """
        Constructor to initialize the Graph class.
        
        Args:
            vertices (int): The number of vertices in the graph.
        """
        self.V = vertices
        self.graph = self.graph = np.full((self.V, self.V), np.inf, dtype=float)
        self.successors = [[] for _ in range(self.V)]
        self.predecessors = [[] for _ in range(self.V)]

    def addEdge(self, u, v, w):
        """
        Add an edge to the graph.
        
        Args:
            u (int): The source vertex.
            v (int): The destination vertex.
            w (float): The weight of the edge.
        """
        self.graph[u][v] = w
        self.successors[u].append(v)   #T+
        self.predecessors[v].append(u) #T-


class Algorithm:
    def __init__(self, graph: Graph):
        """
        Constructor to initialize the Algorithm class.
        
        Args:
            graph (Graph): The graph object.
        """
        self.graph = graph                                   #Graph
        self.n = None                                        #total Vertices
        self.W_star = None                                   #Minimum Mean Cyle
        self.cycle_cost = None                               #Minimum Cost of Cycele
        self.cycle_length = None                             #Minimum Cycle Lentght
        self.cycle = None                                    #Cycle Path
        self.W = None                                        #mean length of a cycle
        self.mldc_val = None                                 #minimum length Directed Cycle
        self.lambda_star = None                              
        self.lambda_ = []                                    #average weight or length per node in the optimal MMC
        self.reduced_costs = None
        self.Buckets = []
        self.d = [[] for _ in range(self.graph.V)]
        self.pred = []
        self.cij = None                                      #Cost of Each Arc
        self.cjs = None

    def dijkstra(self, src):
        """
        Dijkstra's algorithm to find the shortest paths from a source node.
        
        Args:
            src (int): The source node.
        
        Returns:
            dist (numpy.array): The shortest distance from the source node to each node in the graph.
        """
        dist = np.full(self.graph.V, np.inf)
        dist[src] = 0
        heap = [(0, src)]

        while heap:
            cost, node = heapq.heappop(heap)

            if cost > dist[node]:
                continue

            for neighbor in self.graph.successors[node]:
                new_cost = dist[node] + self.graph.graph[node][neighbor]

                if new_cost < dist[neighbor]:
                    dist[neighbor] = new_cost
                    heapq.heappush(heap, (new_cost, neighbor))

        return dist

    def getCost(self, x, y):
        """
        Calculate the cost between two nodes.
        
        Args:
            x (int): The source node.
            y (int): The destination node.
        
        Returns:
            cost (float): The cost between the two nodes.
        """
        dist = self.dijkstra(x)
        return dist[y]

    def karp(self):
        """
        Karp's algorithm to find the minimum mean cycle.
        """
        self.n = self.graph.V
        A = np.full((self.n + 1, self.n), np.inf)
        A[0, :] = 0
        B = np.zeros((self.n + 1, self.n), dtype=int)

        for i in range(1, self.n + 1):
            for v in range(self.n):
                for u in self.graph.predecessors[v]:
                    
                    if A[i, v] > A[i - 1, u] + self.graph.graph[u, v]:
                        A[i, v] = A[i - 1, u] + self.graph.graph[u, v]
                        B[i, v] = u


        self.W_star = np.inf  # Initialize W_star to positive infinity
        v_star = None
        for v in range(self.n):
            if max((A[self.n, v] - A[i, v]) / (self.n - i) for i in range(self.n-1)) < self.W_star:
                self.W_star = max((A[self.n, v] - A[i, v]) / (self.n - i) for i in range(self.n))
                v_star = v

        cycle_start = v_star

        for _ in range(self.n):
            cycle_start = B[self.n, cycle_start]

        # Find the start of the cycle
        self.cycle = []
        
        
        while v_star not in self.cycle:
            self.cycle.append(v_star)
            v_star = B[self.n, v_star]

        self.cycle = self.cycle[self.cycle.index(v_star):]  # Only keep the cycle part
        self.cycle.append(self.cycle[0])  # Add the start node to the end to complete the cycle

        self.cycle = [c + 1 for c in self.cycle[::-1]]  # Adjust indices

        # Calculate the cost of the cycle
        self.cycle_cost = sum(self.graph.graph[self.cycle[i] - 1][self.cycle[i + 1] - 1] for i in range(len(self.cycle) - 1))
        self.cycle_length = len(self.cycle)-1

    def bucket_update(self,s,j):
        """
        Update the buckets based on the minimum length directed cycle.
        
        Args:
            s (int): The source node.
            j (int): The current node to be updated.
        """
        #print(s, j, self.d[s][j]) #kosten scheinen noch nicht geupdated zu werden
        k = int(self.d[s][j] / self.lambda_star) -1
        if (k+1<= self.cycle_length-1) and (j not in self.Buckets[k]): #evtl. -1 weglassen?
            for bucket in self.Buckets:
                if j in bucket:
                    bucket.remove(j)
            self.Buckets[k].append(j)

    def distance_update(self, s, i):
        """
        Update the distance values for successors of a node based on the minimum length directed cycle.

        Args:
            s (int): The source node.
            i (int): The current node for which distances are updated.
        """
        for j in self.graph.successors[i]: #j zählt ab 1
            self.cij = self.graph.graph[i][j] #deshalb hier j-1
            if self.d[s][j] > self.d[s][i] + self.cij:
                self.d[s][j] = self.d[s][i] + self.cij
                self.graph.predecessors[j] = i+1
                self.bucket_update(s,j)

    def min_length_cycle(self, s):
        """
        Find the minimum length directed cycle starting from a given source node.

        Args:
            s (int): The source node.
        """
        self.Buckets = [] #reset
        for k in range(0, abs(self.cycle_length) + 1):
            self.Buckets.append([])
        self.d = [[] for _ in range(self.graph.V)] #reset
        
        for i in range(0, self.n): #eventuell hier -1 ergänzen!
            self.d[s].append(float("inf"))
            
        self.d[s][s] = 0
        self.graph.predecessors[s] = 0
        self.distance_update(s, s)
        
        while (any(bucket for bucket in self.Buckets if bucket)):
            for i in range(0, len(self.Buckets)):
                if self.Buckets[i]:
                    k = i
                    break
            j = self.Buckets[k][0]
            self.Buckets[k].remove(j)

            self.cjs = self.getCost(j, s)

             
            if self.mldc_val >= self.d[s][j] + self.cjs:
                self.mldc_val = self.d[s][j] + self.cjs
                self.W = (s,j)
                self.distance_update(s,j)
                
                if s not in self.cycle:
                    self.cycle.append(s)
                
            


    def mldc(self):
        """
        Run the Minimum Length Directed Cycle (MLDC) algorithm to find the minimum length directed cycle.

        The algorithm performs the following steps:
        1. Run the Karp algorithm to find the minimum mean cycle.
        2. Initialize values and create a new graph object for G'.
        3. Replace the costs cij of each arc (i, j) ∈ A with c'ij = cij - λ*.
        4. Calculate the shortest path lengths p(i) from node 1 to node i in G'.
        5. Calculate the reduced costs cpij for each arc (i, j) ∈ A.
        6. Find the minimum length directed cycle for each source node in the range (2, graph.V).
        7. Print the cycle path, cycle length, and MLDC length.

        """
        #run karp
        self.karp()
        self.W = None
        self.mldc_val = self.cycle_cost

        self.lambda_star = self.cycle_cost / self.cycle_length

        # Erstellen Sie ein neues Graphenobjekt für G'
        graph_prime = Graph(self.graph.V)
        
        # Ersetzen Sie die Kosten cij jedes Bogens (i,j) ∈ A durch c'ij = cij - λ*
        for i in range(self.graph.V):
            for j in range(self.graph.V):
                if self.graph.graph[i][j] != np.inf:
                    graph_prime.addEdge(i, j, self.graph.graph[i][j] - self.lambda_star)
        

        # Berechnen Sie die kürzeste Pfadlänge p(i) von Knoten 1 zu Knoten i in G'
        p = [np.inf] * self.graph.V
        p[0] = 0
        for _ in range(self.graph.V):
            for i in range(self.graph.V):
                for j in range(self.graph.V):
                    if graph_prime.graph[i][j] != np.inf and p[i] + graph_prime.graph[i][j] < p[j]:
                        p[j] = p[i] + graph_prime.graph[i][j]

        # Berechnen Sie die reduzierten Kosten cpij für jeden Bogen (i,j) ∈ A
        self.reduced_costs = np.full((self.graph.V, self.graph.V), np.inf)
        for i in range(self.graph.V):
            for j in range(self.graph.V):
                self.reduced_costs[i][j] = graph_prime.graph[i][j] + p[i] - p[j] + self.lambda_star
                self.graph.graph[i][j] = self.reduced_costs[i][j]
                
        
        self.cycle = []
        for s in range(2, self.graph.V):
            self.min_length_cycle(s)
        
        self.cycle.append(self.cycle[0])
        self.cycle = [(i+1) for i in self.cycle ]
        print("Cycle: ", self.cycle)
        print("Cycle Length: ", len(self.cycle))
        print("MLDC Length:", self.mldc_val)
        

def create_graph():
    """
    Create a sample graph.

    The function creates an instance of the Graph class and adds edges to it.

    Returns:
        graph (Graph): The created graph.
    """
    # Beispiel-Graph
    graph = Graph(6)
    graph.addEdge(0,1,2)#graph.addEdge(1, 2, 2)
    graph.addEdge(0,2,2)#graph.addEdge(1, 3, 2)
    graph.addEdge(1,2,3)#graph.addEdge(2, 3, 3)
    graph.addEdge(1,3,2)#graph.addEdge(2, 4, 2)
    graph.addEdge(2,3,4)#graph.addEdge(3, 4, 4)
    graph.addEdge(3,0,3)#graph.addEdge(4, 1, 3)
    graph.addEdge(3,4,8)#graph.addEdge(4, 5, 8)
    graph.addEdge(3,5,2)#graph.addEdge(4, 6, 2)
    graph.addEdge(4,2,0)#graph.addEdge(5, 3, 0)
    graph.addEdge(5,2,-1)#graph.addEdge(6, 3, -1)
    graph.addEdge(5,4,0)#graph.addEdge(6, 5, 0)
    
    return graph

def test_karp_only():
    """
    Test the Karp algorithm separately.

    The algorithm performs the following steps:
    1. Run the Karp algorithm to find the minimum mean cycle (W_star).
    2. Print the value of W_star.
    3. Print the value of lambda_star (cycle_cost / cycle_length).

    """
    
    graph = create_graph()
    testKarpAlgo = Algorithm(graph)
    testKarpAlgo.karp()
    print("Lambda Star: ", testKarpAlgo.W_star)


def test_mean_length_directed_cycle():
    """
    Test the mean length directed cycle algorithm.

    The function creates a graph using the `create_graph` function, initializes an Algorithm object with the graph,
    runs the Karp algorithm, and then runs the MLDC algorithm. It prints the resulting cycle, cycle length, and MLDC length.

    """
    graph = create_graph()
    testKarpAlgo = Algorithm(graph)
    testKarpAlgo.karp()
    testKarpAlgo.mldc()

print("Testing MLDC Algorithm")
test_mean_length_directed_cycle()
print("",end="-------------------------\n")
print("Testing Karp")
test_karp_only()
