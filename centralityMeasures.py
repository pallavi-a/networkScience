import networkx as nx
import matplotlib.pyplot as plt

G =nx.read_edgelist("/content/twitter_combined.txt", create_using = nx.DiGraph(), nodetype=int)

print(nx.info(G))

degCent = nx.degree_centrality(G) # Degree centrality
degree_sorted= sorted(degCent, key=degCent.get, reverse=True)[:100]

degree_sorted

eigCent = nx.eigenvector_centrality(G, max_iter=600) #eigen-vector centrality

eigen_sorted= sorted(eigCent, key=eigCent.get, reverse=True)[:100]

eigen_sorted

res = len(set(eigen_sorted) & set(degree_sorted)) / float(len(set(eigen_sorted) | set(degree_sorted))) * 100 # code for similarity

cloCent = nx.closeness_centrality(G) # closeness centrality

cl_sorted= sorted(cloCent, key=cloCent.get, reverse=True)[:20]

cl_sorted

list1_as_set = set(eigen_sorted) # checking similarity %
intersection = list1_as_set.intersection(cl_sorted)
intersection_as_list = list(intersection)
print(len(intersection_as_list))

res # similarity % of 100 nodes

pgCent = nx.pagerank(G) # page-rank
pg_sorted= sorted(pgCent, key=pgCent.get, reverse=True)[:100]

betCent = nx.betweenness_centrality(G, normalized=True, endpoints=True) # betweenness centraility 
degree_sorted= sorted(betCent, key=betCent.get, reverse=True)[:100]

nx.number_strongly_connected_components(G) # number of strongly connected components

N, K = G.order(), G.size() # average 
avg_deg = float(K) / N
print(avg_deg)

import networkx as nx
import matplotlib.pyplot as plt

G =nx.read_edgelist("/content/facebook_combined.txt", create_using = nx.Graph(), nodetype=int)

# degree distribution - power law
def degree_distribution(G, in_degree=False, out_degree=False):
    nodes = G.nodes()
    if in_degree:
        in_degree = dict(G.in_degree()). # in-degree
        degseq=[in_degree.get(k,0) for k in nodes]
    elif out_degree:
        out_degree = dict(G.out_degree()) #out-degree
        degseq=[out_degree.get(k,0) for k in nodes]
    else:
        degseq=[v for k, v in G.degree()]
    dmax=max(degseq)+1
    f= [ 0 for d in range(dmax) ] # calculating freq
    for degree in degseq:
        f[degree] += 1
    return fr

in_degree_freq = degree_distribution(G)
degrees = range(len(in_degree_freq))
plt.figure(figsize=(10, 10)) 
plt.loglog(range(len(in_degree_freq)), in_degree_freq, 'ro-', label='degree') 
plt.xlabel('Degree on log scale')
plt.ylabel('Number of Nodes on log scale')
plt.legend(loc="upper right")
plt.title('Degree Distribution-Facebook')
plt.show()

in_degree_freq = degree_histogram_directed(G)
degrees = range(len(in_degree_freq))
plt.figure(figsize=(10, 10)) 
plt.plot(range(len(in_degree_freq)), in_degree_freq, 'ro-', label='degree') 
plt.xlabel('Degree')
plt.ylabel('Number of Nodes')
plt.legend(loc="upper right")
plt.title('Degree Distribution-Facebook')

nx.average_shortest_path_length(G) # average shortest distance

nx.Diameter(G) #diameter of the network