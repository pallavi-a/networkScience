# Reading network data for Facebook (as well as Twitter)
import networkx as nx
import matplotlib.pyplot as plt

G =nx.read_edgelist("/content/facebook_combined.txt", create_using = nx.Graph(), nodetype=int)
     

nx.info(G)


# importing libraries
import community.community_louvain as community_louvain
import networkx as nx
import matplotlib.pyplot as plt
     

num_communities = 0 # initializing the number of communities
best_partition = community_louvain.best_partition(G) # applying the algorithm
partition_size = float(len(set(best_partition.values())))#values of the modules we got
pos = nx.fruchterman_reingold_layout(G) # choosing layout for graph placement

for community in set(best_partition.values()) :
    num_communities = num_communities + 1 # adding the number of communities
    com_nodes = [nodes for nodes in best_partition.keys()
                                if best_partition[nodes] == community]  # listing nodes if they belong to the same community
    nx.draw_networkx_nodes(G, pos, com_nodes, node_size = 20,
                                node_color = 'maroon') # drawing the community with different color to distinguish
    


nx.draw_networkx_edges(G, pos, alpha=0.1) # drawing graph with 0.1 opacity since networks are large
plt.show()

nx.draw_networkx_edges(G, pos, alpha=0.5) # drawing graph with 0.1 opacity since networks are large

print(num_communities) # getting the number of communities