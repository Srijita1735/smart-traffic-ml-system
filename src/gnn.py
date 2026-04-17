print("\n⚡ EDGE AI:\n")
print("Prediction running locally on traffic signal device")
import networkx as nx


G = nx.Graph()

# Add edges (roads with traffic weights)
G.add_edge("A", "B", weight=5)
G.add_edge("B", "C", weight=3)
G.add_edge("A", "C", weight=10)  # alternative longer route


print("\n GRAPH TRAFFIC MODEL:\n")

for edge in G.edges(data=True):
    print(edge)


path = nx.shortest_path(G, source="A", target="C", weight="weight")

cost = nx.shortest_path_length(G, source="A", target="C", weight="weight")

print("\n OPTIMAL ROUTE USING GRAPH:\n")
print("Best Path:", path)
print("Total Cost:", cost)