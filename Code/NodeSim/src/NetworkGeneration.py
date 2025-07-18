import pandas as pd
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
from itertools import combinations
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split

# Define the range of rows to read
start_row = 0
end_row = 10000
num_rows = end_row - start_row

# Read the specified range of rows from the CSV file
df = pd.read_csv("D:/ICPE/dataset/first 1000 ms/CallGraph_0_sorted.csv", skiprows=range(1, start_row), nrows=num_rows, low_memory=False)

# Create a directed graph
G = nx.DiGraph()

# Add edges based on um and dm columns
node_ids = {}  # Dictionary to store unique identifiers for each node
id_counter = 0  # Counter to assign unique identifiers
for index, row in df.iterrows():
    if row['um'] not in node_ids:
        node_ids[row['um']] = id_counter
        id_counter += 1
    if row['dm'] not in node_ids:
        node_ids[row['dm']] = id_counter
        id_counter += 1
    G.add_edge(node_ids[row['um']], node_ids[row['dm']])

# Calculate the sum of in-degree and out-degree for each node
degree_sum = {node: G.in_degree(node) + G.out_degree(node) for node in G.nodes()}

# Filter out nodes where the sum of in-degree and out-degree is less than or equal to 2
filtered_nodes = {node for node, degree in degree_sum.items() if degree > 10}

# Create a subgraph with the filtered nodes
G_filtered = G.subgraph(filtered_nodes).copy()

# Convert to an undirected graph for Louvain method
G_undirected = G_filtered.to_undirected()

# Apply Louvain method with a specific resolution
resolution = 1.0  # You can adjust this value to change the number of communities
partition = community_louvain.best_partition(G_undirected, resolution=resolution)

# Create the nodes_and_edges_filtered file including all nodes with their communities and edges
with open("nodes_and_edges_filtered.txt", "w") as file:
    file.write("Nodes:\n")
    for node, comm in partition.items():
        file.write(f"{node}: {{'community': {comm}}}\n")

    file.write("\nEdges:\n")
    for edge in G_filtered.edges():
        file.write(f"('{edge[0]}', '{edge[1]}', {{}})\n")

# Create the same_community_filtered file with pairs of nodes and whether they belong to the same community
output_data_1 = []
for node1, node2 in combinations(G_filtered.nodes(), 2):
    same_community = 1 if partition[node1] == partition[node2] else 0
    output_data_1.append((node1, node2, same_community))

with open("same_community_filtered.txt", "w") as file:
    for item in output_data_1:
        file.write(f"{item[0]} {item[1]} {item[2]}\n")

# Create the connected_filtered file indicating if the pairs are connected by an edge
output_data_2 = []
for node1, node2 in combinations(G_filtered.nodes(), 2):
    connected = 1 if G_filtered.has_edge(node1, node2) or G_filtered.has_edge(node2, node1) else 0
    output_data_2.append(connected)

with open("connected_filtered.txt", "w") as file:
    for connected in output_data_2:
        file.write(f"{connected}\n")

# Combine the two outputs into a dataframe
data_final = pd.DataFrame({
    'node1': [item[0] for item in output_data_1],
    'node2': [item[1] for item in output_data_1],
    'same_community': [item[2] for item in output_data_1],
    'connected': output_data_2
})

# Split the data into training and testing sets randomly
train, test = train_test_split(data_final, test_size=0.3, random_state=42)

# Write the training set to files
train.to_csv("sample_trainx_filtered.txt", sep=' ', index=False, header=False, columns=['node1', 'node2', 'same_community'])
train.to_csv("sample_trainy_filtered.txt", sep=' ', index=False, header=False, columns=['connected'])

# Write the testing set to files
test.to_csv("sample_testx_filtered.txt", sep=' ', index=False, header=False, columns=['node1', 'node2', 'same_community'])
test.to_csv("sample_testy_filtered.txt", sep=' ', index=False, header=False, columns=['connected'])

# Generate the nodes_and_edges files for train and test sets
def generate_nodes_and_edges(file_path, data, partition):
    with open(file_path, "w") as file:
        file.write("Nodes:\n")
        for node in set(data['node1']).union(set(data['node2'])):
            file.write(f"{node}: {{'community': {partition[node]}}}\n")

        file.write("\nEdges:\n")
        for _, row in data.iterrows():
            if row['connected'] == 1:
                file.write(f"('{row['node1']}', '{row['node2']}', {{}})\n")

generate_nodes_and_edges("nodes_and_edges_train_filtered.txt", train, partition)
generate_nodes_and_edges("nodes_and_edges_test_filtered.txt", test, partition)

# Generate the same_community files for train and test sets
def generate_same_community(file_path, data, partition):
    output_data = []
    for _, row in data.iterrows():
        same_community = 1 if partition[row['node1']] == partition[row['node2']] else 0
        output_data.append((row['node1'], row['node2'], same_community))

    with open(file_path, "w") as file:
        for item in output_data:
            file.write(f"{item[0]} {item[1]} {item[2]}\n")

generate_same_community("same_community_train_filtered.txt", train, partition)
generate_same_community("same_community_test_filtered.txt", test, partition)

# Position the nodes using the spring layout for better visualization
pos = nx.spring_layout(G_filtered)

# Get a color map for the communities
unique_communities = set(partition.values())
colors = plt.cm.rainbow([i / len(unique_communities) for i in range(len(unique_communities))])
community_color_map = {comm: colors[i] for i, comm in enumerate(unique_communities)}

# Assign colors to nodes based on their community
node_colors = [community_color_map[partition[node]] for node in G_filtered.nodes()]

# Draw the graph
print("Start drawing...")
plt.figure(figsize=(12, 8))
nx.draw(G_filtered, pos, node_color=node_colors, with_labels=True, node_size=300, font_size=8, edge_color='gray')
plt.title("Graph Visualization with Community Coloring")
plt.show()

# Number of nodes
num_nodes = G_filtered.number_of_nodes()

# Total number of possible connections in an undirected graph
total_possible_connections = num_nodes * (num_nodes - 1) / 2

# Number of actual connections (edges)
num_actual_connections = G_filtered.number_of_edges()

# Calculate the percentage of connected edges
percentage_connected = (num_actual_connections / total_possible_connections) * 100

print(f"Total possible connections: {total_possible_connections}")
print(f"Number of actual connections: {num_actual_connections}")
print(f"Percentage of connected edges: {percentage_connected:.2f}%")

print("All outputs have been written to their respective files.")


