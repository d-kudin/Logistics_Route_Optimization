# nearest_neighbor_with_map.py

# nearest_neighbor_with_map.py

import networkx as nx
import matplotlib.pyplot as plt
import contextily as ctx
import pandas as pd
import random
from pathlib import Path
from datetime import datetime

def build_complete_graph(data, travel_time_matrix):
    graph = nx.complete_graph(len(data))
    mapping = {i: data.loc[i, 'ID'] for i in range(len(data))}
    graph = nx.relabel_nodes(graph, mapping)

    id_to_address = data.set_index('ID')['ADDRESS'].to_dict()

    for i, j in graph.edges():
        address_i = id_to_address[i]
        address_j = id_to_address[j]
        travel_time = travel_time_matrix.loc[address_i, address_j]

        if pd.isna(travel_time):
            travel_time = float('inf')

        graph.edges[i, j]['time'] = travel_time
        graph.edges[i, j]['weight'] = travel_time

    return graph

def plot_route(graph, tour, current_node, position, ax, visited_nodes, final=False):
    ax.clear()

    if not final:
        ax.text(
            0.01, 0.99, "Calculating shortest path using Nearest Neighbor algorithm...",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(facecolor='yellow', alpha=0.7)
        )
    else:
        ax.text(
            0.5, 0.95, "Close this window to end the program.",
            transform=ax.transAxes,
            fontsize=10,
            color="black",
            ha='center',
            bbox=dict(facecolor='lightgreen', alpha=0.7)
        )

    nx.draw_networkx_nodes(graph, position, node_color="blue", node_size=300, ax=ax)
    nx.draw_networkx_labels(graph, position, font_size=12, font_color="white", ax=ax)

    path_edges = list(zip(tour, tour[1:]))

    if final:
        nx.draw_networkx_edges(graph, position, edgelist=path_edges, edge_color='red', width=2, ax=ax)
        edge_labels = {(i, j): f"{graph[i][j]['weight']:.1f} min" for i, j in path_edges}
        nx.draw_networkx_edge_labels(graph, position, edge_labels=edge_labels, ax=ax)
    else:
        nx.draw_networkx_edges(graph, position, edgelist=path_edges, edge_color='orange', width=2, ax=ax)

    nx.draw_networkx_nodes(graph, position, nodelist=[current_node], node_color='green', ax=ax)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs="EPSG:3857")
    plt.pause(0.25)

def save_tour(tour, total_time, data):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path("output/nearest_neighbor_with_map")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"Step": range(1, len(tour) + 1), "ID": tour})
    df = df.merge(data, on="ID", how="left")

    df.to_csv(output_dir / f"tour_{timestamp}.csv", index=False, encoding="utf-8",
              columns=["Step", "ID", "ADDRESS", "X", "Y"])

    with open(output_dir / f"summary_{timestamp}.txt", "w", encoding="utf-8") as f:
        f.write(f"Total travel time (minutes): {total_time:.2f}\n")

    print("Results")
    print(f"Tour: {tour}")
    print(f"Total travel time (minutes): {total_time:.2f}")

def nearest_neighbor_with_map(graph, data, start_node=None):
    if start_node is None:
        start_node = random.choice(list(graph.nodes))

    position = {data.loc[i, 'ID']: (data.loc[i, 'X'], data.loc[i, 'Y']) for i in range(len(data))}
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.show()

    unvisited = set(graph.nodes)
    unvisited.remove(start_node)
    tour = [start_node]
    current_node = start_node
    visited_nodes = {start_node}

    plot_route(graph, tour, current_node, position, ax, visited_nodes)

    while unvisited:
        next_node = min(
            unvisited,
            key=lambda node: graph[current_node][node]['time'] if graph[current_node][node]['time'] != float('inf') else float('inf')
        )
        unvisited.remove(next_node)
        tour.append(next_node)
        current_node = next_node
        visited_nodes.add(current_node)
        plot_route(graph, tour, current_node, position, ax, visited_nodes)

    tour.append(start_node)
    plot_route(graph, tour, current_node, position, ax, visited_nodes, final=True)

    total_time = sum(graph[tour[i]][tour[i + 1]]['time'] for i in range(len(tour) - 1))
    save_tour(tour, total_time, data)

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    try:
        data_path = Path("data/data.csv")
        matrix_path = Path("output/distance_matrix.csv")

        data = pd.read_csv(data_path)
        travel_time_matrix = pd.read_csv(matrix_path, index_col=0)

        median_val = travel_time_matrix.stack().median()
        if median_val > 180:
            print("Detected high travel time values, likely in seconds. Converting to minutes...")
            travel_time_matrix = travel_time_matrix / 60
        else:
            print("Travel time values seem to be in minutes. Proceeding without conversion.")

        # Filter only addresses that are in the matrix
        allowed_addresses = travel_time_matrix.index.tolist()
        data = data[data["ADDRESS"].isin(allowed_addresses)].reset_index(drop=True)
        data['ID'] = range(len(data))

        # Sample after filtering
        sampled_data = data.sample(n=50, random_state=129).reset_index(drop=True)
        graph = build_complete_graph(sampled_data, travel_time_matrix)

        nearest_neighbor_with_map(graph, sampled_data, start_node=sampled_data.loc[0, 'ID'])

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")