# ant_colony_with_map.py

import networkx as nx
import matplotlib.pyplot as plt
import contextily as ctx
import pandas as pd
import random
import numpy as np
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

        graph.edges[i, j]['time'] = travel_time
        graph.edges[i, j]['weight'] = travel_time

    return graph

def plot_route_on_map(graph, tour, position, ax):
    ax.clear()

    ax.text(
        0.01, 0.99, "The algorithm animation is in progress...\nPlease wait. After the animation finishes, in about 30 seconds a convergence plot will be displayed.",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(facecolor='yellow', alpha=0.7)
        )

    nx.draw_networkx_nodes(graph, position, node_color="blue", node_size=300, ax=ax)
    nx.draw_networkx_labels(graph, position, font_size=12, font_color="white", ax=ax)

    path_edges = list(zip(tour, tour[1:]))
    nx.draw_networkx_edges(graph, position, edgelist=path_edges, edge_color='red', width=2, ax=ax)
    edge_labels = {(i, j): f"{graph[i][j]['time']:.2f} min" for i, j in path_edges}
    nx.draw_networkx_edge_labels(graph, position, edge_labels=edge_labels, ax=ax, font_size=8)

    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs="EPSG:3857")
    plt.pause(0.25)

def save_best_route(tour, total_time, data):
    output_dir = Path("output/ant_colony_with_map")
    output_dir.mkdir(exist_ok=True)
    filename = output_dir / f"best_route_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    df = pd.DataFrame({
        "Step": range(1, len(tour) + 1),
        "ID": [int(node) for node in tour],
    })

    df = df.merge(data, on="ID", how="left")

    df.to_csv(filename, index=False, encoding="utf-8", columns=["Step", "ID", "ADDRESS", "X", "Y"])

    with open(output_dir / f"best_route_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Total travel time (minutes): {total_time:.2f}\n")

    print(f"Saved best route to {filename}")


def plot_convergence(costs):
    plt.figure(figsize=(10, 5))
    plt.plot(costs, label="Best Cost per Iteration", color="green")
    plt.xlabel("Iteration")
    plt.ylabel("Travel Time (minutes)")
    plt.title("ACO Convergence Plot")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/ant_colony_with_map/convergence_plot.png")
    print("Saved convergence plot to output/ant_colony_with_map/convergence_plot.png")
    plt.show()

    plt.text(
        0.5, 0.95, "Close this window to end the program.",
        transform=plt.gca().transAxes,
        fontsize=10,
        color="black",
        ha='center',
        bbox=dict(facecolor='lightgreen', alpha=0.7)
        )


def ant_colony_tsp(graph, data, start_node=None, num_ants=200, num_iterations=100,
                   alpha=1, beta=2, evaporation_rate=0.5, pheromone_weight=1):
    pheromone = {edge: 1.0 for edge in graph.edges()}
    best_tour = None
    best_cost = float('inf')
    convergence = []

    position = {data.loc[i, 'ID']: (data.loc[i, 'X'], data.loc[i, 'Y']) for i in range(len(data))}
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.show()

    for iteration in range(num_iterations):
        all_tours = []
        all_costs = []

        for _ in range(num_ants):
            unvisited = set(graph.nodes)
            current_node = start_node if start_node is not None else random.choice(list(graph.nodes))
            tour = [current_node]
            unvisited.remove(current_node)

            while unvisited:
                neighbors = list(unvisited)
                probabilities = []

                for neighbor in neighbors:
                    edge = (current_node, neighbor) if (current_node, neighbor) in pheromone else (neighbor, current_node)
                    travel_time = graph[current_node][neighbor]['time']
                    if travel_time == 0:
                        travel_time = 0.01  # zabezpieczenie
                    probability = (pheromone[edge] ** alpha) * ((1 / travel_time) ** beta)
                    probabilities.append(probability)

                probabilities = np.array(probabilities) / sum(probabilities)
                next_node = random.choices(neighbors, weights=probabilities)[0]
                tour.append(next_node)
                unvisited.remove(next_node)
                current_node = next_node

            tour.append(tour[0])
            cost = sum(graph[tour[i]][tour[i + 1]]['time'] for i in range(len(tour) - 1))

            all_tours.append(tour)
            all_costs.append(cost)

        best_idx = np.argmin(all_costs)
        if all_costs[best_idx] < best_cost:
            best_cost = all_costs[best_idx]
            best_tour = all_tours[best_idx]

        convergence.append(best_cost)

        for edge in pheromone:
            pheromone[edge] *= (1 - evaporation_rate)

        for tour, cost in zip(all_tours, all_costs):
            for i in range(len(tour) - 1):
                edge = (tour[i], tour[i + 1]) if (tour[i], tour[i + 1]) in pheromone else (tour[i + 1], tour[i])
                pheromone[edge] += pheromone_weight / cost


        plot_route_on_map(graph, best_tour, position, ax)

    print("Results")
    print(f"Best route: {best_tour}")
    print(f"Total travel time (minutes): {best_cost:.2f}")

    save_best_route(best_tour, best_cost, data)
    plot_convergence(convergence)

    # Save final map
    fig.savefig("output/ant_colony_with_map/final_route_map.png", bbox_inches="tight", dpi=300)
    print("Saved final route map to output/ant_colony_with_map/final_route_map.png")

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    data_path = Path("data/data.csv")
    matrix_path = Path("output/distance_matrix.csv")

    data = pd.read_csv(data_path)
    travel_time_matrix = pd.read_csv(matrix_path, index_col=0)

    # NEW: Auto-detect and convert seconds to minutes if needed
    median_val = travel_time_matrix.stack().median()
    if median_val > 180:  # assume seconds if most values are large
        print("Detected high travel time values, likely in seconds. Converting to minutes...")
        travel_time_matrix = travel_time_matrix / 60
    else:
        print("Travel time values seem to be in minutes. Proceeding without conversion.")

    allowed_addresses = travel_time_matrix.index.tolist()
    data = data[data["ADDRESS"].isin(allowed_addresses)].reset_index(drop=True)
    data['ID'] = range(len(data))

    graph = build_complete_graph(data, travel_time_matrix)

    ant_colony_tsp(graph, data, start_node=data.loc[0, 'ID'])