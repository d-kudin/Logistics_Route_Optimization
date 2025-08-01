# ant_colony_with_plot.py

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import time
from pathlib import Path
from tqdm import tqdm

# Ensure output directory exists
output_dir = Path("output/ant_colony_with_plot")
output_dir.mkdir(parents=True, exist_ok=True)

def build_complete_graph(data, travel_time_matrix):
    graph = nx.complete_graph(len(data))
    mapping = {i: data.loc[i, 'ID'] for i in range(len(data))}
    graph = nx.relabel_nodes(graph, mapping)

    id_to_address = data.set_index('ID')['ADDRESS'].to_dict()

    for i, j in graph.edges():
        try:
            address_i = id_to_address[i]
            address_j = id_to_address[j]
            travel_time = travel_time_matrix.at[address_i, address_j]
        except KeyError:
            print(f"Warning: Missing travel time between {i} and {j}. Using fallback value.")
            travel_time = 9999.0  # fallback

        graph.edges[i, j]['time'] = travel_time
        graph.edges[i, j]['weight'] = travel_time

    return graph

def ant_colony_tsp(graph, start_node=None, num_ants=100, num_iterations=100,
                   alpha=1, beta=2, evaporation_rate=0.5, pheromone_weight=1):
    pheromone = {edge: 1.0 for edge in graph.edges()}
    best_tour = None
    best_cost = float('inf')

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
                    if travel_time == 0 or travel_time == float('inf'):
                        probability = 0
                    else:
                        probability = (pheromone[edge] ** alpha) * ((1 / travel_time) ** beta)
                    probabilities.append(probability)

                prob_sum = sum(probabilities)
                if prob_sum == 0:
                    next_node = random.choice(neighbors)
                else:
                    probabilities = np.array(probabilities) / prob_sum
                    next_node = np.random.choice(neighbors, p=probabilities)

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

        for edge in pheromone:
            pheromone[edge] *= (1 - evaporation_rate)

        for tour, cost in zip(all_tours, all_costs):
            for i in range(len(tour) - 1):
                edge = (tour[i], tour[i + 1]) if (tour[i], tour[i + 1]) in pheromone else (tour[i + 1], tour[i])
                pheromone[edge] += pheromone_weight / cost

    return best_cost

def generate_performance_plots(graph, start_node, ants_range, iterations_range):
    costs, times, results_table = [], [], []

    for num_ants in tqdm(ants_range, desc="Ants"):
        row_costs, row_times = [], []

        for num_iterations in iterations_range:
            print(f"▶️ Running for {num_ants} ants and {num_iterations} iterations...")
            start_time = time.time()
            best_cost = ant_colony_tsp(graph, start_node=start_node,
                                       num_ants=num_ants, num_iterations=num_iterations)
            elapsed = time.time() - start_time

            row_costs.append(best_cost)
            row_times.append(elapsed)
            results_table.append((num_ants, num_iterations, best_cost, elapsed))

        costs.append(row_costs)
        times.append(row_times)

    _plot_3d_results(ants_range, iterations_range, costs, times)
    _save_results_csv(results_table)

def _plot_3d_results(ants_range, iterations_range, costs, times):
    X, Y = np.meshgrid(iterations_range, ants_range)
    Z_costs = np.array(costs)
    Z_times = np.array(times)

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z_costs, cmap='viridis')
    ax1.set_title("\nTravel Time vs Ants & Iterations")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Ants")
    ax1.set_zlabel("Travel Time (min)")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, Z_times, cmap='plasma')
    ax2.set_title("\nExecution Time vs Ants & Iterations")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Ants")
    ax2.set_zlabel("Exec Time (s)")

    plt.tight_layout()
    output_file = output_dir / "performance_3d_plot.png"
    plt.savefig(output_file)
    print(f"📊 Saved 3D performance plot to {output_file}")
    plt.show()

def _save_results_csv(results):
    df = pd.DataFrame(results, columns=["num_ants", "num_iterations", "travel_time", "exec_time_sec"])
    output_file = output_dir / "performance_results.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Saved performance results to {output_file}")

if __name__ == '__main__':
    data_path = Path("data/data.csv")
    matrix_path = Path("output/distance_matrix.csv")

    data = pd.read_csv(data_path)
    travel_time_matrix = pd.read_csv(matrix_path, index_col=0)

    # Auto-detect and convert seconds to minutes if needed
    median_val = travel_time_matrix.stack().median()
    if median_val > 180:
        print("⚙️ Detected high travel time values. Converting seconds to minutes...")
        travel_time_matrix = travel_time_matrix / 60
    else:
        print("ℹ️ Travel time values appear to be in minutes. Proceeding without conversion.")

    # Keep only addresses present in the matrix
    data = data[data["ADDRESS"].isin(travel_time_matrix.index)].reset_index(drop=True)
    data['ID'] = range(len(data))

    sampled_data = data.sample(n=50, random_state=124).reset_index(drop=True)
    graph = build_complete_graph(sampled_data, travel_time_matrix)

    ants_range = [50, 100, 150, 200]
    iterations_range = [20, 40, 70, 100]

    generate_performance_plots(
        graph,
        start_node=sampled_data.loc[0, 'ID'],
        ants_range=ants_range,
        iterations_range=iterations_range
    )