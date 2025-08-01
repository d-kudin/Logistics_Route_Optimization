# main.py

from generate_distance_matrix import main as generate_matrix
from algorithms.ant_colony_with_map import ant_colony_tsp, build_complete_graph
from algorithms.nearest_neighbor_with_map import nearest_neighbor_with_map
from algorithms.ant_colony_with_plot import generate_performance_plots
import pandas as pd

def run_all():
    print("\nGenerating distance matrix...")
    generate_matrix()

    print("\nLoading data and distance matrix...")
    data = pd.read_csv("data/data.csv")
    matrix = pd.read_csv("output/distance_matrix.csv", index_col=0)

    # Auto-detect units
    median_val = matrix.stack().median()
    if median_val > 180:
        print("Detected high travel time values, likely in seconds. Converting to minutes...")
        matrix = matrix / 60
    else:
        print("Travel time values seem to be in minutes. Proceeding without conversion.")

    # Filter data to match matrix addresses
    allowed_addresses = matrix.index.tolist()
    filtered_data = data[data["ADDRESS"].isin(allowed_addresses)].reset_index(drop=True)
    filtered_data['ID'] = range(len(filtered_data))

    # Sample AFTER filtering
    sampled_data = filtered_data.sample(n=50, random_state=129).reset_index(drop=True)
    sampled_data['ID'] = range(len(sampled_data))  # reindex IDs for consistency

    # Ensure matrix matches sampled addresses
    sampled_addresses = sampled_data["ADDRESS"].tolist()
    matrix = matrix.loc[sampled_addresses, sampled_addresses]

    print("\nRunning Nearest Neighbor...")
    graph = build_complete_graph(sampled_data, matrix)
    nearest_neighbor_with_map(graph, sampled_data, start_node=sampled_data.loc[0, 'ID'])

    print("\nRunning Ant Colony Optimization...")
    graph = build_complete_graph(sampled_data, matrix)
    ant_colony_tsp(graph, sampled_data, start_node=sampled_data.loc[0, 'ID'])

    print("\nGenerating ACO parameter performance plots...")
    ants_range = [50, 100, 150, 200]
    iter_range = [20, 40, 70, 100]
    generate_performance_plots(graph, sampled_data.loc[0, 'ID'], ants_range, iter_range)

if __name__ == "__main__":
    run_all()