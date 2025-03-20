import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque
import heapq
import time
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Maze Solving Robot",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🤖 Maze Solving Robot")
st.markdown("""
This application demonstrates different search algorithms to solve a maze problem. 
The robot needs to navigate from point A to point B.
""")

# Define the maze graph
@st.cache_data
def get_maze_data():
    maze_graph = {
        'A': ['1'],
        '1': ['A', '2'],
        '2': ['1', '3', '4'],
        '3': ['2', '5'],
        '4': ['2', '10'],
        '5': ['3', '6', '7'],
        '6': ['5', '8'],
        '7': ['5', '9'],
        '8': ['6'],
        '9': ['7'],
        '10': ['4', '11', '12'],
        '11': ['10'],
        '12': ['10', '13'],
        '13': ['12', '14'],
        '14': ['13', '15', '17'],
        '15': ['14', '16'],
        '16': ['15'],
        '17': ['14', '18', '20'],
        '18': ['17', '19'],
        '19': ['18'],
        '20': ['17', '21'],
        '21': ['B'],
        'B': ['21']
    }

    node_positions = {
        'A': (0, 0),
        '1': (2, 0),
        '2': (2, 2),
        '3': (1, 2),
        '4': (3, 2),
        '5': (1, 3),
        '6': (0, 3),
        '7': (2, 3),
        '8': (0, 4),
        '9': (2, 4),
        '10': (3, 3),
        '11': (3, 4),
        '12': (4, 3),
        '13': (5, 3),
        '14': (5, 2),
        '15': (5, 0),
        '16': (8, 0),
        '17': (6, 2),
        '18': (6, 4),
        '19': (7, 4),
        '20': (6, 1),
        '21': (8, 1),
        'B': (8, 4)
    }

    heuristic = {
        'A': 8, '1': 6, '2': 6, '3': 6, '4': 7, '5': 4, '6': 12, '7': 7,
        '8': 15, '9': 18, '10': 6, '11': 8, '12': 6, '13': 5, '14': 4,
        '15': 8, '16': 6, '17': 3, '18': 5, '19': 2, '20': 5, '21': 1, 'B': 0
    }
    
    return maze_graph, node_positions, heuristic

maze_graph, node_positions, heuristic = get_maze_data()

# BFS algorithm
def bfs(graph, start, goal):
    visited = set()
    queue = deque([(start, [start])])
    explored_path = []
    
    while queue:
        vertex, path = queue.popleft()
        explored_path.append(vertex)
        
        if vertex == goal:
            return path, explored_path
        
        if vertex not in visited:
            visited.add(vertex)
            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
    
    return None, explored_path

# DFS algorithm
def dfs(graph, start, goal):
    visited = set()
    stack = [(start, [start])]
    explored_path = []
    
    while stack:
        vertex, path = stack.pop()
        explored_path.append(vertex)
        
        if vertex == goal:
            return path, explored_path
        
        if vertex not in visited:
            visited.add(vertex)
            for neighbor in reversed(graph[vertex]):  # Reversed to explore in the order given
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))
    
    return None, explored_path

# A* algorithm
def a_star(graph, start, goal, h):
    open_set = [(h[start], 0, start, [start])]  # (f, g, node, path)
    closed_set = set()
    explored_path = []
    
    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        explored_path.append(current)
        
        if current == goal:
            return path, explored_path
        
        if current in closed_set:
            continue
            
        closed_set.add(current)
        
        for neighbor in graph[current]:
            if neighbor in closed_set:
                continue
                
            tentative_g = g + 1  # Cost from start to neighbor through current
            tentative_f = tentative_g + h[neighbor]  # f = g + h
            
            heapq.heappush(open_set, (tentative_f, tentative_g, neighbor, path + [neighbor]))
    
    return None, explored_path

# Visualization function - COMPACT VERSION
def visualize_maze(graph, node_pos, explored_nodes=None, explored_step=None, solution_path=None, solution_step=None):
    G = nx.DiGraph()
    
    # Add edges to the graph
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    
    # Create figure - REDUCED SIZE
    fig, ax = plt.subplots(figsize=(5, 3))
    
    # Draw the graph
    nx.draw_networkx_edges(G, node_pos, alpha=0.3, width=1, edge_color='gray')
    
    node_colors = ['whitesmoke'] * len(G.nodes())
    node_border_colors = ['black'] * len(G.nodes())
    node_size = [300] * len(G.nodes())  # SMALLER NODE SIZE
    node_border_width = [0.8] * len(G.nodes())  
    
    # Adjust node appearance for special nodes
    for i, node in enumerate(G.nodes()):
        if node == 'A':
            node_colors[i] = 'limegreen'
            node_size[i] = 500  # SMALLER START NODE
            node_border_width[i] = 2
        elif node == 'B':
            node_colors[i] = 'tomato'
            node_size[i] = 500  # SMALLER GOAL NODE
            node_border_width[i] = 2
    
    # Show explored nodes up to the current step
    if explored_nodes and explored_step is not None:
        for i, node in enumerate(G.nodes()):
            if node in explored_nodes[:explored_step + 1]:
                if node != 'A' and node != 'B':  # Don't change start/goal colors
                    node_colors[i] = 'lightyellow'
                    node_border_colors[i] = 'goldenrod'
                    node_border_width[i] = 2
    
    # Show solution path up to the current step
    if solution_path and solution_step is not None:
        current_solution = solution_path[:min(solution_step + 1, len(solution_path))]
        
        # Draw path edges
        if len(current_solution) > 1:
            path_edges = [(current_solution[i], current_solution[i+1]) for i in range(len(current_solution)-1)]
            nx.draw_networkx_edges(G, node_pos, edgelist=path_edges, width=3, edge_color='royalblue')
        
        # Color solution nodes
        for i, node in enumerate(G.nodes()):
            if node in current_solution:
                if node == 'A' or node == 'B':
                    continue  # Keep start and end colors
                node_colors[i] = 'lightblue'
                node_border_colors[i] = 'blue'
                node_border_width[i] = 2
                
        # Color current position
        if current_solution:
            current_node = current_solution[-1]
            for i, node in enumerate(G.nodes()):
                if node == current_node and node != 'A' and node != 'B':
                    node_colors[i] = 'dodgerblue'
                    node_border_colors[i] = 'midnightblue'
                    node_border_width[i] = 3
    
    # Draw nodes with final colors
    for i, node in enumerate(G.nodes()):
        nx.draw_networkx_nodes(G, node_pos, nodelist=[node], node_color=[node_colors[i]], 
                               node_size=[node_size[i]], edgecolors=[node_border_colors[i]], 
                               linewidths=[node_border_width[i]])
    
    # Draw labels - SMALLER FONT
    nx.draw_networkx_labels(G, node_pos, font_size=8, font_weight='bold')
    
    # Add legend - SMALLER AND MOVED TO BOTTOM
    legend_elements = [
        mpatches.Patch(color='limegreen', label='Start (A)'),
        mpatches.Patch(color='tomato', label='Goal (B)'),
        mpatches.Patch(color='lightyellow', label='Explored'),
        mpatches.Patch(color='lightblue', label='Solution')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize='small')
    
    # Set title - SMALLER FONT
    plt.title("Maze Solving Robot", fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    
    return fig

# Main app layout - IMPROVED UI LAYOUT
# Create two columns for controls in a single row - simplified layout
col1, col2 = st.columns([1, 3])

with col1:
    algorithm = st.selectbox(
        "Algorithm:",
        ["Breadth-First Search (BFS)", "Depth-First Search (DFS)", "A* Search"]
    )

with col2:
    # Replace heuristic values with a bigger Solve Maze button
    st.markdown('<style>div.stButton > button {width: 100%; height: 3em;}</style>', unsafe_allow_html=True)
    run_button = st.button("🚀 SOLVE MAZE", type="primary")

# Create a container for the visualization and results
viz_container = st.container()

with viz_container:
    # Two columns layout for results and visualization
    viz_col, results_col = st.columns([3, 2])
    
    with viz_col:
        # Initial visualization placeholder
        initial_viz = st.empty()
        # Show initial maze
        fig = visualize_maze(maze_graph, node_positions)
        initial_viz.pyplot(fig)
        plt.close(fig)
    
    with results_col:
        # Initialize empty placeholders for results
        stats_container = st.container()
        with stats_container:
            st.markdown("### Select an algorithm and press 'SOLVE MAZE'")
            
            # Pre-create metrics with empty values
            metrics_cols = st.columns(3)
            with metrics_cols[0]:
                nodes_metric = st.empty()
            with metrics_cols[1]:
                path_metric = st.empty()
            with metrics_cols[2]:
                efficiency_metric = st.empty()
            
            # Placeholder for solution path
            solution_path_container = st.empty()
            
            # Placeholder for status
            status_text = st.empty()

# Run algorithm if button is clicked
if run_button:
    # Get selected algorithm and run it
    start, goal = 'A', 'B'
    
    if algorithm == "Breadth-First Search (BFS)":
        solution_path, explored_nodes = bfs(maze_graph, start, goal)
        algorithm_name = "BFS"
    elif algorithm == "Depth-First Search (DFS)":
        solution_path, explored_nodes = dfs(maze_graph, start, goal)
        algorithm_name = "DFS"
    else:  # A* Search
        solution_path, explored_nodes = a_star(maze_graph, start, goal, heuristic)
        algorithm_name = "A*"
    
    # Update results in the results column
    with results_col:
        # Update status
        status_text.text(f"Running {algorithm_name}...")
        
        # Update metrics
        metrics_cols[0].metric("Nodes Visited", len(explored_nodes))
        metrics_cols[1].metric("Path Length", len(solution_path))
        
        efficiency = round((len(solution_path) / len(explored_nodes)) * 100, 2) if explored_nodes else 0
        metrics_cols[2].metric("Efficiency", f"{efficiency}%", 
                  delta=f"{round(efficiency - 50, 2)}%" if efficiency > 50 else f"{round(efficiency - 50, 2)}%",
                  delta_color="normal")
        
        # Update solution path
        solution_path_container.markdown("### Solution")
        solution_path_container.code(" → ".join(solution_path))
    
    # Create progress bar
    progress_bar = st.progress(0)
    
    # Animation Phase 1: Explore the maze
    for step in range(len(explored_nodes)):
        progress = step / (len(explored_nodes) + len(solution_path))
        progress_bar.progress(progress)
        
        status_text.text(f"Exploring: {explored_nodes[step]}")
        
        fig = visualize_maze(
            maze_graph, 
            node_positions,
            explored_nodes=explored_nodes,
            explored_step=step,
            solution_path=None,
            solution_step=None
        )
        initial_viz.pyplot(fig)
        plt.close(fig)
        
        time.sleep(0.2)  # Faster animation

    # Animation Phase 2: Show the solution path
    for step in range(len(solution_path)):
        progress = (len(explored_nodes) + step) / (len(explored_nodes) + len(solution_path))
        progress_bar.progress(progress)
        
        status_text.text(f"Solution: {solution_path[step]}")
        
        fig = visualize_maze(
            maze_graph, 
            node_positions,
            explored_nodes=explored_nodes,
            explored_step=len(explored_nodes)-1,
            solution_path=solution_path,
            solution_step=step
        )
        initial_viz.pyplot(fig)
        plt.close(fig)
        
        time.sleep(0.2)  # Faster animation
    
    # Final state
    progress_bar.progress(1.0)
    status_text.text("Solution complete!")
    
    # Final visualization
    fig = visualize_maze(
        maze_graph, 
        node_positions,
        explored_nodes=explored_nodes,
        explored_step=len(explored_nodes)-1,
        solution_path=solution_path,
        solution_step=len(solution_path)-1
    )
    initial_viz.pyplot(fig)
    plt.close(fig)

# Footer
st.markdown("---")
st.markdown("🤖 by anis belaiouar")