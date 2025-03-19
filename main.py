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

# Visualization function
def visualize_maze(graph, node_pos, explored_nodes=None, explored_step=None, solution_path=None, solution_step=None):
    G = nx.DiGraph()
    
    # Add edges to the graph
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Draw the graph
    nx.draw_networkx_edges(G, node_pos, alpha=0.3, width=1, edge_color='gray')
    
    # Draw all nodes
    node_colors = ['whitesmoke'] * len(G.nodes())
    node_size = [700] * len(G.nodes())
    node_border_colors = ['black'] * len(G.nodes())
    node_border_width = [1] * len(G.nodes())
    
    # Adjust node appearance for special nodes
    for i, node in enumerate(G.nodes()):
        if node == 'A':
            node_colors[i] = 'limegreen'
            node_size[i] = 900
            node_border_width[i] = 2
        elif node == 'B':
            node_colors[i] = 'tomato'
            node_size[i] = 900
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
    
    # Draw labels
    nx.draw_networkx_labels(G, node_pos, font_size=12, font_weight='bold')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='limegreen', label='Start (A)'),
        mpatches.Patch(color='tomato', label='Goal (B)'),
        mpatches.Patch(color='lightyellow', label='Explored Nodes'),
        mpatches.Patch(color='lightblue', label='Solution Path')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Set title
    plt.title("Maze Solving Robot Visualization", fontsize=16)
    plt.axis('off')
    
    return fig

# Main app layout
st.markdown("""
This application demonstrates different search algorithms to solve a maze problem. 
The robot needs to navigate from point A to point B. You can compare how different algorithms explore the maze.
""")

# Create two columns for controls
col1, col2 = st.columns([1, 1])

with col1:
    algorithm = st.selectbox(
        "Select Algorithm:",
        ["Breadth-First Search (BFS)", "Depth-First Search (DFS)", "A* Search"]
    )

with col2:
    # Display heuristic values in an expandable section
    with st.expander("Heuristic Values (h)"):
        # Create columns for heuristic display
        h_col1, h_col2, h_col3 = st.columns(3)
        
        # Split the heuristic into 3 columns
        items = list(heuristic.items())
        third = len(items) // 3
        
        with h_col1:
            for node, h_value in items[:third]:
                st.write(f"{node}: {h_value}")
        
        with h_col2:
            for node, h_value in items[third:2*third]:
                st.write(f"{node}: {h_value}")
        
        with h_col3:
            for node, h_value in items[2*third:]:
                st.write(f"{node}: {h_value}")

# Run button
if st.button("🚀 Solve Maze", type="primary"):
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
    
    # Show algorithm information
    st.markdown(f"## {algorithm} Results")
    
    # Results in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### Explored Path ({len(explored_nodes)} nodes)")
        explored_path_str = " → ".join(explored_nodes)
        st.code(explored_path_str)
    
    with col2:
        st.markdown(f"### Solution Path ({len(solution_path)} nodes)")
        solution_path_str = " → ".join(solution_path)
        st.code(solution_path_str)
    
    # Show statistics
    st.markdown("### Statistics")
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    
    with stats_col1:
        st.metric("Total Nodes Visited", len(explored_nodes))
    
    with stats_col2:
        st.metric("Solution Length", len(solution_path))
    
    with stats_col3:
        efficiency = round((len(solution_path) / len(explored_nodes)) * 100, 2) if explored_nodes else 0
        st.metric("Efficiency", f"{efficiency}%", 
                  delta=f"{round(efficiency - 50, 2)}%" if efficiency > 50 else f"{round(efficiency - 50, 2)}%",
                  delta_color="normal")
    
    # Create placeholder for the visualization
    st.markdown("### Visualization")
    vis_placeholder = st.empty()
    
    # Animation is always on
    # First animate the exploration, then the solution path
    animation_speed = 1.0  # Fixed animation speed
    
    # Phase 1: Explore the maze
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for step in range(len(explored_nodes)):
        progress = step / (len(explored_nodes) + len(solution_path))
        progress_bar.progress(progress)
        
        status_text.text(f"Exploring node: {explored_nodes[step]}")
        
        fig = visualize_maze(
            maze_graph, 
            node_positions,
            explored_nodes=explored_nodes,
            explored_step=step,
            solution_path=None,
            solution_step=None
        )
        vis_placeholder.pyplot(fig)
        plt.close(fig)
        
        time.sleep(0.5)  # Fixed speed

    # Phase 2: Show the solution path
    for step in range(len(solution_path)):
        progress = (len(explored_nodes) + step) / (len(explored_nodes) + len(solution_path))
        progress_bar.progress(progress)
        
        status_text.text(f"Solution path: Reached {solution_path[step]}")
        
        fig = visualize_maze(
            maze_graph, 
            node_positions,
            explored_nodes=explored_nodes,
            explored_step=len(explored_nodes)-1,  # Show all explored nodes
            solution_path=solution_path,
            solution_step=step
        )
        vis_placeholder.pyplot(fig)
        plt.close(fig)
        
        time.sleep(0.5)  # Fixed speed
    
    # Final visualization
    progress_bar.progress(1.0)
    status_text.text("Solution complete!")
    
    # Show final result with all nodes
    fig = visualize_maze(
        maze_graph, 
        node_positions,
        explored_nodes=explored_nodes,
        explored_step=len(explored_nodes)-1,
        solution_path=solution_path,
        solution_step=len(solution_path)-1
    )
    vis_placeholder.pyplot(fig)
    plt.close(fig)

# Initial visualization of the maze without any solution
if 'vis_placeholder' not in locals():
    st.markdown("## Initial Maze")
    fig = visualize_maze(maze_graph, node_positions)
    st.pyplot(fig)
    plt.close(fig)



# Footer
st.markdown("---")
st.markdown("🤖 by anis belaiouar ")