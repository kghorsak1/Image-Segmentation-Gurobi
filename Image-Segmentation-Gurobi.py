import numpy as np
import pandas as pd
import math
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import subprocess
import sys
from PIL import Image
from skimage.filters import threshold_otsu
from skimage import feature

# Function to convert file (image or CSV) to NumPy array with max size
def file_to_numpy(file_path, max_size=(200, 200)):
    try:
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Process image file
            with Image.open(file_path) as img:
                img = img.convert("L")  # Convert to grayscale

                if max_size is not None:
                    # Resize image to max_size while maintaining aspect ratio
                    img.thumbnail(max_size, Image.ANTIALIAS)

                # Convert to NumPy array and normalize pixel values
                img_array = np.array(img) / 255.0  # Normalize to [0, 1]

                return img_array

        elif file_path.lower().endswith('.csv'):
            # Process CSV file
            data = pd.read_csv(file_path, header=None).to_numpy()
            # Normalize CSV data if it's not in [0, 1]
            if data.max() > 1.0:
                data = data / data.max()
            return data

        else:
            raise ValueError("File is not a .jpg, .jpeg, .png, or .csv format.")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

# Load the data using the file_to_numpy function
file_path = 'C:/Users/Kimble/Downloads/trees.jpg'  # Update this to the actual file path you are using, Lion, swan, lizard, trees . jpg
box = file_to_numpy(file_path, max_size=(200, 200))  # Specify max_size here

if box is None:
    raise ValueError("Failed to load the image or CSV file. Please check the file path and format.")

print("Box shape:", box.shape)
print(box)
# Function to compute dynamic sigma
def compute_sigma(box):
    intensity_diffs = []
    n_rows, n_cols = box.shape
    for r in range(n_rows):
        for c in range(n_cols):
            neighbors = [
                (r-1, c-1), (r-1, c), (r-1, c+1),
                (r, c-1),             (r, c+1),
                (r+1, c-1), (r+1, c), (r+1, c+1)
            ]
            for nr, nc in neighbors:
                if 0 <= nr < n_rows and 0 <= nc < n_cols:
                    intensity_diffs.append(abs(box[r, c] - box[nr, nc]))
    sigma = np.std(intensity_diffs)
    print(intensity_diffs)
    return sigma if sigma != 0 else 0.1  # To avoid division by zero

# Function to detect boundary pixels using adaptive thresholding and edge detection
def find_boundary_pixels(box):
    # Adaptive thresholding
    thresh = threshold_otsu(box)
    binary_box = box > thresh

    # Edge detection
    edges = feature.canny(box, sigma=2)
    n_rows, n_cols = box.shape
    boundary_pixels = []
    for r in range(n_rows):
        for c in range(n_cols):
            if edges[r, c]:
                boundary_pixels.append(r * n_cols + c)
    return boundary_pixels

# Get boundary pixels
boundary_pixels = find_boundary_pixels(box)

# Ensure we have boundary pixels
if not boundary_pixels:
    raise ValueError("No boundary pixels found. Please check the image content.")

# Define the source pixel (first boundary pixel)
source_pixel =  boundary_pixels[0]   #Can replace with a number within the sixe of the box array, ei 128x128 can replace with 1 - 16384
print("boundary pixel")
print(boundary_pixels[0])

# Function to perform graph cut segmentation using Gurobi
def graph_cut_segmentation(box):
    n_rows, n_cols = box.shape
    n_pixels = n_rows * n_cols

    # Compute sigma dynamically
    sigma = compute_sigma(box)

    # Initialize the Gurobi model
    model = gp.Model('GraphCut')
    model.Params.LogToConsole = 0

    # Nodes: pixel nodes, source node (n_pixels), sink node (n_pixels + 1)
    source = n_pixels      # Index of the source node
    sink = n_pixels + 1    # Index of the sink node
    total_nodes = n_pixels + 2

    # Create variables for flows on edges
    flow_vars = {}  # Dictionary to hold flow variables
    capacities = {}  # Dictionary to hold capacities of edges

    # Adaptive thresholding
    thresh = threshold_otsu(box)
    binary_box = box > thresh  # foreground pixels

    # Build capacities and variables
    for r in range(n_rows):
        for c in range(n_cols):
            idx = r * n_cols + c  # Node index for this pixel

            # Edge from source to pixel or pixel to sink
            if binary_box[r, c]:
                # Foreground pixel: link from source to pixel with capacity 1
                capacities[(source, idx)] = 1
            else:
                # Background pixel: link from pixel to sink with capacity 1
                capacities[(idx, sink)] = 1

            # Edges between neighboring pixels
            neighbors = [
                (r-1, c-1), (r-1, c), (r-1, c+1),
                (r, c-1),             (r, c+1),
                (r+1, c-1), (r+1, c), (r+1, c+1)
            ]
            for nr, nc in neighbors:
                if 0 <= nr < n_rows and 0 <= nc < n_cols:
                    n_idx = nr * n_cols + nc  # Neighbor pixel node index
                    weight = math.exp(-((box[r, c] - box[nr, nc]) ** 2)/(2 * sigma ** 2))
                    # Add edge between idx and n_idx with capacity = weight
                    capacities[(idx, n_idx)] = weight

    # Now create flow variables for all edges in capacities
    for (i, j), cap in capacities.items():
        flow_vars[(i, j)] = model.addVar(lb=0, ub=cap, vtype=GRB.CONTINUOUS, name=f'f_{i}_{j}')
        
    model.update()

    # Flow conservation constraints for all nodes except source and sink
    for idx in range(n_pixels):
        inflow = gp.quicksum(flow_vars[(i, idx)] for i in range(total_nodes) if (i, idx) in flow_vars)
        outflow = gp.quicksum(flow_vars[(idx, j)] for j in range(total_nodes) if (idx, j) in flow_vars)
        model.addConstr(inflow == outflow, name=f'flow_conservation_{idx}')

    # Objective: Maximize total flow from source to sink
    total_flow = gp.quicksum(flow_vars[(source, j)] for j in range(total_nodes) if (source, j) in flow_vars)
    model.setObjective(total_flow, GRB.MAXIMIZE)

    # Solve the model
    model.optimize()

    # Reconstruct the segmentation
    if model.status == GRB.OPTIMAL:
        # Build residual graph and perform BFS from source
        residual_capacities = {}

        for (i, j), var in flow_vars.items():
            flow = var.X
            cap = var.UB
            residual_capacity = cap - flow
            if residual_capacity > 1e-5:  # Tolerance to avoid floating point errors
                residual_capacities.setdefault(i, []).append(j)
            if flow > 1e-5:
                residual_capacities.setdefault(j, []).append(i)  # Reverse flow

        # Now perform BFS from source
        from collections import deque
        visited = [False] * total_nodes
        queue = deque()
        queue.append(source)
        visited[source] = True

        while queue:
            u = queue.popleft()
            for v in residual_capacities.get(u, []):
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)

        # Pixels reachable from source are in the foreground
        segmentation = np.array(visited[:n_pixels]).reshape((n_rows, n_cols))

        # Print flow variable values
        print("Flow variable values:")
        for var in model.getVars():
            print(var.X)
            if var.X > 1e-5:
                print(f"{var.VarName}: {var.X}")

        # Visualize the result
        plt.figure(figsize=(10, 10))
        plt.imshow(box, cmap='gray')
        plt.contour(segmentation, [0.5], colors='r')
        plt.title('Graph Cut Segmentation with Gurobi')
        plt.axis('off')
        plt.show()
    else:
        print(f"Optimization was not successful. Status code: {model.status}")

# Perform graph cut segmentation
graph_cut_segmentation(box)

# Sample runtime notes:
# - Swan image takes approximately 3 minutes at 128x128 box size (~16,384 pixels).
# - Lizard image takes approximately 7 minutes with 148x217 box size (~32,116 pixels).
# - Tree image takes 7 minutes at 109,200 box size