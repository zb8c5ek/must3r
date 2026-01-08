#!/usr/bin/env python3
# Copyright (C) 2025-present Naver Corporation. All rights reserved.
"""
Export and visualize SLAM scene graph, overlap tree, and connectivity information
"""
import argparse
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path


def load_slam_results(results_dir):
    """Load SLAM results from directory"""
    # Load poses
    poses_file = f"{results_dir}/all_poses.npz"
    poses_data = np.load(poses_file, allow_pickle=True)
    
    # Load memory
    memory_file = f"{results_dir}/memory.pkl"
    with open(memory_file, 'rb') as f:
        memory_tuple = pickle.load(f)
    
    # Memory is saved as: (memory, keyframe_pointmaps, overlap_tree)
    memory, keyframe_pointmaps, overlap_tree = memory_tuple
    
    return poses_data, memory, keyframe_pointmaps, overlap_tree


def extract_overlap_tree_info(overlap_tree):
    """Extract information from the overlap tree"""
    info = {
        'type': type(overlap_tree).__name__,
        'num_points_per_quadrant': []
    }
    
    if hasattr(overlap_tree, 'search_structs'):
        # QuandrantSearcher
        info['quadrant_divider'] = overlap_tree.quadrant_divider
        info['num_quadrants'] = len(overlap_tree.search_structs)
        
        for i, search_struct in enumerate(overlap_tree.search_structs):
            if hasattr(search_struct, 'all_points') and len(search_struct.all_points) > 0:
                info['num_points_per_quadrant'].append({
                    'quadrant_id': i,
                    'num_points': len(search_struct.all_points)
                })
    elif hasattr(overlap_tree, 'all_points'):
        # Simple KDTree
        info['num_points'] = len(overlap_tree.all_points) if overlap_tree.all_points else 0
    
    return info


def build_scene_graph(poses_data, memory, keyframe_pointmaps):
    """Build scene graph from memory structure"""
    # Extract keyframe information from memory
    # Memory structure depends on the model, but we can infer connectivity
    
    num_poses = len(poses_data['poses'])
    timestamps = poses_data['timestamps']
    
    # Build graph
    G = nx.DiGraph()
    
    # Add all frames as nodes
    for i in range(num_poses):
        G.add_node(i, 
                  timestamp=int(timestamps[i]),
                  confidence=float(poses_data['confs'][i]),
                  is_keyframe=False)
    
    # Mark keyframes (we have pointmaps for these)
    # We need to map keyframe indices to frame indices
    # The memory contains the sequence state at different points
    
    # Add sequential edges (temporal connectivity)
    for i in range(num_poses - 1):
        G.add_edge(i, i + 1, edge_type='temporal', weight=1.0)
    
    # Add keyframe information
    if isinstance(memory, tuple) and len(memory) > 0:
        # Try to extract keyframe indices from memory structure
        # This is model-specific, but we can make reasonable assumptions
        num_keyframes = len(keyframe_pointmaps)
        
        # Estimate keyframe indices (they're typically spread through the sequence)
        keyframe_stride = max(1, num_poses // num_keyframes)
        estimated_keyframe_indices = [i * keyframe_stride for i in range(num_keyframes)]
        
        for kf_idx in estimated_keyframe_indices:
            if kf_idx < num_poses:
                G.nodes[kf_idx]['is_keyframe'] = True
    
    return G


def export_scene_graph_json(G, output_file):
    """Export scene graph to JSON format"""
    data = {
        'nodes': [],
        'edges': []
    }
    
    for node_id, node_data in G.nodes(data=True):
        data['nodes'].append({
            'id': int(node_id),
            **{k: (int(v) if isinstance(v, (np.integer, np.int64)) else 
                   float(v) if isinstance(v, (np.floating, np.float32, np.float64)) else v) 
               for k, v in node_data.items()}
        })
    
    for source, target, edge_data in G.edges(data=True):
        data['edges'].append({
            'source': int(source),
            'target': int(target),
            **{k: (float(v) if isinstance(v, (np.floating, np.float32, np.float64)) else v) 
               for k, v in edge_data.items()}
        })
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported scene graph to {output_file}")
    print(f"  Nodes: {len(data['nodes'])}")
    print(f"  Edges: {len(data['edges'])}")
    print(f"  Keyframes: {sum(1 for n in data['nodes'] if n.get('is_keyframe', False))}")


def export_overlap_tree_json(overlap_info, output_file):
    """Export overlap tree information to JSON"""
    # Convert numpy types to native Python types
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    overlap_info = convert_numpy(overlap_info)
    
    with open(output_file, 'w') as f:
        json.dump(overlap_info, f, indent=2)
    
    print(f"Exported overlap tree info to {output_file}")


def visualize_scene_graph(G, output_file=None):
    """Visualize scene graph with matplotlib"""
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Position nodes based on timestamp
    pos = {}
    keyframes = []
    regular_frames = []
    
    for node_id, node_data in G.nodes(data=True):
        # Position based on node id (temporal order)
        pos[node_id] = (node_id, node_data.get('confidence', 0))
        
        if node_data.get('is_keyframe', False):
            keyframes.append(node_id)
        else:
            regular_frames.append(node_id)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5, arrows=False, ax=ax)
    
    # Draw regular frames
    if regular_frames:
        nx.draw_networkx_nodes(G, pos, nodelist=regular_frames, 
                               node_color='lightblue', node_size=20, 
                               alpha=0.6, ax=ax)
    
    # Draw keyframes
    if keyframes:
        nx.draw_networkx_nodes(G, pos, nodelist=keyframes, 
                               node_color='red', node_size=100, 
                               alpha=0.8, ax=ax)
    
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Confidence')
    ax.set_title('SLAM Scene Graph\n(Red: Keyframes, Blue: Regular Frames)')
    ax.grid(True, alpha=0.3)
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved scene graph visualization to {output_file}")
    else:
        plt.show()
    
    plt.close()


def export_statistics(results_dir, output_file):
    """Export comprehensive statistics"""
    poses_data, memory, keyframe_pointmaps, overlap_tree = load_slam_results(results_dir)
    
    poses = poses_data['poses']
    cam_centers = poses[:, :3, 3]
    displacements = np.linalg.norm(np.diff(cam_centers, axis=0), axis=1)
    
    overlap_info = extract_overlap_tree_info(overlap_tree)
    
    stats = {
        'trajectory': {
            'num_poses': int(len(poses)),
            'num_keyframes': int(len(keyframe_pointmaps)),
            'fps': float(poses_data['fps']),
            'gpu_memory_mb': float(poses_data['gpumem']),
            'total_path_length': float(displacements.sum()),
            'mean_displacement': float(displacements.mean()),
            'max_displacement': float(displacements.max()),
            'min_displacement': float(displacements.min()),
            'mean_confidence': float(poses_data['confs'].mean()),
            'std_confidence': float(poses_data['confs'].std()),
        },
        'overlap_tree': overlap_info,
        'keyframes': {
            'total': len(keyframe_pointmaps),
            'average_points_per_keyframe': int(np.mean([np.prod(pts.shape[:-1]) for pts, _, _ in keyframe_pointmaps]))
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Exported statistics to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Export SLAM scene graph and connectivity")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing all_poses.npz and memory.pkl")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for exports (default: same as results_dir)")
    parser.add_argument("--export_graph", action="store_true",
                        help="Export scene graph to JSON")
    parser.add_argument("--export_overlap", action="store_true",
                        help="Export overlap tree info to JSON")
    parser.add_argument("--export_stats", action="store_true",
                        help="Export statistics to JSON")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize scene graph")
    parser.add_argument("--all", action="store_true",
                        help="Export everything")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.results_dir
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading SLAM results from {args.results_dir}...")
    poses_data, memory, keyframe_pointmaps, overlap_tree = load_slam_results(args.results_dir)
    
    print(f"\nLoaded:")
    print(f"  Total poses: {len(poses_data['poses'])}")
    print(f"  Keyframes: {len(keyframe_pointmaps)}")
    
    # Build scene graph
    print("\nBuilding scene graph...")
    G = build_scene_graph(poses_data, memory, keyframe_pointmaps)
    
    # Extract overlap tree info
    overlap_info = extract_overlap_tree_info(overlap_tree)
    print(f"\nOverlap tree type: {overlap_info['type']}")
    if 'num_quadrants' in overlap_info:
        print(f"  Quadrants: {overlap_info['num_quadrants']}")
        active_quadrants = len(overlap_info['num_points_per_quadrant'])
        print(f"  Active quadrants: {active_quadrants}")
    
    # Exports
    if args.all or args.export_graph:
        print("\nExporting scene graph...")
        export_scene_graph_json(G, f"{args.output_dir}/scene_graph.json")
    
    if args.all or args.export_overlap:
        print("\nExporting overlap tree...")
        export_overlap_tree_json(overlap_info, f"{args.output_dir}/overlap_tree.json")
    
    if args.all or args.export_stats:
        print("\nExporting statistics...")
        export_statistics(args.results_dir, f"{args.output_dir}/statistics.json")
    
    if args.all or args.visualize:
        print("\nVisualizing scene graph...")
        output_file = f"{args.output_dir}/scene_graph.png" if args.all else None
        visualize_scene_graph(G, output_file)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
