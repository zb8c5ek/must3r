#!/usr/bin/env python3
# Copyright (C) 2025-present Naver Corporation. All rights reserved.
"""
Visualize SLAM results from saved memory.pkl and all_poses.npz
"""
import argparse
import numpy as np
import open3d as o3d
import pickle
import torch


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


def extract_point_cloud_from_keyframes(keyframe_pointmaps, conf_threshold=3.0):
    """Extract 3D points and colors from keyframe pointmaps"""
    all_pts = []
    all_colors = []
    
    print(f"Processing {len(keyframe_pointmaps)} keyframes...")
    
    for i, (pts3d, colors, conf) in enumerate(keyframe_pointmaps):
        # Convert to numpy if needed
        if torch.is_tensor(pts3d):
            pts3d = pts3d.cpu().numpy()
        if torch.is_tensor(colors):
            colors = colors.cpu().numpy()
        if torch.is_tensor(conf):
            conf = conf.cpu().numpy()
        
        # Flatten if needed (they might be [H, W, C] format)
        if pts3d.ndim == 3:
            H, W = pts3d.shape[:2]
            pts3d = pts3d.reshape(-1, 3)
            colors = colors.reshape(-1, 3)
            conf = conf.reshape(-1)
        
        # Apply confidence threshold
        mask = conf > conf_threshold
        pts_masked = pts3d[mask]
        colors_masked = colors[mask]
        
        all_pts.append(pts_masked)
        all_colors.append(colors_masked)
        
        if i % 10 == 0 or i == len(keyframe_pointmaps) - 1:
            print(f"  Processed keyframe {i+1}/{len(keyframe_pointmaps)}: {pts_masked.shape[0]:,} points")
    
    # Concatenate all points
    if all_pts:
        all_pts = np.concatenate(all_pts, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)
    else:
        all_pts = np.zeros((0, 3))
        all_colors = np.zeros((0, 3))
    
    return all_pts, all_colors


def create_camera_trajectory(poses):
    """Create camera trajectory line set"""
    # Extract camera centers (translation part of c2w matrices)
    cam_centers = poses[:, :3, 3]
    
    # Create line set for trajectory
    points = cam_centers
    lines = [[i, i+1] for i in range(len(points)-1)]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in lines])  # Red trajectory
    
    return line_set, cam_centers


def create_camera_frustums(poses, scale=0.1, subsample=1):
    """Create camera frustums for visualization"""
    frustums = []
    sampled_poses = poses[::subsample]
    
    for i, c2w in enumerate(sampled_poses):
        # Create simple camera marker (coordinate frame)
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
        mesh.transform(c2w)
        frustums.append(mesh)
    
    return frustums


def visualize_slam_results(results_dir, conf_threshold=3.0, show_cameras=True, 
                          show_trajectory=True, camera_scale=0.1, camera_subsample=10):
    """Main visualization function"""
    print(f"Loading SLAM results from {results_dir}...")
    
    # Load data
    poses_data, memory, keyframe_pointmaps, overlap_tree = load_slam_results(results_dir)
    
    print(f"\nLoaded:")
    print(f"  Total poses: {len(poses_data['poses'])}")
    print(f"  Keyframes with pointmaps: {len(keyframe_pointmaps)}")
    print(f"  FPS: {poses_data['fps']:.2f}")
    print(f"  GPU Memory: {poses_data['gpumem']:.2f} MB")
    
    # Extract point cloud from keyframes
    print(f"\nExtracting point cloud (conf_threshold={conf_threshold})...")
    points, colors = extract_point_cloud_from_keyframes(keyframe_pointmaps, conf_threshold)
    print(f"Total points after filtering: {len(points):,}")
    
    if len(points) == 0:
        print("\nWARNING: No points found! Try lowering --conf_threshold")
        return
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create visualization list
    geometries = [pcd]
    
    poses = poses_data['poses']
    
    # Add camera trajectory
    if show_trajectory:
        print("\nCreating camera trajectory...")
        line_set, cam_centers = create_camera_trajectory(poses)
        geometries.append(line_set)
        
        # Compute trajectory stats
        displacements = np.linalg.norm(np.diff(cam_centers, axis=0), axis=1)
        total_distance = displacements.sum()
        print(f"  Trajectory: {len(cam_centers)} poses, total length: {total_distance:.2f} units")
    
    # Add camera frustums
    if show_cameras:
        print(f"\nCreating camera frustums (subsampling every {camera_subsample} frames)...")
        frustums = create_camera_frustums(poses, scale=camera_scale, subsample=camera_subsample)
        geometries.extend(frustums)
        print(f"  Added {len(frustums)} camera frustums")
    
    # Add coordinate frame at origin
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    geometries.append(origin_frame)
    
    # Visualize
    print("\nLaunching visualizer...")
    print("Controls:")
    print("  - Mouse: Rotate view")
    print("  - Scroll: Zoom")
    print("  - Ctrl+Mouse: Pan")
    print("  - Red line: Camera trajectory")
    print("  - RGB axes: Camera orientations")
    
    o3d.visualization.draw_geometries(geometries,
                                      window_name="SLAM 3D Reconstruction",
                                      width=1920,
                                      height=1080,
                                      left=50,
                                      top=50)


def export_ply(results_dir, output_ply, conf_threshold=3.0):
    """Export point cloud to PLY file"""
    print(f"Loading and exporting to {output_ply}...")
    
    poses_data, memory, keyframe_pointmaps, overlap_tree = load_slam_results(results_dir)
    points, colors = extract_point_cloud_from_keyframes(keyframe_pointmaps, conf_threshold)
    
    if len(points) == 0:
        print(f"ERROR: No points found with conf_threshold={conf_threshold}!")
        print("Try lowering the threshold.")
        return
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.io.write_point_cloud(output_ply, pcd)
    print(f"Saved point cloud with {len(points):,} points to {output_ply}")


def print_statistics(results_dir):
    """Print detailed statistics"""
    poses_data, memory, keyframe_pointmaps, overlap_tree = load_slam_results(results_dir)
    poses = poses_data['poses']
    cam_centers = poses[:, :3, 3]
    
    # Compute statistics
    displacements = np.linalg.norm(np.diff(cam_centers, axis=0), axis=1)
    total_distance = displacements.sum()
    
    # Count points at different thresholds
    print(f"\n{'='*70}")
    print(f"SLAM RESULTS STATISTICS")
    print(f"{'='*70}")
    print(f"\nTrajectory:")
    print(f"  Total poses:          {len(poses)}")
    print(f"  Keyframes:            {len(keyframe_pointmaps)}")
    print(f"  FPS:                  {poses_data['fps']:.2f}")
    print(f"  GPU Memory:           {poses_data['gpumem']:.2f} MB")
    print(f"  Total path length:    {total_distance:.3f} units")
    print(f"  Mean step size:       {displacements.mean():.4f} units")
    print(f"  Max step size:        {displacements.max():.4f} units")
    print(f"  Mean confidence:      {poses_data['confs'].mean():.3f}")
    
    print(f"\nPoint cloud at different confidence thresholds:")
    for thr in [1.0, 2.0, 3.0, 4.0, 5.0]:
        pts, _ = extract_point_cloud_from_keyframes(keyframe_pointmaps, thr)
        print(f"  conf > {thr:.1f}:  {len(pts):>12,} points")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SLAM results")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing all_poses.npz and memory.pkl")
    parser.add_argument("--conf_threshold", type=float, default=3.0,
                        help="Confidence threshold for filtering points (default: 3.0)")
    parser.add_argument("--no_cameras", action="store_true",
                        help="Don't show camera frustums")
    parser.add_argument("--no_trajectory", action="store_true",
                        help="Don't show camera trajectory")
    parser.add_argument("--camera_scale", type=float, default=0.1,
                        help="Size of camera frustums (default: 0.1)")
    parser.add_argument("--camera_subsample", type=int, default=10,
                        help="Show every Nth camera (default: 10)")
    parser.add_argument("--export_ply", type=str, default=None,
                        help="Export point cloud to PLY file instead of visualizing")
    parser.add_argument("--stats", action="store_true",
                        help="Print statistics and exit")
    
    args = parser.parse_args()
    
    if args.stats:
        print_statistics(args.results_dir)
    elif args.export_ply:
        export_ply(args.results_dir, args.export_ply, args.conf_threshold)
    else:
        visualize_slam_results(args.results_dir, 
                              args.conf_threshold,
                              show_cameras=not args.no_cameras,
                              show_trajectory=not args.no_trajectory,
                              camera_scale=args.camera_scale,
                              camera_subsample=args.camera_subsample)
