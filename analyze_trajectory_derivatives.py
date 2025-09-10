#!/usr/bin/env python3
"""
Trajectory Derivative Analysis Script

This script loads trajectory data from stored JSON files, combines them into a 
single DataFrame, calculates position derivatives (velocity components), and 
creates histograms to visualize the distribution of these derivatives.

The derivatives represent the change in position between consecutive frames:
- Δx = x_{t+1} - x_t
- Δy = y_{t+1} - y_t

These values represent the instantaneous velocity components in pixels/frame.
"""

import json
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime


class TrajectoryDerivativeAnalyzer:
    """Analyzes trajectory derivatives from stored trajectory data."""
    
    def __init__(self, data_dir: str = "data/trajectories"):
        """
        Initialize the analyzer.
        
        Args:
            data_dir: Directory containing trajectory JSON files
        """
        self.data_dir = Path(data_dir)
        self.trajectories_df = None
        self.derivatives_df = None
        
    def load_trajectories(self, max_files: Optional[int] = None) -> pd.DataFrame:
        """
        Load trajectory data from JSON files and combine into a single DataFrame.
        
        Args:
            max_files: Maximum number of files to load (None for all)
            
        Returns:
            DataFrame with columns: track_id, frame_id, x, y, session_id
        """
        json_files = sorted(glob.glob(str(self.data_dir / "*.json")))
        
        if not json_files:
            raise ValueError(f"No JSON files found in {self.data_dir}")
        
        if max_files:
            json_files = json_files[:max_files]
            
        print(f"Loading {len(json_files)} trajectory files...")
        print(f"This may take a while for large datasets...")
        
        all_trajectories = []
        
        for idx, file_path in enumerate(json_files, 1):
            if idx % 10 == 0 or idx == 1:
                print(f"  Processing file {idx}/{len(json_files)}: {Path(file_path).name}")
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                session_id = data.get('session_id', Path(file_path).stem)
                trajectories = data.get('trajectories', {})
                
                points_in_file = 0
                for track_id, track_points in trajectories.items():
                    for point in track_points:
                        if 'center' in point:
                            all_trajectories.append({
                                'track_id': f"{session_id}_{track_id}",
                                'frame_id': point['frame_id'],
                                'x': point['center'][0],
                                'y': point['center'][1],
                                'session_id': session_id
                            })
                            points_in_file += 1
                
                if idx % 10 == 0:
                    print(f"    -> Added {points_in_file} points from this file (total so far: {len(all_trajectories)})")
                            
            except Exception as e:
                print(f"  Error loading file {idx}: {Path(file_path).name}: {e}")
                continue
        
        if not all_trajectories:
            raise ValueError("No trajectory data could be loaded")
        
        print(f"\nCreating DataFrame and sorting data...")
        self.trajectories_df = pd.DataFrame(all_trajectories)
        self.trajectories_df = self.trajectories_df.sort_values(['track_id', 'frame_id'])
        
        print(f"✓ Successfully loaded {len(self.trajectories_df)} trajectory points from {self.trajectories_df['track_id'].nunique()} tracks")
        
        return self.trajectories_df
    
    def calculate_derivatives(self) -> pd.DataFrame:
        """
        Calculate position derivatives (Δx, Δy) for each track.
        
        Returns:
            DataFrame with derivative columns added
        """
        if self.trajectories_df is None:
            raise ValueError("No trajectories loaded. Call load_trajectories() first.")
        
        print("Calculating derivatives...")
        
        derivatives_list = []
        unique_tracks = self.trajectories_df['track_id'].unique()
        total_tracks = len(unique_tracks)
        
        print(f"Processing {total_tracks} unique tracks...")
        
        for idx, track_id in enumerate(unique_tracks, 1):
            if idx % 100 == 0 or idx == 1:
                print(f"  Processing track {idx}/{total_tracks} ({idx/total_tracks*100:.1f}%)")
            track_data = self.trajectories_df[self.trajectories_df['track_id'] == track_id].copy()
            track_data = track_data.sort_values('frame_id')
            
            if len(track_data) < 2:
                continue
            
            track_data['dx'] = track_data['x'].diff()
            track_data['dy'] = track_data['y'].diff()
            
            track_data['velocity_magnitude'] = np.sqrt(track_data['dx']**2 + track_data['dy']**2)
            
            derivatives_list.append(track_data[1:])
        
        if not derivatives_list:
            raise ValueError("No valid derivatives could be calculated")
        
        print(f"  Combining all derivative data...")
        self.derivatives_df = pd.concat(derivatives_list, ignore_index=True)
        
        valid_derivatives = self.derivatives_df.dropna(subset=['dx', 'dy'])
        print(f"✓ Successfully calculated {len(valid_derivatives)} derivative values")
        
        return self.derivatives_df
    
    def plot_histograms(self, bins: int = 50, save_path: Optional[str] = None, limit: Optional[float] = None):
        """
        Create histograms of the position derivatives.
        
        Args:
            bins: Number of bins for the histograms
            save_path: Path to save the figure (None to display only)
            limit: Optional limit for the range of values to plot (e.g., 25 for [-25, 25])
        """
        if self.derivatives_df is None:
            raise ValueError("No derivatives calculated. Call calculate_derivatives() first.")
        
        valid_data = self.derivatives_df.dropna(subset=['dx', 'dy'])
        
        # Note: Statistics are calculated from ALL data, limit only affects visualization range
        if limit is not None:
            print(f"Applying visualization limit: displaying range [-{limit}, {limit}]")
            print(f"Note: Statistics are calculated from the full dataset")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Trajectory Derivative Analysis', fontsize=16, fontweight='bold')
        
        # Determine histogram range
        dx_range = (-limit, limit) if limit else (valid_data['dx'].min(), valid_data['dx'].max())
        
        ax1 = axes[0, 0]
        ax1.hist(valid_data['dx'], bins=bins, alpha=0.7, color='blue', edgecolor='black', range=dx_range)
        ax1.set_xlabel('Δx (pixels/frame)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Distribution of X-axis Derivatives', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        dx_mean = valid_data['dx'].mean()
        dx_std = valid_data['dx'].std()
        ax1.axvline(dx_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {dx_mean:.2f}')
        ax1.axvline(dx_mean + dx_std, color='orange', linestyle=':', linewidth=1, label=f'±σ: {dx_std:.2f}')
        ax1.axvline(dx_mean - dx_std, color='orange', linestyle=':', linewidth=1)
        ax1.legend()
        if limit:
            ax1.set_xlim(-limit, limit)
        
        # Determine histogram range for dy
        dy_range = (-limit, limit) if limit else (valid_data['dy'].min(), valid_data['dy'].max())
        
        ax2 = axes[0, 1]
        ax2.hist(valid_data['dy'], bins=bins, alpha=0.7, color='green', edgecolor='black', range=dy_range)
        ax2.set_xlabel('Δy (pixels/frame)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Distribution of Y-axis Derivatives', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        dy_mean = valid_data['dy'].mean()
        dy_std = valid_data['dy'].std()
        ax2.axvline(dy_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {dy_mean:.2f}')
        ax2.axvline(dy_mean + dy_std, color='orange', linestyle=':', linewidth=1, label=f'±σ: {dy_std:.2f}')
        ax2.axvline(dy_mean - dy_std, color='orange', linestyle=':', linewidth=1)
        ax2.legend()
        if limit:
            ax2.set_xlim(-limit, limit)
        
        ax3 = axes[1, 0]
        # Apply range to 2D histogram if limit is set
        if limit:
            h = ax3.hist2d(valid_data['dx'], valid_data['dy'], bins=bins, cmap='YlOrRd', 
                          range=[[-limit, limit], [-limit, limit]])
        else:
            h = ax3.hist2d(valid_data['dx'], valid_data['dy'], bins=bins, cmap='YlOrRd')
        ax3.set_xlabel('Δx (pixels/frame)', fontsize=12)
        ax3.set_ylabel('Δy (pixels/frame)', fontsize=12)
        ax3.set_title('2D Distribution of Derivatives', fontsize=14)
        plt.colorbar(h[3], ax=ax3, label='Frequency')
        
        ax4 = axes[1, 1]
        # Apply range to velocity magnitude histogram if limit is set
        vel_range = (0, limit) if limit else (0, valid_data['velocity_magnitude'].max())
        ax4.hist(valid_data['velocity_magnitude'], bins=bins, alpha=0.7, color='purple', edgecolor='black', range=vel_range)
        ax4.set_xlabel('Velocity Magnitude (pixels/frame)', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title('Distribution of Velocity Magnitude', fontsize=14)
        ax4.grid(True, alpha=0.3)
        
        # Calculate statistics from ALL data
        vel_mean = valid_data['velocity_magnitude'].mean()
        vel_std = valid_data['velocity_magnitude'].std()
        ax4.axvline(vel_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {vel_mean:.2f}')
        ax4.axvline(vel_mean + vel_std, color='orange', linestyle=':', linewidth=1, label=f'±σ: {vel_std:.2f}')
        ax4.legend()
        if limit:
            ax4.set_xlim(0, limit)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def get_statistics(self) -> Dict:
        """
        Calculate and return statistics about the derivatives.
        
        Returns:
            Dictionary containing various statistics
        """
        if self.derivatives_df is None:
            raise ValueError("No derivatives calculated. Call calculate_derivatives() first.")
        
        valid_data = self.derivatives_df.dropna(subset=['dx', 'dy'])
        
        stats = {
            'total_tracks': self.trajectories_df['track_id'].nunique(),
            'total_points': len(self.trajectories_df),
            'total_derivatives': len(valid_data),
            'dx': {
                'mean': valid_data['dx'].mean(),
                'std': valid_data['dx'].std(),
                'min': valid_data['dx'].min(),
                'max': valid_data['dx'].max(),
                'median': valid_data['dx'].median(),
                'q25': valid_data['dx'].quantile(0.25),
                'q75': valid_data['dx'].quantile(0.75)
            },
            'dy': {
                'mean': valid_data['dy'].mean(),
                'std': valid_data['dy'].std(),
                'min': valid_data['dy'].min(),
                'max': valid_data['dy'].max(),
                'median': valid_data['dy'].median(),
                'q25': valid_data['dy'].quantile(0.25),
                'q75': valid_data['dy'].quantile(0.75)
            },
            'velocity_magnitude': {
                'mean': valid_data['velocity_magnitude'].mean(),
                'std': valid_data['velocity_magnitude'].std(),
                'min': valid_data['velocity_magnitude'].min(),
                'max': valid_data['velocity_magnitude'].max(),
                'median': valid_data['velocity_magnitude'].median(),
                'q25': valid_data['velocity_magnitude'].quantile(0.25),
                'q75': valid_data['velocity_magnitude'].quantile(0.75)
            }
        }
        
        return stats
    
    def print_statistics(self):
        """Print formatted statistics about the derivatives."""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("TRAJECTORY DERIVATIVE STATISTICS")
        print("="*60)
        print(f"Total Tracks: {stats['total_tracks']:,}")
        print(f"Total Points: {stats['total_points']:,}")
        print(f"Total Derivatives: {stats['total_derivatives']:,}")
        
        print("\n--- X-axis Derivatives (Δx) ---")
        print(f"Mean: {stats['dx']['mean']:.3f} pixels/frame")
        print(f"Std Dev: {stats['dx']['std']:.3f}")
        print(f"Min: {stats['dx']['min']:.3f}")
        print(f"Max: {stats['dx']['max']:.3f}")
        print(f"Median: {stats['dx']['median']:.3f}")
        print(f"Q25-Q75: [{stats['dx']['q25']:.3f}, {stats['dx']['q75']:.3f}]")
        
        print("\n--- Y-axis Derivatives (Δy) ---")
        print(f"Mean: {stats['dy']['mean']:.3f} pixels/frame")
        print(f"Std Dev: {stats['dy']['std']:.3f}")
        print(f"Min: {stats['dy']['min']:.3f}")
        print(f"Max: {stats['dy']['max']:.3f}")
        print(f"Median: {stats['dy']['median']:.3f}")
        print(f"Q25-Q75: [{stats['dy']['q25']:.3f}, {stats['dy']['q75']:.3f}]")
        
        print("\n--- Velocity Magnitude ---")
        print(f"Mean: {stats['velocity_magnitude']['mean']:.3f} pixels/frame")
        print(f"Std Dev: {stats['velocity_magnitude']['std']:.3f}")
        print(f"Min: {stats['velocity_magnitude']['min']:.3f}")
        print(f"Max: {stats['velocity_magnitude']['max']:.3f}")
        print(f"Median: {stats['velocity_magnitude']['median']:.3f}")
        print(f"Q25-Q75: [{stats['velocity_magnitude']['q25']:.3f}, {stats['velocity_magnitude']['q75']:.3f}]")
        print("="*60)


def main():
    """Main function to run the trajectory derivative analysis."""
    parser = argparse.ArgumentParser(description='Analyze trajectory derivatives and create histograms')
    parser.add_argument('--data-dir', type=str, default='data/trajectories',
                        help='Directory containing trajectory JSON files')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of files to load (None for all)')
    parser.add_argument('--bins', type=int, default=50,
                        help='Number of bins for histograms')
    parser.add_argument('--limit', type=float, default=None,
                        help='Limit for visualization range (e.g., 25 for [-25, 25] on x/y axes, [0, 25] for velocity)')
    parser.add_argument('--save-path', type=str, default=None,
                        help='Path to save the histogram figure')
    parser.add_argument('--export-csv', type=str, default=None,
                        help='Path to export derivatives data as CSV')
    
    args = parser.parse_args()
    
    analyzer = TrajectoryDerivativeAnalyzer(data_dir=args.data_dir)
    
    try:
        analyzer.load_trajectories(max_files=args.max_files)
        
        analyzer.calculate_derivatives()
        
        analyzer.print_statistics()
        
        analyzer.plot_histograms(bins=args.bins, save_path=args.save_path, limit=args.limit)
        
        if args.export_csv:
            analyzer.derivatives_df.to_csv(args.export_csv, index=False)
            print(f"\nDerivatives data exported to {args.export_csv}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())