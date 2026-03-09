"""
Cosmological Structure Formation Module
Enterprise-grade gravitational clustering simulation for feature clusters.
"""

import torch
import numpy as np
from typing import Union, Tuple, List, Optional, Dict, Any
from scipy.spatial.distance import pdist, squareform
import logging
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class CosmoConfig:
    """Configuration for cosmological simulation."""
    grid_size: Tuple[int, int] = (64, 64)
    gravitational_constant: float = 1.0
    device: str = 'cpu'
    softening_length: float = 0.1
    time_step: float = 0.1
    max_velocity: float = 10.0


class CosmologicalStructureModule:
    """Simulates gravitational clustering of feature clusters."""

    def __init__(
        self,
        grid_size: Tuple[int, int] = (64, 64),
        G: float = 1.0,
        device: str = 'cpu',
        config: Optional[CosmoConfig] = None
    ):
        """
        Initialize the cosmological structure module.

        Args:
            grid_size: Simulation grid size (height, width).
            G: Gravitational constant (normalized).
            device: Device to run computations on.
            config: Optional CosmoConfig object.
        """
        if config is not None:
            self.grid_size = config.grid_size
            self.G = config.gravitational_constant
            self.device = torch.device(config.device)
            self.config = config
        else:
            self.grid_size = grid_size
            self.G = G
            self.device = torch.device(device)
            self.config = CosmoConfig(
                grid_size=grid_size,
                gravitational_constant=G,
                device=device
            )

        # Validate grid size
        if len(self.grid_size) != 2:
            raise ValueError("grid_size must be a tuple of (height, width)")
        if self.grid_size[0] <= 0 or self.grid_size[1] <= 0:
            raise ValueError("Grid dimensions must be positive")

        logger.info(f"Initialized CosmologicalStructureModule with grid: {self.grid_size}")
        logger.info(f"Gravitational constant: {self.G}")

    def initialize_clusters(
        self,
        n_clusters: int,
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize cluster positions and masses.

        Args:
            n_clusters: Number of clusters to initialize.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (positions, masses) tensors.
        """
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer")

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            logger.debug(f"Set random seed: {seed}")

        height, width = self.grid_size

        # Initialize positions randomly within grid
        positions = torch.rand(n_clusters, 2, device=self.device) * torch.tensor(
            [width, height], device=self.device
        )

        # Initialize masses between 1-11
        masses = torch.rand(n_clusters, device=self.device) * 10.0 + 1.0

        logger.info(f"Initialized {n_clusters} clusters")
        logger.debug(f"Position range: [{positions.min():.2f}, {positions.max():.2f}]")
        logger.debug(f"Mass range: [{masses.min():.2f}, {masses.max():.2f}]")

        return positions, masses

    def gravitational_force(
        self,
        positions: torch.Tensor,
        masses: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gravitational forces between clusters using vectorized operations.

        Args:
            positions: Cluster positions tensor [N, 2].
            masses: Cluster masses tensor [N].

        Returns:
            Force tensor [N, 2].
        """
        if not isinstance(positions, torch.Tensor):
            raise TypeError("positions must be a torch.Tensor")
        if not isinstance(masses, torch.Tensor):
            raise TypeError("masses must be a torch.Tensor")

        n_clusters = len(positions)

        if n_clusters == 0:
            return torch.zeros((0, 2), device=self.device)

        # Vectorized force calculation
        # Compute pairwise differences
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # [N, N, 2]

        # Compute distances with softening
        distances = torch.norm(diff, dim=2)  # [N, N]
        distances_softened = torch.sqrt(
            distances ** 2 + self.config.softening_length ** 2
        )

        # Compute force magnitudes (inverse square law with softening)
        # F = G * m1 * m2 / r^2
        force_magnitudes = (
            self.G * masses.unsqueeze(1) * masses.unsqueeze(0) /
            (distances_softened ** 2 + 1e-10)  # Avoid division by zero
        )

        # Zero out self-interaction
        mask = torch.eye(n_clusters, device=self.device, dtype=torch.bool)
        force_magnitudes = force_magnitudes.masked_fill(mask, 0.0)

        # Compute force vectors
        # Normalize direction vectors
        directions = diff / (distances_softened.unsqueeze(2) + 1e-10)

        # Force vectors
        forces = force_magnitudes.unsqueeze(2) * directions

        # Sum forces for each cluster
        total_forces = forces.sum(dim=1)

        logger.debug(f"Force magnitudes: [{total_forces.norm(dim=1).min():.4f}, "
                    f"{total_forces.norm(dim=1).max():.4f}]")

        return total_forces

    def simulate_clustering(
        self,
        n_clusters: int = 10,
        steps: int = 50,
        dt: Optional[float] = None,
        seed: Optional[int] = None
    ) -> List[torch.Tensor]:
        """
        Simulate gravitational clustering over time using leapfrog integration.

        Args:
            n_clusters: Number of clusters.
            steps: Number of simulation steps.
            dt: Time step (uses config default if None).
            seed: Random seed.

        Returns:
            List of position tensors at each time step.
        """
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer")
        if not isinstance(steps, int) or steps <= 0:
            raise ValueError("steps must be a positive integer")

        dt = dt if dt is not None else self.config.time_step

        logger.info(f"Starting simulation: {n_clusters} clusters, {steps} steps")

        # Initialize
        positions, masses = self.initialize_clusters(n_clusters, seed)
        velocities = torch.zeros_like(positions)

        trajectory = [positions.clone()]

        # Simulation loop with velocity Verlet integration
        for step in range(steps):
            # Compute accelerations
            forces = self.gravitational_force(positions, masses)
            accelerations = forces / masses.unsqueeze(-1)

            # Update velocities (half step)
            velocities_half = velocities + 0.5 * accelerations * dt

            # Update positions
            positions = positions + velocities_half * dt

            # Compute new accelerations
            forces_new = self.gravitational_force(positions, masses)
            accelerations_new = forces_new / masses.unsqueeze(-1)

            # Update velocities (full step)
            velocities = velocities_half + 0.5 * accelerations_new * dt

            # Apply velocity limits
            velocities = torch.clamp(velocities, -self.config.max_velocity, self.config.max_velocity)

            # Keep positions within bounds (periodic boundary conditions optional)
            positions[:, 0] = torch.clamp(positions[:, 0], 0, self.grid_size[1])
            positions[:, 1] = torch.clamp(positions[:, 1], 0, self.grid_size[0])

            # Store trajectory
            if (step + 1) % max(1, steps // 10) == 0:
                logger.debug(f"Step {step + 1}/{steps} completed")

            trajectory.append(positions.clone())

        logger.info(f"Simulation completed. Final positions: {positions.shape}")

        return trajectory

    def get_cluster_statistics(
        self,
        positions: torch.Tensor,
        masses: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Calculate statistics for cluster configuration.

        Args:
            positions: Cluster positions.
            masses: Cluster masses.

        Returns:
            Dictionary of statistics.
        """
        # Center of mass
        total_mass = masses.sum()
        center_of_mass = (positions * masses.unsqueeze(1)).sum(dim=0) / total_mass

        # Average distance from center
        distances_from_center = torch.norm(positions - center_of_mass, dim=1)
        avg_distance = distances_from_center.mean().item()

        # Total kinetic energy (assuming unit velocity for simplicity)
        # In real simulation, you'd use actual velocities

        stats = {
            'n_clusters': len(positions),
            'total_mass': total_mass.item(),
            'center_of_mass': center_of_mass.cpu().numpy().tolist(),
            'avg_distance_from_center': avg_distance,
            'grid_size': list(self.grid_size)
        }

        logger.debug(f"Cluster statistics: {stats}")

        return stats

    def save_trajectory(
        self,
        trajectory: List[torch.Tensor],
        masses: torch.Tensor,
        filepath: Union[str, Path]
    ) -> None:
        """
        Save trajectory to file.

        Args:
            trajectory: List of position tensors.
            masses: Cluster masses.
            filepath: Output file path.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert to numpy and save
        trajectory_np = [t.cpu().numpy() for t in trajectory]
        masses_np = masses.cpu().numpy()

        np.savez_compressed(
            filepath,
            trajectory=np.array(trajectory_np),
            masses=masses_np,
            grid_size=np.array(self.grid_size)
        )

        logger.info(f"Saved trajectory to {filepath}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    module = CosmologicalStructureModule(grid_size=(64, 64))
    trajectory = module.simulate_clustering(n_clusters=8, steps=20, seed=42)

    final_positions = trajectory[-1]
    print(f"Final cluster positions shape: {final_positions.shape}")
    print(f"Number of time steps: {len(trajectory)}")

    # Get statistics
    _, masses = module.initialize_clusters(8, seed=42)
    stats = module.get_cluster_statistics(final_positions, masses)
    print(f"Statistics: {stats}")
