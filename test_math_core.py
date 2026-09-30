import numpy as np
from pegasus_chronos.chronos_math.topology import SheavedTopologyDetector

# Create dummy detector
detector = SheavedTopologyDetector(n_neighbors=5)

# Simulate 50 benign memories (clustered)
benign_vectors = np.random.normal(loc=0.0, scale=1.0, size=(50, 768))

# Simulate 5 Ghost State memories (steganographically shifted)
ghost_vectors = np.random.normal(loc=0.5, scale=1.0, size=(5, 768))

# Compute Dirichlet Energy
# Build manifold on benign memories, then measure localized Dirichlet energy
detector.build_manifold_graph(benign_vectors)
energies_b = detector.compute_localized_dirichlet_energy(benign_vectors)
energies_g = detector.compute_localized_dirichlet_energy(ghost_vectors)

print(f"Benign Mean Energy: {energies_b.mean():.4f}")
print(f"Ghost Mean Energy: {energies_g.mean():.4f}")
print(f"Separation Ratio: {energies_g.mean() / energies_b.mean():.4f}")
