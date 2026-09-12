"""WyckoffGNN — Crystal Representation Learning on Wyckoff Orbit Instances.

Nodes = occupied Wyckoff orbit instances.
Dual edges = geometric (inter-orbit distances) + symmetry (site-symmetry hierarchy).
SE(3)-equivariant e3nn message passing with Clebsch–Gordan tensor products.
Stage-A: template prior over 230-sg Wyckoff template graph.
Stage-B: equivariant MP over occupied Wyckoff orbit instance graph.
"""

__version__ = "0.2.0"
