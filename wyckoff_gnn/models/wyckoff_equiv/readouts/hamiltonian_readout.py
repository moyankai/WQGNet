"""Hamiltonian head — orbit-pair block prediction (Phase 5 MVP).

Predicts local Hamiltonian matrix blocks between Wyckoff orbits:

    H_{i mu, j nu}(R)

where:
    i, j = Wyckoff orbit (node) indices
    mu, nu = local orbital basis indices (s, p_x, p_y, p_z, d_xy, ..., d_z2)
    R = lattice translation (encoded via edge shift + source image)

Two block types:
    Onsite  (i = j, R = 0):  H_ii = symmetrize under site-symmetry group H_i
    Offsite (i != j or R != 0): H_ij(R) = f(h_i, h_j, edge features), enforce
                                Hermiticity H_ji(-R) = H_ij(R)^†

Bloch Hamiltonian:
    H(k) = sum_R  H(R) exp(i k · R)

Phase-5 hardening scope (documented explicitly):

* **Onsite symmetrization** takes a *padded* stabilizer rep tensor plus a
  boolean mask so different sites can have different stabilizer sizes without
  the caller having to pad with identities (which would double-count and
  break the Reynolds average).
* **Offsite Hermiticity** consumes a ``reverse_edge_map`` built by
  :func:`find_reverse_edge_map`, so callers do not need to hand-align the
  forward/reverse edge slots.
* **Bloch construction** takes explicit ``k_coord_type`` (``"cartesian"`` |
  ``"fractional"``) and ``include_2pi`` flags so the exponent's units are
  unambiguous. There is **no default guess** — omitting them is fine because
  we ship a documented default (``cartesian``, ``include_2pi=False``) that
  matches the previous behavior, but callers should set them explicitly.
* **Space-group covariance of offsite blocks is not enforced** by this head.
  Applying a crystal operation ``g`` should send
  ``H_ij(R)`` to ``U_i(g) H_{g·i, g·j}(g·R) U_j(g)^†``; the MLP here has no
  such symmetry projection and is only invariant to permutation-consistent
  relabelling of ``(i, j, R)``.
* **Input features for offsite blocks must be local edge features** (invariant
  in ``(node_i, node_j, edge_features)`` under proper rotations). This head
  does not rotate them into a common frame.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from e3nn import o3


__all__ = [
    "WyckoffHamiltonianHead",
    "symmetrize_onsite_block",
    "enforce_offsite_hermiticity",
    "build_bloch_hamiltonian",
    "hermitize_bloch",
    "basis_dim",
    "orbital_rep_matrix",
    "find_reverse_edge_map",
    "hermitize_with_reverse_map",
]


# ---------------------------------------------------------------------------
# Basis dimensions
# ---------------------------------------------------------------------------
# For the MVP we support fixed s/p/d local basis:
#     s: 1 orbital (l=0)
#     p: 3 orbitals (l=1)
#     d: 5 orbitals (l=2)
#     spd: 9 orbitals total
# Real-valued Hamiltonian throughout (no spin-orbit).

_BASIS_DIMS = {
    "s": 1,
    "p": 3,
    "d": 5,
    "sp": 4,
    "spd": 9,
}

# For each basis, the (l, parity) blocks the basis contains, in the order
# they occupy rows/columns of the block. Parity is +1 for s/d (even) and
# -1 for p (odd), matching the physical parity of hydrogenic orbitals.
_BASIS_BLOCKS: Dict[str, List[Tuple[int, int]]] = {
    "s": [(0, +1)],
    "p": [(1, -1)],
    "d": [(2, +1)],
    "sp": [(0, +1), (1, -1)],
    "spd": [(0, +1), (1, -1), (2, +1)],
}


def basis_dim(basis: str) -> int:
    """Return the number of orbitals per site for a named basis."""
    if basis not in _BASIS_DIMS:
        raise ValueError(f"Unknown basis: {basis}. Options: {list(_BASIS_DIMS)}")
    return _BASIS_DIMS[basis]


# ---------------------------------------------------------------------------
# E1: orbital rep matrix
# ---------------------------------------------------------------------------

def orbital_rep_matrix(R_cart: torch.Tensor, basis: str) -> torch.Tensor:
    """Assemble the block-diagonal orbital-basis representation D(R) for
    a real 3x3 Cartesian rotation ``R_cart`` and the requested atomic basis.

    Each (l, parity) block of the basis contributes a Wigner-D matrix
    ``D^{(l, parity)}(R)`` (real spherical-harmonic convention, following
    e3nn). For an improper rotation (det R = -1) the block is multiplied by
    ``parity`` — s and d blocks are invariant under inversion, p blocks
    flip sign.

    Args:
        R_cart: (3, 3) or (N, 3, 3) real Cartesian rotation matrix (proper
            or improper).
        basis: name of the atomic basis (see :data:`_BASIS_BLOCKS`).

    Returns:
        (n_orb, n_orb) or (N, n_orb, n_orb) real block-diagonal rep matrix,
        where ``n_orb = basis_dim(basis)``.
    """
    if basis not in _BASIS_BLOCKS:
        raise ValueError(f"Unknown basis: {basis!r}. Options: {list(_BASIS_BLOCKS)}")

    if R_cart.dim() not in (2, 3):
        raise ValueError(f"R_cart must have shape (3,3) or (N,3,3); got {tuple(R_cart.shape)}")
    single = R_cart.dim() == 2
    R = R_cart.unsqueeze(0) if single else R_cart
    N = R.shape[0]
    n_orb = _BASIS_DIMS[basis]

    D = torch.zeros(N, n_orb, n_orb, dtype=R.dtype, device=R.device)
    offset = 0
    for l, p in _BASIS_BLOCKS[basis]:
        ir = o3.Irrep(l, p)
        D_block = ir.D_from_matrix(R)  # (N, 2l+1, 2l+1)
        d = 2 * l + 1
        D[:, offset:offset + d, offset:offset + d] = D_block
        offset += d
    return D.squeeze(0) if single else D


# ---------------------------------------------------------------------------
# Onsite symmetrization  (E2: masked / padded)
# ---------------------------------------------------------------------------

def symmetrize_onsite_block(
    A: torch.Tensor,
    site_rep_matrices: torch.Tensor,
    op_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Symmetrize an onsite Hamiltonian block over the site-symmetry group.

    Pi_H(A) = (1/|H|) sum_{g in H}  U(g) A U(g)^dagger

    Args:
        A: (..., n, n) raw block (per orbit or per batch of orbits).
        site_rep_matrices: (n_ops, n, n) OR (..., n_ops, n, n) orbital-basis
            representation matrices of the stabilizer group. Real-valued.
        op_mask: optional (n_ops,) or (..., n_ops) boolean mask marking which
            operations are real (True). Padded slots (False) are ignored in
            both the sum and the normalization, so different sites can be
            padded to the same n_ops without contaminating the Reynolds
            average.

    Returns:
        (..., n, n) symmetrized block.
    """
    if site_rep_matrices.numel() == 0 or site_rep_matrices.shape[-3] == 0:
        return A

    U = site_rep_matrices
    # Broadcast A to match U's batch dims if needed
    n_ops = U.shape[-3]

    if op_mask is None:
        # All ops real
        # UAUt: sum_g U_g A U_g^T
        UAUt = torch.einsum("...oij,...jk,...olk->...oil", U, A, U)
        return UAUt.mean(dim=-3)

    # Masked path
    if op_mask.shape[-1] != n_ops:
        raise ValueError(
            f"op_mask last dim {op_mask.shape[-1]} != n_ops {n_ops}"
        )
    UAUt = torch.einsum("...oij,...jk,...olk->...oil", U, A, U)  # (..., n_ops, n, n)
    mask_f = op_mask.to(dtype=UAUt.dtype)  # (..., n_ops)
    # Sum masked contributions
    masked = UAUt * mask_f[..., None, None]
    numer = masked.sum(dim=-3)  # (..., n, n)
    denom = mask_f.sum(dim=-1).clamp(min=1.0)  # (...,)
    return numer / denom[..., None, None]


# ---------------------------------------------------------------------------
# Offsite Hermiticity + reverse-edge map  (E3)
# ---------------------------------------------------------------------------

def enforce_offsite_hermiticity(
    H_forward: torch.Tensor,
    H_reverse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Enforce Hermiticity between forward and reverse edge blocks.

    Given predicted blocks H_ij(R) and H_ji(-R), the Hermitian pair satisfies:

        H_ji(-R) = H_ij(R)^dagger

    For real Hamiltonians this is transpose. We average the two directions:

        H_forward_sym = (H_forward + H_reverse^T) / 2
        H_reverse_sym = H_forward_sym^T
    """
    H_reverse_T = H_reverse.transpose(-1, -2)
    H_sym = 0.5 * (H_forward + H_reverse_T)
    H_rev_sym = H_sym.transpose(-1, -2)
    return H_sym, H_rev_sym


def find_reverse_edge_map(
    edge_index: torch.Tensor,
    edge_shift: torch.Tensor,
) -> torch.Tensor:
    """Match each edge (i → j, R) with its reverse (j → i, -R).

    Args:
        edge_index: (2, E) [target, source] node indices.
        edge_shift: (E, 3) integer lattice shift R for each edge.

    Returns:
        (E,) long tensor ``rev[k] = k'`` such that edge k' is the reverse of
        edge k. If no partner exists, ``rev[k] = k`` (self-pair, i.e. the
        edge is treated as its own conjugate; used for on-site self-loops).
    """
    E = edge_index.shape[1]
    if E == 0:
        return torch.empty(0, dtype=torch.long, device=edge_index.device)

    target = edge_index[0]
    source = edge_index[1]
    shift = edge_shift.to(torch.long)

    # Build a key per edge: (target, source, R). We look up (source, target, -R).
    # Use a python dict of int tuples so ordering is well-defined.
    key_to_idx: Dict[Tuple[int, int, int, int, int], int] = {}
    for k in range(E):
        key = (
            int(target[k].item()),
            int(source[k].item()),
            int(shift[k, 0].item()),
            int(shift[k, 1].item()),
            int(shift[k, 2].item()),
        )
        key_to_idx[key] = k

    rev = torch.arange(E, dtype=torch.long, device=edge_index.device)
    for k in range(E):
        rev_key = (
            int(source[k].item()),
            int(target[k].item()),
            -int(shift[k, 0].item()),
            -int(shift[k, 1].item()),
            -int(shift[k, 2].item()),
        )
        j = key_to_idx.get(rev_key)
        if j is not None:
            rev[k] = j
    return rev


def hermitize_with_reverse_map(
    H_edges: torch.Tensor,
    reverse_map: torch.Tensor,
) -> torch.Tensor:
    """Symmetrize a per-edge block tensor using a reverse-edge map.

    For each edge k with partner k' (given by ``reverse_map``), replace
    ``H[k]`` with ``(H[k] + H[k'].T) / 2``. If k is its own partner
    (self-loop), the operation reduces to ``(H[k] + H[k].T) / 2`` — i.e. it
    enforces symmetry of the block.

    Args:
        H_edges: (E, n, m) per-edge block tensor.
        reverse_map: (E,) long, ``reverse_map[k]`` is the reverse-edge index.

    Returns:
        (E, n, m) Hermitized block tensor.
    """
    if H_edges.shape[0] == 0:
        return H_edges
    H_rev = H_edges[reverse_map]
    return 0.5 * (H_edges + H_rev.transpose(-1, -2))


# ---------------------------------------------------------------------------
# Bloch Hamiltonian construction  (E4)
# ---------------------------------------------------------------------------

def build_bloch_hamiltonian(
    real_space_blocks: Dict[Tuple[int, int, Tuple[int, int, int]], torch.Tensor],
    k_points: torch.Tensor,
    n_orbits: int,
    orbital_dims: torch.Tensor,
    lattice: torch.Tensor,
    k_coord_type: str = "cartesian",
    include_2pi: bool = False,
) -> torch.Tensor:
    """Construct Bloch Hamiltonian H(k) from real-space blocks.

        H(k) = sum_R  H(R) exp(i * phase(k, R))

    where the phase depends on the ``k_coord_type`` convention:

    * ``k_coord_type="cartesian"``, ``include_2pi=False``:
      ``phase = k_cart · R_cart``; ``k_cart`` in ``1/Angstrom``.
    * ``k_coord_type="cartesian"``, ``include_2pi=True``:
      ``phase = 2π (k_cart · R_cart)``; same as above but scaled by 2π.
    * ``k_coord_type="fractional"``, ``include_2pi=True``:
      ``phase = 2π (k_frac · R_frac)``; the canonical solid-state convention.
    * ``k_coord_type="fractional"``, ``include_2pi=False``:
      ``phase = k_frac · R_frac`` (no 2π factor).

    The two conventions are equivalent iff you're careful to use matching
    ``2π`` bookkeeping; this function does not guess.

    Args:
        real_space_blocks: dict {(i, j, (Rx, Ry, Rz)): H_ij(R)}.
        k_points: (n_k, 3) in either Cartesian (1/Å) or fractional
            reciprocal-lattice coords. See ``k_coord_type``.
        n_orbits: total number of orbit nodes (sites).
        orbital_dims: (n_orbits,) long, number of orbitals per site.
        lattice: (3, 3) lattice matrix (rows are lattice vectors).
        k_coord_type: "cartesian" | "fractional".
        include_2pi: whether to multiply the phase by 2π.

    Returns:
        (n_k, N, N) complex Hamiltonian, N = sum(orbital_dims).
    """
    if k_coord_type not in ("cartesian", "fractional"):
        raise ValueError(
            f"k_coord_type must be 'cartesian' or 'fractional'; got {k_coord_type!r}"
        )
    device = k_points.device
    dtype_complex = torch.complex64 if k_points.dtype == torch.float32 else torch.complex128
    n_k = k_points.shape[0]

    # Offsets per site
    offsets = torch.zeros(n_orbits + 1, dtype=torch.long, device=device)
    offsets[1:] = orbital_dims.cumsum(dim=0)
    N = int(offsets[-1].item())

    H_k = torch.zeros(n_k, N, N, dtype=dtype_complex, device=device)

    lattice_dev = lattice.to(device=device, dtype=k_points.dtype)
    two_pi = 2.0 * torch.pi if include_2pi else 1.0

    for (i, j, R_frac), block in real_space_blocks.items():
        R_frac_t = torch.tensor(R_frac, dtype=k_points.dtype, device=device)

        if k_coord_type == "cartesian":
            # phase = 2π? * k_cart · R_cart, with R_cart = R_frac @ lattice
            R_cart = R_frac_t @ lattice_dev  # (3,)
            phase = two_pi * (k_points @ R_cart)  # (n_k,)
        else:  # fractional
            # phase = 2π? * k_frac · R_frac
            phase = two_pi * (k_points @ R_frac_t)  # (n_k,)

        cexp = torch.exp(1j * phase)  # (n_k,)

        i_start = int(offsets[i].item())
        i_end = int(offsets[i + 1].item())
        j_start = int(offsets[j].item())
        j_end = int(offsets[j + 1].item())

        block_c = block.to(dtype=dtype_complex, device=device)
        H_k[:, i_start:i_end, j_start:j_end] += cexp[:, None, None] * block_c

    return H_k


def hermitize_bloch(H_k: torch.Tensor) -> torch.Tensor:
    """Enforce Hermiticity on a Bloch Hamiltonian: H(k) = (H(k) + H(k)^†) / 2."""
    return 0.5 * (H_k + H_k.conj().transpose(-1, -2))


# ---------------------------------------------------------------------------
# WyckoffHamiltonianHead — main module
# ---------------------------------------------------------------------------

class WyckoffHamiltonianHead(nn.Module):
    """Predict onsite and offsite Hamiltonian blocks from Wyckoff GNN features.

    Onsite blocks: MLP on node features, then symmetrize with site-symmetry.
    Offsite blocks: MLP on concatenated (node_i, node_j, edge_features).

    Args:
        node_feature_dim: input dim of node scalar features (0e channels).
        edge_feature_dim: input dim of edge invariant features.
        basis: one of "s" / "p" / "d" / "sp" / "spd". All sites use same basis.
        hidden_dim: hidden width of block-prediction MLPs.

    See module docstring for the Phase-5 scope statement — in particular the
    lack of enforced space-group covariance for offsite blocks and the
    requirement that ``edge_features`` be locally invariant.
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        basis: str = "s",
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.basis = basis
        self.n_orb = basis_dim(basis)
        self.n_orb_sq = self.n_orb * self.n_orb

        self.onsite_head = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.n_orb_sq),
        )

        offsite_input_dim = 2 * node_feature_dim + edge_feature_dim
        self.offsite_head = nn.Sequential(
            nn.Linear(offsite_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.n_orb_sq),
        )

    def predict_onsite(
        self,
        h_scalars: torch.Tensor,
        site_rep_matrices: Optional[torch.Tensor] = None,
        site_rep_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict onsite blocks with optional site-symmetry symmetrization.

        Args:
            h_scalars: (K, node_feature_dim) invariant scalar features.
            site_rep_matrices: optional (K, n_ops_max, n_orb, n_orb) — per-site
                stabilizer orbital reps. Padded slots must be marked in
                ``site_rep_mask``.
            site_rep_mask: optional (K, n_ops_max) boolean mask; True where
                the corresponding rep matrix is real, False where it is
                padding. Ignored if ``site_rep_matrices`` is None.

        Returns:
            (K, n_orb, n_orb) onsite blocks.
        """
        raw = self.onsite_head(h_scalars)  # (K, n_orb^2)
        blocks = raw.reshape(-1, self.n_orb, self.n_orb)

        blocks = 0.5 * (blocks + blocks.transpose(-1, -2))

        if site_rep_matrices is not None:
            if site_rep_mask is not None:
                blocks = symmetrize_onsite_block(
                    blocks, site_rep_matrices, op_mask=site_rep_mask
                )
            else:
                K = h_scalars.shape[0]
                symmetrized = torch.zeros_like(blocks)
                for k in range(K):
                    U_k = site_rep_matrices[k]
                    if U_k.shape[0] > 0:
                        symmetrized[k] = symmetrize_onsite_block(blocks[k], U_k)
                    else:
                        symmetrized[k] = blocks[k]
                blocks = symmetrized

        return blocks

    def predict_offsite(
        self,
        h_scalars: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        reverse_edge_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict offsite blocks for each edge.

        Args:
            h_scalars: (K, node_feature_dim).
            edge_features: (E, edge_feature_dim). Must be locally invariant
                edge features (see module docstring).
            edge_index: (2, E) [target, source] orbit indices.
            reverse_edge_map: optional (E,) long tensor. If provided, output
                blocks are Hermitized against their reverse-edge partners
                using :func:`hermitize_with_reverse_map`. Recommended.

        Returns:
            (E, n_orb, n_orb) offsite blocks. Hermitized w.r.t. the reverse-
            edge map when provided; otherwise raw predictions.
        """
        target = edge_index[0]
        source = edge_index[1]
        h_target = h_scalars[target]
        h_source = h_scalars[source]
        cat = torch.cat([h_target, h_source, edge_features], dim=-1)
        raw = self.offsite_head(cat)
        blocks = raw.reshape(-1, self.n_orb, self.n_orb)

        if reverse_edge_map is not None:
            blocks = hermitize_with_reverse_map(blocks, reverse_edge_map)

        return blocks

    def forward(
        self,
        h_scalars: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        site_rep_matrices: Optional[torch.Tensor] = None,
        site_rep_mask: Optional[torch.Tensor] = None,
        reverse_edge_map: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict both onsite and offsite blocks.

        Returns:
            dict with keys "onsite" (K, n_orb, n_orb) and
            "offsite" (E, n_orb, n_orb).
        """
        onsite = self.predict_onsite(
            h_scalars,
            site_rep_matrices=site_rep_matrices,
            site_rep_mask=site_rep_mask,
        )
        offsite = self.predict_offsite(
            h_scalars, edge_features, edge_index,
            reverse_edge_map=reverse_edge_map,
        )
        return {"onsite": onsite, "offsite": offsite}
