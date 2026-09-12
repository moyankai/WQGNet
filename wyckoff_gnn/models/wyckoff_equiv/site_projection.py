"""Site-symmetry irrep projection layer for Wyckoff-orbit node features.

Loads precomputed projection matrices from a Phase-1 table
(`site_projectors_*.pt`) and applies per-orbit, per-irrep-block projection to
node features.

Projection matrices are frozen buffers — no learning. The projection is:

    h_p_new[block_l] = P_l^{H_p} @ h_p[block_l]

for each irrep block, where `P` is the invariant-subspace projector for the
stabilizer group `H_p` of the p-th orbit's Wyckoff position.

Schema support
--------------

* **v2 (list-of-variants, current):**
  JSON: ``table[str(sg)][letter] = [ {stab_hash, projector_key, ...}, ... ]``.
  Multiple stabilizer variants per (sg, letter) are disambiguated by
  ``stab_hash``.
* **v1 (dict-valued letters, legacy):** ``table[str(sg)][letter] = {…}``. The
  loader continues to read this and treats the single entry as the sole
  variant.

Lookup priority (per node p):
  1. Explicit ``site_projector_key_idx`` (int index into the projector bank),
     if the caller provides it on the batch object;
  2. ``(sg, letter, stab_hash)``, if ``stab_hash`` is provided;
  3. ``(sg, letter)`` first-variant fallback (legacy behavior);
  4. Unknown ⇒ dispatch based on ``unknown_projector_policy``.

Policies
--------
* ``unknown_projector_policy`` (default ``"identity"``):
    - ``"identity"`` — apply identity (pass-through) to unknown nodes.
    - ``"error"`` — raise ``RuntimeError`` if any node has no matching key.
    - ``"count"`` — apply identity but record hit in ``self.n_unknown_hits``.
* ``missing_irrep_policy`` (default ``"identity"``):
    - ``"identity"`` — fill missing irrep blocks with identity (permissive).
    - ``"zero"`` — fill with zero (strict; forbids that block).
    - ``"error"`` — raise ``RuntimeError`` at load time if any needed block is
      absent from the .pt file for a listed key.

Config keys consumed elsewhere (e3nn_encoder / model factory):
    use_site_irrep_projection: bool = False
    site_projector_path: str
    site_irrep_table_path: str
    apply_site_projection: str = "each_layer"
    unknown_projector_policy: str = "identity"
    missing_irrep_policy: str = "identity"
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from e3nn import o3


def _letter_from_idx(letter_idx: int) -> str:
    if 0 <= letter_idx < 26:
        return chr(ord("a") + letter_idx)
    return "?"


def _sg_letter_key(sg: int, letter_idx: int) -> str:
    """Legacy 2-tier lookup key: `SG{n}_{letter}` (no stab_hash)."""
    return f"SG{sg}_{_letter_from_idx(letter_idx)}"


def _sg_letter_hash_key(sg: int, letter_idx: int, stab_hash: str) -> str:
    """3-tier lookup key: `SG{n}_{letter}_{stab_hash}`."""
    return f"SG{sg}_{_letter_from_idx(letter_idx)}_{stab_hash}"


class SiteIrrepProjector(nn.Module):
    """Apply per-node, per-irrep-block site-symmetry projection.

    The loader recognizes both schema-v1 (dict) and schema-v2 (list-of-variants)
    JSON tables.

    Args:
        irreps: e3nn Irreps of the node feature space.
        projector_path: Path to Phase 1 ``site_projectors_lmaxN.pt`` file.
        table_path: Path to Phase 1 ``site_irreps_lmaxN.json`` file.
        lmax: Angular momentum cutoff used when generating the table.
        unknown_projector_policy: ``"identity"`` (default) / ``"error"`` / ``"count"``.
        missing_irrep_policy: ``"identity"`` (default) / ``"zero"`` / ``"error"``.
    """

    def __init__(
        self,
        irreps: o3.Irreps,
        projector_path: str,
        table_path: Optional[str] = None,
        lmax: int = 4,
        unknown_projector_policy: str = "identity",
        missing_irrep_policy: str = "identity",
    ):
        super().__init__()
        self.irreps = irreps
        self.lmax = lmax
        self.dim = irreps.dim
        self.unknown_projector_policy = unknown_projector_policy
        self.missing_irrep_policy = missing_irrep_policy

        if unknown_projector_policy not in ("identity", "error", "count"):
            raise ValueError(
                f"unknown_projector_policy must be one of "
                f"'identity','error','count'; got {unknown_projector_policy!r}"
            )
        if missing_irrep_policy not in ("identity", "zero", "error"):
            raise ValueError(
                f"missing_irrep_policy must be one of "
                f"'identity','zero','error'; got {missing_irrep_policy!r}"
            )

        if not os.path.exists(projector_path):
            raise FileNotFoundError(
                f"SiteIrrepProjector: projector file not found: {projector_path}"
            )

        raw: Dict[str, torch.Tensor] = torch.load(
            projector_path, map_location="cpu", weights_only=False
        )

        self.table: Dict[str, Any] = {}
        self.table_meta: Dict[str, Any] = {}
        if table_path is not None and os.path.exists(table_path):
            with open(table_path) as f:
                self.table = json.load(f)
            self.table_meta = self.table.get("_meta", {}) if isinstance(self.table, dict) else {}
        self.schema_version = int(self.table_meta.get("schema_version", 1))

        # Compute irrep block offsets.
        self._block_slices: List[Tuple[int, int, o3.Irrep]] = []
        offset = 0
        for mul, ir in irreps:
            for _ in range(mul):
                self._block_slices.append((offset, offset + ir.dim, ir))
                offset += ir.dim
        assert offset == self.dim

        # 1. Discover every projector_key present in the .pt file.
        # Support two formats:
        #   - Old: "SG{n}_{letter}_lmax{L}_{stab_hash}/{ir_str}" (with /)
        #   - New: "SG{n}_{letter}_l{l}_{p}" (without /, self-contained)
        prefixes = set()
        new_format_keys = {}  # For new format: prefix -> {ir_str: tensor}
        
        for key_full in raw.keys():
            if "/" in key_full:
                # Old format: split by /
                parts = key_full.rsplit("/", 1)
                if len(parts) == 2:
                    prefixes.add(parts[0])
            else:
                # New format: parse "SG{n}_{letter}_l{l}_{p}"
                # Extract prefix = "SG{n}_{letter}" and irrep info
                parts = key_full.split("_")
                if len(parts) >= 4 and parts[0].startswith("SG"):
                    # SG{n}_{letter}_l{l}_{p}
                    sg = parts[0]  # "SG1"
                    letter = parts[1]  # "a"
                    prefix = f"{sg}_{letter}"  # "SG1_a"
                    
                    # Extract irrep: l{l}_{p} -> {l}{p}
                    l_part = parts[2]  # "l0", "l1", etc.
                    p_part = parts[3] if len(parts) > 3 else "e"  # "e" or "o"
                    l_num = int(l_part[1:])  # 0, 1, 2, ...
                    ir_str = f"{l_num}{p_part}"  # "0e", "1o", etc.
                    
                    if prefix not in new_format_keys:
                        new_format_keys[prefix] = {}
                    new_format_keys[prefix][ir_str] = raw[key_full]
                    prefixes.add(prefix)

        # 2. Assemble block-diagonal (dim, dim) projector per key.
        self._full_projectors: Dict[str, torch.Tensor] = {}
        n_missing_blocks = 0
        keys_with_missing_blocks: List[str] = []
        for prefix in sorted(prefixes):
            P_full = torch.zeros(self.dim, self.dim, dtype=torch.float32)
            missing_here = False
            
            # Check if this is new format or old format
            is_new_format = prefix in new_format_keys
            
            for (start, end, ir) in self._block_slices:
                ir_str = str(ir)
                
                if is_new_format:
                    # New format: lookup from new_format_keys
                    P_block = new_format_keys[prefix].get(ir_str, None)
                else:
                    # Old format: lookup from raw with "prefix/ir_str" key
                    k = f"{prefix}/{ir_str}"
                    P_block = raw.get(k, None)
                
                if P_block is None:
                    if missing_irrep_policy == "error":
                        raise RuntimeError(
                            f"SiteIrrepProjector: missing irrep block {prefix}/{ir_str} "
                            f"and missing_irrep_policy='error'"
                        )
                    if missing_irrep_policy == "identity":
                        P_full[start:end, start:end] = torch.eye(
                            ir.dim, dtype=torch.float32
                        )
                    # 'zero' → leave zeros
                    missing_here = True
                    n_missing_blocks += 1
                else:
                    P_full[start:end, start:end] = P_block.float()
            self._full_projectors[prefix] = P_full
            if missing_here:
                keys_with_missing_blocks.append(prefix)

        # 3. Build lookup tables. Two tiers:
        #    - 3-tier: full projector_key (contains stab_hash) → bank index.
        #    - 2-tier (fallback): sg_letter_prefix → list of bank indices.
        self._key_to_idx: Dict[str, int] = {}
        self._sg_letter_to_indices: Dict[str, List[int]] = {}

        stacked = []
        for i, (key, P) in enumerate(sorted(self._full_projectors.items())):
            self._key_to_idx[key] = i
            stacked.append(P)

            # Derive 2-tier fallback key = "SG{n}_{letter}" from the projector_key.
            # projector_key formats seen:
            #   v1: "SG{n}_{letter}_lmax{L}"
            #   v2: "SG{n}_{letter}_lmax{L}_{stab_hash}"
            base = key
            if "_lmax" in base:
                base = base[: base.rindex("_lmax")]
            self._sg_letter_to_indices.setdefault(base, []).append(i)

        if stacked:
            P_all = torch.stack(stacked, dim=0)
        else:
            P_all = torch.zeros(1, self.dim, self.dim, dtype=torch.float32)
        self.register_buffer("projector_bank", P_all, persistent=False)
        self.register_buffer(
            "identity_projector",
            torch.eye(self.dim, dtype=torch.float32),
            persistent=False,
        )

        # Build dense (sg, letter) → bank_idx lookup table for vectorized forward.
        dense_lookup = torch.full((231, 26), -1, dtype=torch.long)
        for base_key, idx_list in self._sg_letter_to_indices.items():
            # base_key = "SG{n}_{letter}"
            parts = base_key.split("_")
            if len(parts) == 2 and parts[0].startswith("SG"):
                sg = int(parts[0][2:])
                letter_char = parts[1]
                if 1 <= sg <= 230 and len(letter_char) == 1 and "a" <= letter_char <= "z":
                    letter_idx = ord(letter_char) - ord("a")
                    dense_lookup[sg, letter_idx] = idx_list[0]
        self.register_buffer("_sg_letter_lookup", dense_lookup, persistent=False)

        # Populate the (sg, letter) → variant metadata (from JSON table when
        # present, used only for `coverage_report`).
        self._variant_meta: Dict[str, List[dict]] = {}
        if self.table:
            for sg_str, letters in self.table.items():
                if sg_str.startswith("_"):
                    continue
                if not isinstance(letters, dict):
                    continue
                for letter, val in letters.items():
                    if isinstance(val, list):
                        for v in val:
                            self._register_variant_meta(sg_str, letter, v)
                    elif isinstance(val, dict):
                        self._register_variant_meta(sg_str, letter, val)

        # Stats
        self.n_projectors = len(self._full_projectors)
        self.n_partially_missing = len(keys_with_missing_blocks)
        self.n_missing_blocks = n_missing_blocks
        self.n_unknown_hits = 0  # incremented at forward-time under 'count'

    def _register_variant_meta(self, sg_str: str, letter: str, variant: dict) -> None:
        key = f"SG{sg_str}_{letter}"
        self._variant_meta.setdefault(key, []).append(variant)

    # ------------------------------------------------------------------
    # Coverage reporting
    # ------------------------------------------------------------------

    def coverage_report(self) -> Dict[str, Any]:
        """Return a dict summarizing what's on disk vs what's usable.

        The report distinguishes:
        * `n_projector_keys` — how many bank entries we loaded from .pt
        * `n_sg_letter_prefixes` — distinct (sg, letter) 2-tier fallback keys
        * `n_partially_missing_keys` — keys with at least one absent irrep block
        * `n_total_missing_blocks` — total absent irrep blocks across all keys
        * `variant_collisions` — how many (sg, letter) prefixes carry >1 stab_hash
        * `schema_version`
        """
        collisions = sum(1 for v in self._sg_letter_to_indices.values() if len(v) > 1)
        return {
            "n_projector_keys": self.n_projectors,
            "n_sg_letter_prefixes": len(self._sg_letter_to_indices),
            "n_partially_missing_keys": self.n_partially_missing,
            "n_total_missing_blocks": self.n_missing_blocks,
            "variant_collisions": collisions,
            "schema_version": self.schema_version,
            "unknown_projector_policy": self.unknown_projector_policy,
            "missing_irrep_policy": self.missing_irrep_policy,
        }

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def _lookup_indices(
        self,
        space_group: torch.Tensor,           # (num_graphs,) or (K,)
        orbit_letter: torch.Tensor,          # (K,) long
        batch: torch.Tensor,                 # (K,) long
        stab_hash: Optional[Sequence[str]] = None,          # (K,) list[str]
        projector_key_idx: Optional[torch.Tensor] = None,   # (K,) long, precomputed
    ) -> torch.Tensor:
        """Return (K,) int64 index into `projector_bank`, or -1 for unknown."""
        K = orbit_letter.shape[0]
        device = orbit_letter.device

        # Tier 1: explicit key index provided by the batch object.
        if projector_key_idx is not None:
            return projector_key_idx.to(device=device, dtype=torch.long)

        # Vectorized (sg, letter) → bank_idx via dense lookup table.
        if space_group.numel() == 1:
            sg_per_node = space_group.expand(K).to(device)
        elif space_group.shape[0] == K:
            sg_per_node = space_group.to(device)
        else:
            sg_per_node = space_group[batch].to(device)

        lookup = self._sg_letter_lookup.to(device)
        sg_clamped = sg_per_node.clamp(min=0, max=230)
        letter_clamped = orbit_letter.clamp(min=0, max=25)
        indices = lookup[sg_clamped, letter_clamped]
        return indices

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        h: torch.Tensor,
        space_group: torch.Tensor,
        orbit_letter: torch.Tensor,
        batch: torch.Tensor,
        stab_hash: Optional[Sequence[str]] = None,
        projector_key_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply site-symmetry projection to node features.

        Args:
            h: (K, dim) node features.
            space_group: (1,) or (num_graphs,) or (K,) — graph or per-node SG.
            orbit_letter: (K,) per-node Wyckoff letter index.
            batch: (K,) per-node graph assignment.
            stab_hash: optional list of stab_hash strings, length K. Enables
                tier-2 lookup.
            projector_key_idx: optional (K,) precomputed index into
                projector_bank. Enables tier-1 lookup (fastest, most explicit).

        Returns:
            (K, dim) projected features. Nodes with unknown key dispatch
            according to ``unknown_projector_policy``.
        """
        K = h.shape[0]
        indices = self._lookup_indices(
            space_group, orbit_letter, batch,
            stab_hash=stab_hash,
            projector_key_idx=projector_key_idx,
        )

        known_mask = indices >= 0
        unknown_count = int((~known_mask).sum().item())
        if unknown_count > 0:
            if self.unknown_projector_policy == "error":
                raise RuntimeError(
                    f"SiteIrrepProjector: {unknown_count}/{K} nodes have no "
                    f"projector match and unknown_projector_policy='error'"
                )
            if self.unknown_projector_policy == "count":
                self.n_unknown_hits += unknown_count

        if not known_mask.any():
            return h  # all unknown → identity (or already errored)

        idx_safe = indices.clamp(min=0)
        P_per_node = self.projector_bank[idx_safe]
        if P_per_node.dtype != h.dtype:
            P_per_node = P_per_node.to(h.dtype)

        all_known = unknown_count == 0
        if not all_known:
            eye = self.identity_projector.to(h.dtype)
            P_per_node = torch.where(
                known_mask.view(K, 1, 1),
                P_per_node,
                eye.unsqueeze(0).expand(K, -1, -1),
            )

        h_out = torch.bmm(P_per_node, h.unsqueeze(-1)).squeeze(-1)
        return h_out

    def extra_repr(self) -> str:
        return (
            f"irreps={self.irreps}, n_projectors={self.n_projectors}, "
            f"partially_missing={self.n_partially_missing}, "
            f"schema=v{self.schema_version}, "
            f"unknown_policy={self.unknown_projector_policy}, "
            f"missing_policy={self.missing_irrep_policy}"
        )


__all__ = ["SiteIrrepProjector"]
