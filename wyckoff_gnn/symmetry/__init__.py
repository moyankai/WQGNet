"""Site-symmetry constrained irrep tables, projections, and tensor conversions."""

from wyckoff_gnn.symmetry.site_irreps import (
    allowed_irrep_table_for_site,
    frac_op_to_cart_op,
    frac_ops_to_cart_ops,
    get_stabilizer_ops,
    irrep_matrix_e3nn,
    parse_irreps_list,
    site_projection_matrix,
    verify_projector,
)
from wyckoff_gnn.symmetry.tensor_conversions import (
    cartesian_to_irreps_1e,
    cartesian_to_irreps_1o,
    general_tensor_to_irreps_0e1e2e,
    get_irreps_for_tensor,
    irreps_0e1e2e_to_general_tensor,
    irreps_0e2e_to_symmetric_tensor,
    irreps_1e_to_cartesian,
    irreps_1o_to_cartesian,
    symmetric_tensor_to_irreps_0e2e,
)

__all__ = [
    "allowed_irrep_table_for_site",
    "frac_op_to_cart_op",
    "frac_ops_to_cart_ops",
    "get_stabilizer_ops",
    "irrep_matrix_e3nn",
    "parse_irreps_list",
    "site_projection_matrix",
    "verify_projector",
    "cartesian_to_irreps_1e",
    "cartesian_to_irreps_1o",
    "general_tensor_to_irreps_0e1e2e",
    "get_irreps_for_tensor",
    "irreps_0e1e2e_to_general_tensor",
    "irreps_0e2e_to_symmetric_tensor",
    "irreps_1e_to_cartesian",
    "irreps_1o_to_cartesian",
    "symmetric_tensor_to_irreps_0e2e",
]
