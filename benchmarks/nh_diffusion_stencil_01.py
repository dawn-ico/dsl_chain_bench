from dusk.script import *

smag_offset = Global("smag_offset")

lb, nudging, interior, halo, end = HorizontalDomains(0, 1000, 2000, 3000, 4000)

@stencil
def mo_nh_diffusion_stencil_01(
    diff_multfac_smag: Field[K],
    tangent_orientation: Field[Edge],
    inv_primal_edge_length: Field[Edge],
    inv_vert_vert_length: Field[Edge],
    u_vert: Field[Vertex, K],
    v_vert: Field[Vertex, K],
    primal_normal_vert_x: Field[Edge > Cell > Vertex],
    primal_normal_vert_y: Field[Edge > Cell > Vertex],
    dual_normal_vert_x: Field[Edge > Cell > Vertex],
    dual_normal_vert_y: Field[Edge > Cell > Vertex],
    vn: Field[Edge, K],
    smag_limit: Field[K],
    kh_smag_e: Field[Edge, K],
    kh_smag_ec: Field[Edge, K],
    z_nabla2_e: Field[Edge, K]
) -> None:

    vn_vert: Field[Edge > Cell > Vertex, K]
    dvt_tang: Field[Edge, K]
    dvt_norm: Field[Edge, K]
    kh_smag_1: Field[Edge, K]
    kh_smag_2: Field[Edge, K]

    with domain.upward:

        # fill sparse dimension vn vert using the loop concept
        with sparse[Edge > Cell > Vertex]:
            vn_vert = u_vert * primal_normal_vert_x + v_vert * primal_normal_vert_y

        # dvt_tang for smagorinsky
        dvt_tang = sum_over(
            Edge > Cell > Vertex,
            (u_vert * dual_normal_vert_x) + (v_vert * dual_normal_vert_y),
            weights=[-1.0, 1.0, 0.0, 0.0]
        )


        # dvt_norm for smagorinsky
        dvt_norm = sum_over(
            Edge > Cell > Vertex,
            u_vert * dual_normal_vert_x + v_vert * dual_normal_vert_y,
            weights=[0.0, 0.0, -1.0, 1.0],
        )

        # compute smagorinsky
        kh_smag_1 = sum_over(
            Edge > Cell > Vertex,
            vn_vert,
            weights=[-1.0, 1.0, 0.0, 0.0]
        )

        dvt_tang = dvt_tang * tangent_orientation

        kh_smag_1 = (kh_smag_1 * tangent_orientation * inv_primal_edge_length) + (
            dvt_norm * inv_vert_vert_length
        )

        kh_smag_1 = kh_smag_1 * kh_smag_1

        kh_smag_2 = sum_over(
            Edge > Cell > Vertex,
            vn_vert,
            weights=[0.0, 0.0, -1.0, 1.0]
        )

        kh_smag_2 = (kh_smag_2 * inv_vert_vert_length) - (
            dvt_tang * inv_primal_edge_length
        )

        kh_smag_2 = kh_smag_2 * kh_smag_2

        kh_smag_e = diff_multfac_smag * sqrt(kh_smag_2 + kh_smag_1)

        # compute nabla2 using the diamond reduction

        z_nabla2_e = (sum_over(
            Edge > Cell > Vertex,
            vn_vert,
            weights=[1.0, 1.0, 0.0, 0.0],
        ) - 2.0 * vn) * (inv_primal_edge_length * inv_primal_edge_length)

        z_nabla2_e = z_nabla2_e + (sum_over(
            Edge > Cell > Vertex,
            vn_vert,
            weights=[0.0, 0.0, 1.0, 1.0],
        ) - 2.0 * vn) * (inv_vert_vert_length * inv_vert_vert_length)

        z_nabla2_e = 4.0 * z_nabla2_e

        kh_smag_ec = kh_smag_e
        kh_smag_e = max(0., kh_smag_e - smag_offset)
        kh_smag_e = min(kh_smag_e, smag_limit)
