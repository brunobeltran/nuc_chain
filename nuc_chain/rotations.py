"""Helper functions to deal with various representations of rotations."""
import numpy as np
from . import wignerD as wd

TOL = 10e-14

def zyz_from_matrix(R):
    """Convert from a rotation matrix to (extrinsic zyz) Euler angles.

    Parameters
    ----------
    R : (3, 3) array_like
        Rotation matrix to convert

    Returns
    -------
    alpha : float
        The final rotation about z''
    beta : float
        The second rotation about y'
    gamma : float
        the initial rotation about z

    Notes
    -----
    See pg. 13 of Bruno's (NCG) nucleosome geometry notes for a detailed
    derivation of this formula. Easy to test correctness by simply
    reversing the process R == Rz(alpha)@Ry(beta)@Rz(gamma).
    """
    beta = np.arccos(R[2,2])
    if np.abs(1 - np.abs(R[2,2])) < TOL:
        beta = 0
        gamma = 0
        Rz_alpha = R
    else:
        gamma = np.arctan2(-R[2,1]/np.sin(beta), R[2,0]/np.sin(beta))
        Rz_alpha = R @ np.linalg.inv(Rz(gamma)) @ np.linalg.inv(Ry(beta))
    alpha = np.arccos(Rz_alpha[0,0])
    # # couldn't get this part of the formula to work for some reason
    # alpha = np.arctan2(R[0,2]/np.sin(beta), -R[1,2]/np.sin(beta))
    return alpha, beta, gamma

def phi_theta_alpha_from_R(R):
    r"""Return the (intrinsic?) z, y', z'' euler angles corresponding to the
    rotation R.  These have a natural interpretation for various rotation
    matrices representing DNA kinked by nucleosomes.

    For example, if called on the output of dp_omega_exit, the angles
    returned will correspond to the magnitudes of the twist being
    performed by the linker+upstream unwrap (phi), the pure bend from
    nucleosome (theta), and the twist from nucleosome+downstream unwrap
    (alpha).

    Notes
    -----
    The angles returned by this function (:math:`\phi_i, \theta_i,
    \alpha_i`) are related to the extrinsic zyz euler angles :math:`\alpha_e,
    \beta_e, \gamma_e` by the relationships :math:`alpha_e - phi_i = \pi`,
    :math:`\theta_i = -\beta_e`, :math:`alpha_i - \gamma_e = \pi`.
    """
    # z_pp := z'', x_p = x', etc, where the coordinate axes before rotation
    # are called x,y,z, the resultant axes after a rotation about z are
    # called x',y',z' (z'==z), the axes after a further rotation about y'
    # are called x'', y'', z'', (y''==y'), and teh final axes after a final
    # rotation about z'' are called x''', y''', z''' (z'''==z''). The third
    # column of the R matrix is z'' in the coordinates of x,y,z
    z_pp = R[:, 2]
    # the x-y projection of z'' contains phi, since we spun around y' second
    phi = np.arctan2(z_pp[1], z_pp[0])
    # we spun about y' by theta to get from z to z', so the z projection of
    # z' and the x-y projection of z' are kind of like the x and y coordinates
    # in the sense that arctan2 requires, but left-handed
    theta = -np.arctan2(np.linalg.norm(z_pp[0:2]), z_pp[2])
    # finally, to get alpha, we simply see how much we would have to rotate the
    # coordinate system we've created so far
    xyz_pp = Rz(phi)@Ry(theta)
    x_final = R[:,0]
    x_final_in_pp_coords = x_final @ xyz_pp
    alpha = np.arctan2(x_final_in_pp_coords[1], x_final_in_pp_coords[0])
    return phi, theta, alpha

def matrix_from_zyz(alpha, beta, gamma):
    """Inverse of zyz_from_matrix.

    matrix_from_zyz(*zyz_from_matrix(R)) == R, up to floating error."""
    return Rz(alpha) @ Ry(beta) @ Rz(gamma)

def wignerD_from_R(R, nlam):
    alpha, beta, gamma = zyz_from_matrix(R)
    mymat = wd.wigner_d_vals()
    Dlist = mymat.get_mtrx(alpha, beta, gamma, nlam)
    return Dlist

def wignerD_from_R1R2(R1, R2, nlam):
    Dlist1 = wignerD_from_R(R1, nlam)
    Dlist2 = wignerD_from_R(R2, nlam)
    Dlist3 = []
    #for each value of l, simply do a matrix multiplication of Dlist1[l] and Dlist2[l]
    #this is equivalent to contracting one index
    for l in range(0,nlam):
        Dlist3.append(Dlist1[l]@Dlist2[l])
    return Dlist3


def axis_angle_from_matrix(R):
    w, v = np.linalg.eig(R)
    principle_i = np.argmin(np.abs(w - 1))
    axis = np.real(v[:,principle_i])
    angle = np.arccos((np.trace(R) - 1)/2)
    return axis, angle

#TODO bug in off-diagonal case, fix
# def matrix_from_axis_angle(axis, angle):
#     a = angle
#     u = axis
#     eijk = np.zeros((3, 3, 3))
#     eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
#     eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
#     R = np.zeros((3,3))
#     for j in range(3):
#         for k in range(3):
#             if j == k:
#                 R[j,k] = np.power(np.cos(a/2), 2) \
#                        + np.power(np.sin(a/2), 2) \
#                        * (2*np.power(u[j], 2) - 1)
#             else:
#                 R[j,k] = np.sum(2*u[j]*u[k]*np.power(np.sin(a/2), 2) \
#                        - eijk [j, k, :]*u[:]*np.sin(a))
#     return R

def Rx(theta):
    r"""Rotation matrix about the x axis with angle theta.

    Notes
    -----
    ..math::

        \frac{1}{\sqrt{3}}
        \begin{bmatrix}
            1 &             0 &              0\\
            0 & np.cos(theta) & -np.sin(theta)\\
            0 & np.sin(theta) &  np.cos(theta)
        \end{bmatrix}
    """
    return np.array([[1,             0,              0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta),  np.cos(theta)]])



def Ry(theta):
    r"""Rotation matrix about the z axis with angle theta.

    Notes
    -----
    ..math::

        \frac{1}{\sqrt{3}}
        \begin{bmatrix}
            np.cos(theta) & 0 & -np.sin(theta) \\
                        0 & 1 &              0 \\
            np.sin(theta) & 0 &  np.cos(theta)
        \end{bmatrix}
    """
    return np.array([[np.cos(theta), 0, -np.sin(theta)],
                     [            0, 1,              0],
                     [np.sin(theta), 0,  np.cos(theta)]])

def Rz(theta):
    r"""Rotation matrix about the z axis with angle theta.

    Notes
    -----
    ..math::

        \frac{1}{\sqrt{3}}
        \begin{bmatrix}
            np.cos(theta) & -np.sin(theta) & 0 \\
            np.sin(theta) &  np.cos(theta) & 0 \\
                        0 &             0  & 1
        \end{bmatrix}
    """
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta),  np.cos(theta), 0],
                    [            0,             0,  1]])


def R_dP_to_screw_coordinates(R, dp):
    """Takes a rotation matrix and an offset and gets the parameters
    of the effective superhelix generated by iterating those two offsets by
    putting the rotation+translation operation into "screw" coordinates.

    Returns
    -------
    rise : float
        amount of distance along screw axis traveled with each iteration in nm
    angle : float
        amount of rotation about screw axis for each iteration in rad
    radius : float
        radius of superhelical screw in nm
    """
    u, theta = axis_angle_from_matrix(R)
    u = u/np.linalg.norm(u) # just in case
    dp_aligned = np.dot(dp, u)
    rise = np.linalg.norm(dp_aligned)
    # no rotation, so just translation along the dp axis
    if np.isclose(theta, 0) or np.isclose(theta, 2*np.pi):
        rise = np.linalg.norm(dp)
        radius = 0
        theta = 0
    # screw axis goes through midpoint of translation vector dp
    elif np.isclose(theta, np.pi):
        radius = np.linalg.norm((dp - dp_aligned)/2)
    # law of sines, see pg 3 of Bruno's (NCG) nucleosome chain geometry notes
    else:
        radius = np.sin((np.pi - theta)/2)/np.sin(theta) \
                * np.linalg.norm(dp - dp_aligned)
    return rise, theta, radius

