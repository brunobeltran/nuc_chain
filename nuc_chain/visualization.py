import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from mayavi import mlab
import numpy as np
import pandas as pd

from . import *
from .geometry import helix_params_best
from .linkers import convert


def plot_circle(phis, rs, values, ax=None, **kwargs):
    if ax is None:
        fix, ax = plt.subplots()
    x, y = rs*np.cos(phis), rs*np.sin(phis)
    return ax.scatter(x, y, c=values, **kwargs)

def plot_nucleosome(entry_rot, entry_pos, Lw=default_Lw,
                    helix_params=helix_params_best,
                    tube_radius=dna_params['r_dna'], mfig=None, **kwargs):
    if mfig is None:
        mfig = mlab.figure()
    nuc = H_oriented(i=None, entry_pos=entry_pos, entry_rot=entry_rot, Lw=Lw,
                     **helix_params)
    mlab.plot3d(nuc[0], nuc[1], nuc[2], tube_radius=tube_radius, figure=mfig,
                **kwargs)
    return nuc, mfig

def visualize_chain(entry_rots, entry_pos, links, w_ins=default_w_in,
        w_outs=default_w_out, lpb=dna_params['lpb'], r_dna=dna_params['r_dna'],
        helix_params=helix_params_best, mfig=None, palette="husl",
        unwraps=None, plot_entry=False, plot_exit=False, plot_nucleosomes=False,
        plot_spheres=True, nucleosome_color=None, **kwargs):
    """Visualize output of :py:func:`minimum_energy_no_sterics`.

    Parameters
    ----------
    entry_rots : (L+1,3,3)
        Orientation of the first bound base pair of each nucleosome.
    entry_pos : (L+1,3)
        Position of the first bound base pair of each nucleosome.
    links : (L,)
        Length of the linkers joining the nucleosomes.
    w_ins : float or (L+1,) array_like
        amount of DNA wrapped on entry side of central dyad base
    w_outs : float or (L+1,) array_like
        amount of DNA wrapped on exit side of central dyad base
    helix_params : (optional) Dict[str, float]
        The helix parameters to use. Defaults to geometry.helix_params_best
    mfig : (optional) matplotlib.Axes
        mlab figure to plot in.
    palette : (optional) str
        Seaborn palette to draw colors from to color each nucleosome

    Returns
    -------
    mfig : mlab.figure
        Figure containing the rendering.
    """

    if mfig is None:
        mfig = mlab.figure()
    b = helix_params['b']
    num_nucleosomes = len(entry_rots)
    links = np.atleast_1d(links)
    w_ins, w_outs = convert.resolve_wrapping_params(unwraps, w_ins, w_outs, num_nucleosomes)
    if len(links) == 1:
        links = np.tile(links, (num_nucleosomes-1,))
    assert(len(links) == num_nucleosomes - 1)
    assert(np.all(entry_rots.shape[:2] == entry_pos.shape))
    colors = sns.color_palette(palette, num_nucleosomes)
    if nucleosome_color is not None:
        colors = [nucleosome_color]*num_nucleosomes
    if plot_spheres:
        mlab.points3d(entry_pos[0, 0], entry_pos[0, 1], entry_pos[0, 2], scale_factor=5, figure=mfig,
                color=colors[0], **kwargs)

    if plot_nucleosomes:
        plot_nucleosome(entry_rot=entry_rots[0], entry_pos=entry_pos[0],
            Lw=w_ins[0]+w_outs[0], helix_params=helix_params,
            mfig=mfig, color=colors[0], **kwargs)
    if plot_entry:
        mlab.quiver3d(*entry_pos[0], *entry_rots[0][:,0], mode='arrow',
                      scale_factor=5, color=(1,0,0))
        mlab.quiver3d(*entry_pos[0], *entry_rots[0][:,1], mode='arrow',
                      scale_factor=5, color=(0,1,0))
        mlab.quiver3d(*entry_pos[0], *entry_rots[0][:,2], mode='arrow',
                      scale_factor=5, color=(0,0,1))
    for i in range(1,num_nucleosomes):
        entry_orientation = entry_rots[i,:,2]
        prev_linker = -entry_orientation/np.linalg.norm(entry_orientation)
        mu_in = (b - 1)/2 - w_ins[i]
        mu_out = (b - 1)/2 - w_outs[i-1]
        prev_exit_pos = entry_pos[i] + lpb*prev_linker*(mu_in + links[i-1] + mu_out)
        mlab.plot3d([prev_exit_pos[0], entry_pos[i][0]],
                    [prev_exit_pos[1], entry_pos[i][1]],
                    [prev_exit_pos[2], entry_pos[i][2]],
                    tube_radius=r_dna,
                    figure=mfig, color=(0.3,0.3,0.3), **kwargs)
        if plot_exit:
            mlab.quiver3d(*entry_pos[i], *entry_rots[i][:,0], mode='arrow',
                          scale_factor=5, color=(1,0,0))
            mlab.quiver3d(*entry_pos[i], *entry_rots[i][:,1], mode='arrow',
                          scale_factor=5, color=(0,1,0))
            mlab.quiver3d(*entry_pos[i], *entry_rots[i][:,2], mode='arrow',
                          scale_factor=5, color=(0,0,1))
        if plot_nucleosomes:
            plot_nucleosome(entry_rot=entry_rots[i], entry_pos=entry_pos[i],
                Lw=w_ins[i]+w_outs[i], helix_params=helix_params,
                mfig=mfig, color=colors[i], **kwargs)
        if plot_spheres:
            mlab.points3d(entry_pos[i, 0], entry_pos[i, 1], entry_pos[i, 2], scale_factor=5, figure=mfig,
                color=colors[i], **kwargs)
    return mfig

def visualize_MLC_chain(entry_pos, r_dna=dna_params['r_dna'],
        mfig=None, palette="husl", nucleosome_color=None,
        unwraps=None, plot_entry=False, plot_exit=False, plot_spheres=True, entry_rots=None, **kwargs):
    """Visualize output of :py:func:`minimum_energy_no_sterics`.

    Parameters
    ----------
    entry_rots : (L+1,3,3)
        Orientation of the first bound base pair of each nucleosome.
    entry_pos : (L+1,3)
        Position of the first bound base pair of each nucleosome.
    links : (L,)
        Length of the linkers joining the nucleosomes.
    w_ins : float or (L+1,) array_like
        amount of DNA wrapped on entry side of central dyad base
    w_outs : float or (L+1,) array_like
        amount of DNA wrapped on exit side of central dyad base
    helix_params : (optional) Dict[str, float]
        The helix parameters to use. Defaults to geometry.helix_params_best
    mfig : (optional) matplotlib.Axes
        mlab figure to plot in.
    palette : (optional) str
        Seaborn palette to draw colors from to color each nucleosome

    Returns
    -------
    mfig : mlab.figure
        Figure containing the rendering.
    """

    if mfig is None:
        mfig = mlab.figure()
    num_nucleosomes = len(entry_pos)
    #assert(np.all(entry_rots.shape[:2] == entry_pos.shape))
    colors = sns.color_palette(palette, num_nucleosomes)
    if nucleosome_color is not None:
        colors = [nucleosome_color]*num_nucleosomes
    if plot_spheres:
        mlab.points3d(entry_pos[0, 0], entry_pos[0, 1], entry_pos[0, 2], scale_factor=5, figure=mfig,
                color=colors[0], **kwargs)
    # if plot_entry:
    #     mlab.quiver3d(*entry_pos[0], *entry_rots[0][:,0], mode='arrow',
    #                   scale_factor=5, color=(1,0,0))
    #     mlab.quiver3d(*entry_pos[0], *entry_rots[0][:,1], mode='arrow',
    #                   scale_factor=5, color=(0,1,0))
    #     mlab.quiver3d(*entry_pos[0], *entry_rots[0][:,2], mode='arrow',
    #                   scale_factor=5, color=(0,0,1))
    for i in range(1,num_nucleosomes):
        prev_exit_pos = entry_pos[i-1]
        mlab.plot3d([prev_exit_pos[0], entry_pos[i][0]],
                    [prev_exit_pos[1], entry_pos[i][1]],
                    [prev_exit_pos[2], entry_pos[i][2]],
                    tube_radius=r_dna,
                    figure=mfig, color=(0.3,0.3,0.3), **kwargs)
        if plot_exit:
            mlab.quiver3d(*entry_pos[i], *entry_rots[i][:,0], mode='arrow',
                          scale_factor=5, color=(1,0,0))
            mlab.quiver3d(*entry_pos[i], *entry_rots[i][:,1], mode='arrow',
                          scale_factor=5, color=(0,1,0))
            mlab.quiver3d(*entry_pos[i], *entry_rots[i][:,2], mode='arrow',
                          scale_factor=5, color=(0,0,1))
        if plot_spheres:
            mlab.points3d(entry_pos[i, 0], entry_pos[i, 1], entry_pos[i, 2], scale_factor=5, figure=mfig,
                color=colors[i], **kwargs)
    return mfig


def tangent_propogator_heatmap_in_piball_space(links, w_ins = default_w_in, w_outs = default_w_out,
    tau_n=dna_params['tau_n'], tau_d=dna_params['tau_d'], unwraps=None, color="linker", mfig=None, **kwargs):
    """Plots entry orientations of second nucleosome in each chain starting at the origin
    oriented in the z direction, in pi ball space, given a certain linker length and
    unwrapping amount.

    Parameters
    ----------
    links : (N,) array_like
        lengths of each linker segment for each of N dinucleosome chains
    w_ins : float or (N,) array_like
        amount of DNA wrapped on entry side of second nucleosome for each of N dinucleosome chains
    w_outs : float or (N,) array_like
        amount of DNA wrapped on exit side of first nucleosome for each of N dinucleosome chains
    unwraps : float or (N,) array_like
        amount of DNA unwrapped on both sides for each of the N dinucleosome chains
    color : string, "linker" or "unwrap"
        whether to color the points by linker length or unwrapping amount
    mfig : mlab.figure
        Figure to contain the heatmap.

    Returns
    -------
    mfig : mlab.figure
        Figure containing the heatmap.
    """
    # Create a sphere
    pi = np.pi
    r = pi
    cos = np.cos
    sin = np.sin
    theta, phi = np.mgrid[0:pi:101j, 0:2*pi:101j]
    x = r * sin(theta) * cos(phi)
    y = r * sin(theta) * sin(phi)
    z = r * cos(theta)

    if mfig is None:
        mfig = mlab.figure()

    #draw a translucent pi ball
    mlab.mesh(x,y,z,color=(0.67, 0.77, 0.93), opacity=0.5)

    #first generate axis angle representations of orientations from given linker lengths
    num_chains = links.size
    nodes = np.zeros((num_chains, 3)) #points in pi ball space
    w_ins, w_outs = convert.resolve_wrapping_params(unwraps, w_ins, w_outs, num_chains)

    for i in range(num_chains):
        #compute orientation of second nucleosome in dinucleosome chain (assumes
        #first entry orientation is the identity)
        Onext = OmegaNextEntry(links[i], tau_n=tau_n, tau_d=tau_d,
                w_in=w_ins[i], w_out=w_outs[i], helix_params=helix_params_best)
        axis, angle = ncr.axis_angle_from_matrix(Onext)
        #scale unit vector by angle of rotation
        nodes[i] = axis*angle

    #plot points in pi ball space
    pts = mlab.points3d(nodes[:, 0], nodes[:, 1], nodes[:, 2], scale_factor = 0.25, **kwargs)
    pts.glyph.scale_mode = 'scale_by_vector'
    #color by linker length
    if color == "linker":
        colors = 1.0 * (links - min(links))/max((links - min(links)))
    #color by unwrapping amount
    else:
        colors = 1.0 * (unwraps - min(unwraps))/(max(unwraps) - min(unwraps))

    pts.mlab_source.dataset.point_data.scalars = colors
    mlab.axes()
    return mfig, nodes

