"""
Plotting functions for the core-loss transition potentials.

A port of the new abtem's transition-potential visualization: the domain-coloring figure of the
real-space transition potentials (the new abtem's ``TransitionPotential.show`` /
``AmplitudePhaseArtist``) and the per-transition integrated-intensity bar chart from the new
abtem's core-loss demo.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .core_loss import CoreLossTransitionPotential

__all__ = ["plot_transition_potentials", "plot_transition_potential_intensities"]


def _get_norm(vmin: float, vmax: float, power: float) -> matplotlib.colors.Normalize:
    """
    The display norm for the amplitudes: linear for ``power == 1``, a power law otherwise.

    Port of the new abtem's ``_get_norm`` (without the log-scale option, which the
    transition-potential figures do not use).
    """

    if power == 1.:
        return matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    return matplotlib.colors.PowerNorm(gamma=power, vmin=vmin, vmax=vmax)


def _default_cbar_scalar_formatter() -> ScalarFormatter:
    """The new abtem's default colorbar formatter for scalar (amplitude) axes."""

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))
    return formatter


def _transition_label(transition) -> str:
    """
    The transition's label, with the quantum numbers as plain Python ints.

    The legacy ``ProjectedAtomicTransition.__str__`` formats the state tuples, and tuples
    render their elements with ``repr`` — which shows the numpy quantum numbers as
    ``np.uint64(0)`` instead of ``0``.
    """

    return (f'{tuple(int(q) for q in transition._bound_state)} -> '
            f'{tuple(int(q) for q in transition._continuum_state)}')


def plot_transition_potentials(transition_potential: CoreLossTransitionPotential,
                               power: float = 1.0,
                               vmin: float = None,
                               vmax: float = None,
                               common_color_scale: bool = True,
                               phase_cmap: str = 'hsv',
                               cbar: bool = True,
                               figsize: tuple = None,
                               ax=None) -> plt.Figure:
    """
    Plot the transition potentials with domain coloring.

    A port of the new abtem's ``TransitionPotential.show``: each transition is shown in its own
    panel as the centered real-space transition potential (the inverse Fourier transform of the
    reciprocal-space array), with a gray amplitude background and the phase overlaid, weighted by
    the power-scaled amplitude so that the phase is only visible where the amplitude is strong.

    Parameters
    ----------
    transition_potential : abtem_tools.CoreLossTransitionPotential
        The transition potentials to plot.
    power : float, optional
        The power law applied to the amplitude for display. Default is 1.
    vmin : float, optional
        The minimum of the amplitude color scale. Default is 0.
    vmax : float, optional
        The maximum of the amplitude color scale. The default is the maximum amplitude over all
        panels (or over the single panel, if ``common_color_scale`` is False).
    common_color_scale : bool, optional
        Whether all panels share the same amplitude color scale. Default is True.
    phase_cmap : str, optional
        The colormap of the phase overlay. The new abtem's default is 'hsluv' (a perceptually
        uniform hue colormap); since it requires the optional ``hsluv`` package, this defaults to
        'hsv'.
    cbar : bool, optional
        Whether to show the phase (``arg``) and amplitude (``abs``) colorbars. Default is True.
    figsize : (float, float), optional
        The size of the figure. Default is (4 inches per panel, 4 inches).
    ax : matplotlib axis, optional
        The axis to plot on. Only supported for a single transition; a figure and axes are created
        otherwise.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """

    if not isinstance(transition_potential, CoreLossTransitionPotential):
        raise TypeError('transition_potential must be a CoreLossTransitionPotential')

    # The centered real-space transition potentials, the new abtem's to_images()
    array = np.fft.fftshift(np.fft.ifft2(transition_potential.array, axes=(-2, -1)))
    n = len(transition_potential)
    amplitudes = np.abs(array)

    if ax is not None:
        if n != 1:
            raise ValueError('an axis was given, but there are multiple transitions')
        fig = ax.figure
        axes = [ax]
    else:
        if figsize is None:
            figsize = (4. * n, 4.)
        fig, axes = plt.subplots(1, n, squeeze=False, figsize=figsize, layout='constrained')
        axes = axes.ravel()

    if common_color_scale:
        limits = [(vmin, float(amplitudes.max()) if vmax is None else vmax)] * n
    else:
        limits = [(vmin, float(amplitudes[i].max()) if vmax is None else vmax) for i in range(n)]
    limits = [(0. if lo is None else lo, hi) for lo, hi in limits]

    extent = transition_potential.extent
    extent = (-0.5 * extent[0], 0.5 * extent[0], -0.5 * extent[1], 0.5 * extent[1])

    phase_images = []
    amplitude_images = []
    for i, panel_ax in enumerate(axes):
        norm = _get_norm(*limits[i], power=power)
        amplitude_images.append(
            panel_ax.imshow(amplitudes[i].T,
                            origin='lower',
                            interpolation='none',
                            cmap='gray',
                            zorder=-1,
                            extent=extent))
        amplitude_images[-1].set_norm(norm)
        phase_images.append(
            panel_ax.imshow(np.angle(array[i]).T,
                            origin='lower',
                            interpolation='none',
                            alpha=np.clip(norm(amplitudes[i]), 0., 1.).T,
                            vmin=-np.pi,
                            vmax=np.pi,
                            cmap=phase_cmap,
                            extent=extent))
        panel_ax.set_title(_transition_label(transition_potential.transitions[i]))
        panel_ax.set_xlabel('x [Å]')
        panel_ax.set_ylabel('y [Å]')

    if cbar:
        for i, panel_ax in enumerate(axes):
            divider = make_axes_locatable(panel_ax)
            phase_cax = divider.append_axes('right', size='5%', pad=0.08)
            amplitude_cax = divider.append_axes('right', size='5%', pad=0.08)
            phase_cbar = fig.colorbar(phase_images[i], cax=phase_cax)
            phase_cbar.set_label('arg', rotation=0, ha='center', va='top')
            phase_cbar.set_ticks([-np.pi, -np.pi / 2., 0., np.pi / 2., np.pi])
            phase_cbar.set_ticklabels([r'$-\pi$', r'$-\dfrac{\pi}{2}$', r'$0$',
                                       r'$\dfrac{\pi}{2}$', r'$\pi$'])
            phase_cbar.ax.yaxis.set_label_coords(0.5, -0.02)
            amplitude_cbar = fig.colorbar(amplitude_images[i], cax=amplitude_cax,
                                          format=_default_cbar_scalar_formatter())
            amplitude_cbar.set_label('abs', rotation=0, ha='center', va='top')
            amplitude_cbar.ax.yaxis.set_label_coords(0.5, -0.02)
            amplitude_cbar.ax.yaxis.set_offset_position('left')

    return fig


def plot_transition_potential_intensities(transition_potential: CoreLossTransitionPotential,
                                          max_angle: float = None,
                                          ax=None) -> plt.Figure:
    """
    Bar chart of the integrated intensity of each transition, with the cumulative fraction.

    A port of the per-transition intensity figure from the new abtem's core-loss demo: the bars
    show the integrated intensity of each transition, sorted in descending order, and the red line
    shows the cumulative fraction of the total, which is used to decide how many transitions to
    keep with ``filter_by_intensity``.

    Parameters
    ----------
    transition_potential : abtem_tools.CoreLossTransitionPotential
        The transition potentials to plot.
    max_angle : float, optional
        The maximum angle [mrad] to include in the integrated intensity.
    ax : matplotlib axis, optional
        The axis to plot on. A figure and axis are created if None.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """

    if not isinstance(transition_potential, CoreLossTransitionPotential):
        raise TypeError('transition_potential must be a CoreLossTransitionPotential')

    intensities = np.asarray(transition_potential.integrated_intensities(max_angle=max_angle))
    order = np.argsort(-intensities)
    intensities = intensities[order]
    labels = [_transition_label(transition_potential.transitions[i]) for i in order]

    if ax is None:
        fig, ax1 = plt.subplots(figsize=(6., 4.), layout='constrained')
    else:
        fig = ax.figure
        ax1 = ax

    ax1.bar(labels, intensities)
    ax1.set_xticks(labels, labels, rotation=90)
    ax1.set_ylabel(r'$\int V_{fi, \ local} \mathrm{d}k \ [eV]$')

    ax2 = ax1.twinx()
    ax2.axhline(95, color='k', label='95 %')
    ax2.text(labels[-1], 94, '95 %', color='k', ha='center', va='top')
    ax2.plot(labels, np.cumsum(intensities / intensities.sum()) * 100, 'r.-')
    ax2.set_ylim([0, 105])
    ax2.set_ylabel('Cumulative local potential [%]')

    return fig
