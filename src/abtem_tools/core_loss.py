"""
Tools for double-channeling core-loss multislice simulations with abtem-legacy.

This module ports the brute-force (non-S-matrix) double-channeling core-loss driver of the new abtem
package to the legacy numpy/cupy stack. It is intended for core-loss filtered diffraction (e.g. C KLL
or Si KLL): the elastic wave is propagated through a multislice potential with the ordinary Fresnel
multislice algorithm, and at every slice containing atoms of the transition element the local elastic
wave is scattered with the Dwyer transition potential. The resulting scattered waves are either
re-propagated through all remaining slices (double-channeling) or detected immediately
(single-channeling, the first-order approximation).

The port is first order in the transition potential and includes no absorption; the elastic wave is
never attenuated.

Convention differences versus the new abtem (this port follows abtem-legacy):

- The interaction parameter sigma(E) is already absorbed into the legacy transition-potential
  evaluation, so :meth:`CoreLossTransitionPotential.scatter` does not multiply by it (the new abtem
  compensates by multiplying ``energy2sigma(energy)`` in its ``scatter``; the product is the same).
- The legacy filling factor is ``sqrt(4 * l + 2)`` rather than the new abtem's ``sqrt(2)``; the two
  are identical for K-shell transitions (``l = 0``), which is the main use case (e.g. C KLL, Si KLL).
- The legacy transition-potential machinery integrates over a fixed radial grid
  (``linspace(0, 20, 10000)``). The new abtem's reciprocal-space mask
  (``k <= 2/3 * max(k)``, computed per transition because the axial momentum transfer differs for
  each transition) is applied by :attr:`CoreLossTransitionPotential.array`.

numpy 2.x compatibility
-----------------------

The legacy :class:`abtem.ProjectedAtomicTransition.overlap_integral` calls ``np.trapz``, which was
removed in numpy 2.0 (renamed to ``np.trapezoid``), so the legacy transition-potential evaluation
raises ``AttributeError`` under a numpy 2.x installation. This module works around that with a set
of drop-in subclasses: :class:`CoreLossProjectedAtomicTransition` overrides ``overlap_integral``
with the same computation using the available name, and :class:`CoreLossSubshellTransitions` /
:class:`CoreLossSubshellTransitionsArrays` return instances of it from
``get_transition_potentials``. :class:`CoreLossTransitionPotential` also upgrades any plain legacy
transitions it is given, so it works with transitions obtained from either package.
"""
import abtem
import abtem.ionization.transitions
import ase
import numpy as np
from bisect import bisect_left
from typing import List, Sequence, Union

from .probes import PositionAwareProbe

__all__ = ["CoreLossProjectedAtomicTransition", "CoreLossSubshellTransitions",
           "CoreLossSubshellTransitionsArrays", "CoreLossTransitionPotential", "CoreLossProbe"]


def _trapz(values, x, axis) -> np.ndarray:
    """
    Integrate ``values`` over ``x`` along ``axis``, using ``np.trapezoid`` or ``np.trapz``.

    The legacy :meth:`abtem.ProjectedAtomicTransition.overlap_integral` calls ``np.trapz``, which
    was removed in numpy 2.0 (renamed to ``np.trapezoid``); this uses whichever name is available.
    """
    trapz = getattr(np, 'trapezoid', None)
    if trapz is None:
        trapz = np.trapz
    return trapz(values, x, axis=axis)


class CoreLossProjectedAtomicTransition(abtem.ionization.transitions.ProjectedAtomicTransition):
    """
    A legacy :class:`abtem.ProjectedAtomicTransition` that is compatible with numpy 2.x.

    The legacy :meth:`~abtem.ProjectedAtomicTransition.overlap_integral` calls ``np.trapz``, which
    was removed in numpy 2.0 (renamed to ``np.trapezoid``), so evaluating the transition potential
    of a plain legacy instance raises ``AttributeError`` under a numpy 2.x installation. This
    subclass overrides it with the same computation using the name available in the installed
    numpy, so the transition potential can be evaluated under either.
    """

    def overlap_integral(self, k, lprimeprime) -> np.ndarray:
        rmax = 20
        grid = 2 * np.pi * k * abtem.ionization.transitions.units.Bohr
        r = np.linspace(0, rmax, 10000)
        values = (self._bound_wave(r) *
                  abtem.ionization.transitions.spherical_jn(lprimeprime, grid[:, None] * r[None]) *
                  self._continuum_wave(r))
        return _trapz(values, r, axis=1)


def _as_numpy2_transition(transition) -> CoreLossProjectedAtomicTransition:
    """
    Return ``transition`` as an instance of :class:`CoreLossProjectedAtomicTransition`.

    Plain legacy :class:`abtem.ProjectedAtomicTransition` instances are converted in place by
    reassigning their ``__class__``, which preserves the object identity and any per-instance
    attributes or caches. Instances of :class:`CoreLossProjectedAtomicTransition` itself or of any
    other subclass are returned unchanged.

    Parameters
    ----------
    transition : abtem.ProjectedAtomicTransition
        The transition potential to convert.

    Returns
    -------
    CoreLossProjectedAtomicTransition
        The same object, with its class re-assigned if necessary.
    """
    if type(transition) is abtem.ionization.transitions.ProjectedAtomicTransition:
        transition.__class__ = CoreLossProjectedAtomicTransition
    return transition


class CoreLossSubshellTransitions(abtem.ionization.transitions.SubshellTransitions):
    """
    A legacy :class:`abtem.SubshellTransitions` that yields numpy 2.x compatible transition
    potentials.

    :meth:`get_transition_potentials` is overridden to return
    :class:`CoreLossProjectedAtomicTransition` objects instead of plain
    :class:`abtem.ProjectedAtomicTransition` objects, so the resulting potentials can be evaluated
    under a numpy 2.x installation. Everything else, including the gpaw calculation of the bound
    and continuum states, is inherited unchanged.
    """

    def get_transition_potentials(self, *args, **kwargs):
        return [_as_numpy2_transition(t) for t in super().get_transition_potentials(*args, **kwargs)]


class CoreLossSubshellTransitionsArrays(abtem.ionization.transitions.SubshellTransitionsArrays):
    """
    A legacy :class:`abtem.SubshellTransitionsArrays` that yields numpy 2.x compatible transition
    potentials.

    The gpaw-free counterpart of :class:`CoreLossSubshellTransitions` for precomputed transition
    arrays stored in npz files. Since :meth:`read` is a classmethod that constructs its own class,
    it is inherited unchanged and already returns instances of this class.
    """

    def get_transition_potentials(self, *args, **kwargs):
        return [_as_numpy2_transition(t) for t in super().get_transition_potentials(*args, **kwargs)]


def _validate_exit_planes(exit_planes, num_slices) -> List[int]:
    """
    Validate the exit planes and return them as a sorted list of unique plane indices.

    Plane index ``i`` is the plane immediately after slice ``i`` (``0 <= i < num_slices``).

    Parameters
    ----------
    exit_planes : int, list of int or None
        If None, only the final plane is used. If an integer ``p``, every ``p``-th plane is used
        (``p = 1`` gives all planes). If a list, the given plane indices are used.
    num_slices : int
        The number of slices of the potential.

    Returns
    -------
    list of int
        The sorted, unique exit plane indices. The final plane (``num_slices - 1``) is always
        included.
    """

    if exit_planes is None:
        return [num_slices - 1]

    if isinstance(exit_planes, (int, np.integer)):
        if exit_planes < 1:
            raise ValueError('exit_planes must be a positive integer, a list of plane indices, or None')
        if exit_planes > num_slices:
            return [num_slices - 1]
        planes = list(range(exit_planes - 1, num_slices, int(exit_planes)))
        if planes[-1] != num_slices - 1:
            planes.append(num_slices - 1)
        return planes

    planes = sorted({int(p) for p in exit_planes})
    for p in planes:
        if not 0 <= p < num_slices:
            raise ValueError(f'exit plane {p} is out of range [0, {num_slices - 1}]')
    return planes


class CoreLossTransitionPotential:
    """
    A set of projected atomic transition potentials for a single element, with site filtering.

    Wraps one or more legacy :class:`abtem.ProjectedAtomicTransition` objects (e.g. created with
    ``SubshellTransitions.get_transition_potentials``) into a single object whose reciprocal-space
    array is stacked over the transitions. In addition to the legacy transition-potential machinery,
    it adds the site-filtering functionality of the new abtem: an absolute intensity threshold
    relative to the local elastic wave (:meth:`absolute_threshold`, :meth:`filter_sites`) and a
    filter that keeps the transitions carrying a given fraction of the total integrated intensity
    (:meth:`filter_by_intensity`).

    Parameters
    ----------
    transitions : sequence of abtem.ProjectedAtomicTransition
        The transition potentials to wrap. All transitions must describe the same element and be
        defined on the same grid and at the same energy. Plain legacy
        :class:`abtem.ProjectedAtomicTransition` instances are converted in place to
        :class:`CoreLossProjectedAtomicTransition` so that they can be evaluated under numpy 2.x
        (the legacy ``overlap_integral`` calls ``np.trapz``, which numpy 2.0 removed); the
        conversion preserves the object identity.
    """

    def __init__(self, transitions: Sequence):
        self.transitions = [_as_numpy2_transition(t) for t in transitions]

        if len(self.transitions) == 0:
            raise RuntimeError('Must provide at least one transition potential.')

        for transition in self.transitions:
            if not isinstance(transition, abtem.ionization.transitions.ProjectedAtomicTransition):
                raise TypeError('transitions must be abtem.ProjectedAtomicTransition objects')

        reference = self.transitions[0]
        for transition in self.transitions:
            if transition._Z != reference._Z:
                raise RuntimeError('All transitions must describe the same element')
            if tuple(transition.gpts) != tuple(reference.gpts):
                raise RuntimeError('All transitions must have the same number of grid points')
            if not np.allclose(transition.extent, reference.extent):
                raise RuntimeError('All transitions must have the same extent')
            if not np.allclose(transition.sampling, reference.sampling):
                raise RuntimeError('All transitions must have the same sampling')
            if transition.energy != reference.energy:
                raise RuntimeError('All transitions must have the same energy')

        self._Z = reference._Z
        self._array_cache = None
        self._device_array_cache = {}
        self._local_potential_cache = {}

    @property
    def Z(self) -> int:
        """The atomic number of the transitions."""
        return self._Z

    @property
    def gpts(self) -> tuple:
        """The number of grid points of the transition potentials."""
        return self.transitions[0].gpts

    @property
    def extent(self) -> tuple:
        """The extent of the transition potentials [Å]."""
        return self.transitions[0].extent

    @property
    def sampling(self) -> tuple:
        """The sampling of the transition potentials [1 / Å]."""
        return self.transitions[0].sampling

    @property
    def energy(self) -> float:
        """The energy of the transitions [eV]."""
        return self.transitions[0].energy

    @property
    def array(self) -> np.ndarray:
        """
        The stacked reciprocal-space transition potentials.

        Shape is ``(n_transitions, gpts[0], gpts[1])``. Built on first access and cached.

        The new abtem's reciprocal-space mask is applied, so that only scattering vectors with
        magnitude ``k <= 2/3 * max(k)`` contribute; ``k`` includes the axial momentum transfer of
        the transition, which differs for each transition, so the mask is computed per transition.
        """

        if self._array_cache is None:
            array = np.stack([t._evaluate_potential() for t in self.transitions]).astype(
                np.complex64, copy=False)
            # The new abtem's reciprocal-space mask (k <= 2/3 * max(k)), applied per transition
            # because the axial momentum transfer kz = k0 - kn depends on the energy loss of the
            # transition. Zeroing the high-k pixels here is equivalent to the new abtem skipping
            # them in the form-factor accumulation: every pixel has k > 0 (the center pixel has
            # k = |kz|), so no division by zero is introduced.
            kx, ky = abtem.utils.spatial_frequencies(self.gpts, self.sampling)
            kt, _ = abtem.utils.polar_coordinates(kx, ky)
            for i, t in enumerate(self.transitions):
                k = np.sqrt(kt**2 + t.momentum_transfer**2)
                array[i][k > np.max(k) * 2. / 3.] = 0.
            self._array_cache = array
        return self._array_cache

    def __len__(self) -> int:
        return len(self.transitions)

    def _array_on_device(self, xp) -> np.ndarray:
        """
        Get the stacked reciprocal-space transition potentials as an array on the given device.

        Parameters
        ----------
        xp : numpy or cupy
            The array module of the device.

        Returns
        -------
        ndarray
            The stacked transition potentials, shape ``(n_transitions, gpts[0], gpts[1])``.
        """

        if xp not in self._device_array_cache:
            self._device_array_cache[xp] = xp.asarray(self.array)
        return self._device_array_cache[xp]

    def _local_potential(self, xp) -> np.ndarray:
        """
        Get the sum of the real-space local potentials over all transitions, on the given device.

        This is the quantity compared to the local wave intensity to rank scattering sites; it is
        computed without any angular mask, mirroring the new abtem.

        Parameters
        ----------
        xp : numpy or cupy
            The array module of the device.

        Returns
        -------
        ndarray
            The local potential, shape ``(gpts[0], gpts[1])``.
        """

        if xp not in self._local_potential_cache:
            ifft2 = abtem.device.get_device_function(xp, 'ifft2')
            # the device ifft2 is destructive by default, so operate on a copy
            array = ifft2(self._array_on_device(xp).copy())
            self._local_potential_cache[xp] = (xp.abs(array) ** 2).sum(axis=0)
        return self._local_potential_cache[xp]

    def local_potential(self, max_angle: float = None, space: str = 'real') -> np.ndarray:
        """
        The local potential (intensity) of the transition potentials.

        Parameters
        ----------
        max_angle : float, optional
            The maximum angle [mrad] to include in the local potential. If None, no angular mask is
            applied.
        space : 'real' or 'reciprocal'
            The space in which to calculate the local potential.

        Returns
        -------
        numpy.ndarray
            The local potential, shape ``(n_transitions, gpts[0], gpts[1])``.
        """

        if space == 'reciprocal':
            array = np.abs(self.array) ** 2
        elif space == 'real':
            array = np.abs(np.fft.ifft2(self.array)) ** 2
        else:
            raise ValueError('space must be "reciprocal" or "real"')

        if max_angle is not None:
            kx, ky = abtem.utils.spatial_frequencies(self.gpts, self.sampling)
            alpha, _ = abtem.utils.polar_coordinates(kx, ky)
            alpha *= abtem.utils.energy2wavelength(self.energy) * 1e3
            array = array * (alpha <= max_angle)

        return array

    def integrated_intensities(self, max_angle: float = None) -> np.ndarray:
        """
        The integrated intensity of each transition potential.

        Parameters
        ----------
        max_angle : float, optional
            The maximum angle [mrad] to include. If None, no angular mask is applied.

        Returns
        -------
        numpy.ndarray
            The integrated intensity of each transition, shape ``(n_transitions,)``.
        """

        array = self.local_potential(max_angle=max_angle, space='real')
        return array.sum(axis=(-2, -1)) * np.prod(self.sampling)

    def filter_by_intensity(self, threshold: float, max_angle: float = None) -> 'CoreLossTransitionPotential':
        """
        Keep the transitions that carry the given fraction of the total integrated intensity.

        The transitions are sorted by integrated intensity, and the smallest number of the strongest
        transitions whose cumulative integrated intensity reaches ``threshold`` is kept.

        Parameters
        ----------
        threshold : float
            The cumulative fraction of the total integrated intensity to keep (0 < threshold <= 1).
        max_angle : float, optional
            The maximum angle [mrad] to include in the integrated intensity.

        Returns
        -------
        CoreLossTransitionPotential
            A new transition potential containing only the kept transitions.
        """

        intensities = self.integrated_intensities(max_angle=max_angle)
        order = np.argsort(-intensities)
        cumulative = np.cumsum(intensities[order]) / intensities.sum()
        index = int(np.searchsorted(cumulative, threshold)) + 1
        return self.__class__([self.transitions[i] for i in order[:index]])

    def absolute_threshold(self, waves: abtem.Waves, threshold: float = 1.) -> float:
        """
        Calculate an absolute intensity threshold for scattering-site filtering.

        The threshold is the intensity of the pixel that carries the given cumulative fraction of
        the total overlap between the local potential and the local wave intensity, so that
        scattering sites whose local potential does not overlap the wave above this level are
        negligible.

        Parameters
        ----------
        waves : abtem.Waves
            The local elastic wave at the scattering plane.
        threshold : float
            The cumulative fraction of the total overlap to keep (0 < threshold < 1). A threshold
            of 1.0 or above disables filtering and returns 0.0.

        Returns
        -------
        float
            The absolute intensity threshold.
        """

        if threshold >= 1.:
            return 0.

        xp = abtem.device.get_array_module(waves.array)
        abs2 = abtem.device.get_device_function(xp, 'abs2')
        fft2 = abtem.device.get_device_function(xp, 'fft2')
        fft2_convolve = abtem.device.get_device_function(xp, 'fft2_convolve')

        local_potential = self._local_potential(xp)
        # the device FFTs are destructive by default, so pass fresh arrays
        signal = local_potential.astype(xp.complex64)
        kernel = fft2(abs2(waves.array).astype(xp.complex64))
        overlap = fft2_convolve(signal, kernel).real

        overlap = abtem.device.asnumpy(overlap).ravel()
        overlap = np.sort(overlap)[::-1]

        if overlap[0] <= 0.:
            return 0.

        cumulative = np.cumsum(overlap)
        cumulative /= cumulative[-1]
        index = min(int(np.searchsorted(cumulative, threshold, side='left')), len(overlap) - 1)
        tolerance = 8. * np.finfo(overlap.dtype).eps * float(overlap[0])
        return max(float(overlap[index]) - tolerance, 0.)

    def validate_sites(self, sites) -> np.ndarray:
        """
        Validate the scattering sites and return their xy-positions [Å].

        Parameters
        ----------
        sites : ase.Atoms, ase.Atom or array
            The scattering sites. If an :class:`ase.Atoms` object, only the atoms of the transition
            element are used. If an :class:`ase.Atom`, it is used only if it matches the transition
            element. Otherwise it is interpreted as an ``(n, 2)`` array of xy-positions [Å].

        Returns
        -------
        numpy.ndarray
            The xy-positions of the sites [Å], shape ``(n, 2)``.
        """

        if isinstance(sites, ase.Atoms):
            sites = sites[sites.numbers == self._Z].positions[:, :2]
        elif isinstance(sites, ase.Atom):
            if sites.number == self._Z:
                sites = sites.position[:2]
            else:
                return np.zeros((0, 2), dtype=np.float32)
        else:
            sites = np.array(sites)
            if len(sites.shape) == 1:
                sites = sites[None]

        return np.array(sites, dtype=np.float32)

    def filter_sites(self, waves: abtem.Waves, sites, threshold: float) -> np.ndarray:
        """
        Keep only the scattering sites whose local potential overlaps the local wave above the given
        intensity threshold.

        Parameters
        ----------
        waves : abtem.Waves
            The local elastic wave at the scattering plane.
        sites : ase.Atoms, ase.Atom or array
            The scattering sites, validated with :meth:`validate_sites`.
        threshold : float
            The absolute intensity threshold. Sites are kept where the overlap of their local
            potential with the local wave intensity exceeds this value. If None or <= 0, no
            filtering is applied.

        Returns
        -------
        numpy.ndarray
            The xy-positions of the kept sites [Å], shape ``(n, 2)``.
        """

        sites = self.validate_sites(sites)

        if (threshold is None) or (threshold <= 0.):
            return sites

        xp = abtem.device.get_array_module(waves.array)
        abs2 = abtem.device.get_device_function(xp, 'abs2')
        local_potential = self._local_potential(xp)
        abs2_waves = abs2(waves.array).reshape((-1,) + tuple(self.gpts))
        gy, gx = self.gpts

        # periodic shift of the local potential to each site, vectorized over the site chunk
        # (equivalent to np.roll(local_potential, (sy, sx), axis=(0, 1)) for each site)
        iy = xp.asarray(np.arange(gy, dtype=np.int64).reshape((1, -1, 1)))
        ix = xp.asarray(np.arange(gx, dtype=np.int64).reshape((1, 1, -1)))
        rounded_sites = np.round(sites / np.array(self.sampling)).astype(np.int64)

        kept = np.zeros(len(sites), dtype=bool)
        for start, end in abtem.utils.generate_batches(len(sites), max_batch=1024):
            sy = xp.asarray(rounded_sites[start:end, 0].reshape((-1, 1, 1)))
            sx = xp.asarray(rounded_sites[start:end, 1].reshape((-1, 1, 1)))
            shifted = local_potential[(iy - sy) % gy, (ix - sx) % gx]
            overlaps = (shifted[:, None] * abs2_waves[None]).sum(axis=(-2, -1))
            kept[start:end] = abtem.device.asnumpy(overlaps > threshold).any(axis=1)

        return sites[kept]

    def scatter(self, waves: abtem.Waves, sites, threshold: float = None) -> abtem.Waves:
        """
        Scatter the local elastic wave with the transition potentials at the given sites.

        This is a batched version of :meth:`abtem.ProjectedAtomicTransition.build`: each site and
        transition contributes the transition potential (centered at the site) multiplied by the
        local elastic wave.

        Parameters
        ----------
        waves : abtem.Waves
            The local elastic wave at the scattering plane.
        sites : ase.Atoms, ase.Atom or array
            The scattering sites, validated with :meth:`validate_sites`.
        threshold : float, optional
            An absolute intensity threshold; sites are filtered with :meth:`filter_sites` before
            scattering.

        Returns
        -------
        abtem.Waves
            The scattered waves in real space, shape ``(n_sites * n_transitions, gpts[0], gpts[1])``.
        """

        sites = self.validate_sites(sites)

        if threshold is not None:
            sites = self.filter_sites(waves, sites, threshold)

        if len(sites) == 0:
            return abtem.Waves(np.zeros((0,) + tuple(self.gpts), dtype=np.complex64),
                               extent=waves.extent, energy=waves.energy)

        xp = abtem.device.get_array_module(waves.array)
        ifft2 = abtem.device.get_device_function(xp, 'ifft2')

        # positions in pixels
        sites = sites / np.array(self.sampling)
        kernel = abtem.utils.fourier_translation_operator(xp.asarray(sites), self.gpts)

        # fresh array, so the destructive device ifft2 is safe
        array = self._array_on_device(xp)[None] * kernel[:, None]
        array = ifft2(array)

        array = array.reshape(array.shape[:2] + (1,) * (len(waves.array.shape) - 2) + tuple(self.gpts))
        array = array * waves.array[None, None]

        return abtem.Waves(array.reshape((-1,) + tuple(self.gpts)), extent=waves.extent, energy=waves.energy)

    def generate_scattered_waves(self, waves: abtem.Waves, sites, max_batch: int = 1, threshold: float = None):
        """
        Yield the scattered waves in chunks of at most ``max_batch`` scattering sites.

        The sites are filtered with :meth:`filter_sites` once for the whole set (if a threshold is
        given), and :meth:`scatter` is then applied to each chunk.

        Parameters
        ----------
        waves : abtem.Waves
            The local elastic wave at the scattering plane.
        sites : ase.Atoms, ase.Atom or array
            The scattering sites, validated with :meth:`validate_sites`.
        max_batch : int
            The maximum number of sites per chunk.
        threshold : float, optional
            An absolute intensity threshold for :meth:`filter_sites`.

        Yields
        ------
        (numpy.ndarray, abtem.Waves)
            A chunk of site positions [Å] and the corresponding scattered waves.
        """

        sites = self.validate_sites(sites)

        if (threshold is not None) and (threshold > 0.):
            sites = self.filter_sites(waves, sites, threshold)
            threshold = None

        if len(sites) == 0:
            return

        for start, end in abtem.utils.generate_batches(len(sites), max_batch=max_batch):
            yield sites[start:end], self.scatter(waves, sites[start:end], threshold=None)


class CoreLossProbe(PositionAwareProbe):
    """
    A probe that can record core-loss filtered diffraction via double-channeling multislice
    propagation with a transition potential.

    This probe is a :class:`PositionAwareProbe` (so it supports elastic position-aware scanning
    with e.g. probe-centered confocal detectors) and adds the
    :meth:`transition_potential_scan` method for core-loss filtered diffraction. In that scan,
    detectors with a ``detect_at`` method (e.g. a probe-centered
    :class:`~abtem_tools.detectors.ConfocalDetector`) are called with each wave's probe position,
    and the scattered waves of a batch are detected incoherently (detected individually and then
    summed).
    """

    def transition_potential_scan(self,
                                  scan: abtem.scan.AbstractScan,
                                  potential: abtem.potentials.AbstractPotential,
                                  detectors: Union[abtem.detect.AbstractDetector,
                                                   Sequence[abtem.detect.AbstractDetector]],
                                  transition_potential: CoreLossTransitionPotential,
                                  sites: Union[ase.Atoms, None] = None,
                                  exit_planes: Union[int, Sequence[int], None] = None,
                                  double_channel: bool = True,
                                  threshold: float = 1.,
                                  scatter_max_batch: int = 1,
                                  pbar: bool = True) -> Union[abtem.Measurement, List[abtem.Measurement]]:
        """
        Scan the probe and record core-loss scattered waves detected with the given detector(s).

        The elastic wave is propagated through the potential with the ordinary Fresnel multislice
        algorithm. At every slice containing atoms of the transition element, the local elastic wave
        is scattered with the transition potential at those sites. If ``double_channel`` is True,
        each scattered wave is re-propagated through all remaining slices before detection
        (brute-force double-channeling); if False, the scattered waves are detected immediately and
        the same (un-propagated) pattern is added to every exit plane at or after the scattering
        slice.

        The elastic wave is never detected and the entrance plane stays zero. As for
        :meth:`abtem.Probe.scan`, the probe's cutoff scattering angle must cover the detectors'
        maximum angle.

        Parameters
        ----------
        scan : abtem.scan.AbstractScan
            The scan to perform.
        potential : abtem.potentials.AbstractPotential
            The potential to propagate through. Must have the same energy as the probe.
        detectors : abtem.detect.AbstractDetector or list of abtem.detect.AbstractDetector
            The detector(s) to record the scattered waves with. Detectors with a ``detect_at``
            method (e.g. a probe-centered confocal detector) are called with each wave's probe
            position.
        transition_potential : CoreLossTransitionPotential
            The transition potential to scatter with.
        sites : ase.Atoms, optional
            The atoms to use as scattering sites. If None, all atoms of the potential are used.
            Only atoms of the transition element scatter.
        exit_planes : int, list of int or None, optional
            The planes at which to detect the scattered waves. Plane index ``i`` is the plane
            immediately after slice ``i``. If an integer ``p``, every ``p``-th plane is used
            (``p = 1`` gives all planes). If None, only the final plane.
        double_channel : bool
            If True, re-propagate the scattered waves through all remaining slices. Default is
            True.
        threshold : float
            The cumulative fraction of the local overlap intensity to keep when filtering
            scattering sites (0 < threshold < 1). If >= 1, no filtering is applied. Default is
            1.
        scatter_max_batch : int
            The maximum number of sites per scattering batch. Default is 1.
        pbar : bool
            If True, show a progress bar over the scan positions. Default is True.

        Returns
        -------
        abtem.Measurement or list of abtem.Measurement
            A single measurement if there is one detector and one exit plane; otherwise a flat
            list of measurements, ordered detector-major and plane-minor.
        """

        self.grid.match(potential.grid)
        self.grid.check_is_defined()

        detectors = self._validate_detectors(detectors)

        if any(d.save_file is not None for d in detectors):
            raise NotImplementedError('Detectors with a save_file are not supported.')

        num_slices = potential.num_slices
        exit_planes = _validate_exit_planes(exit_planes, num_slices)

        base_measurements = self.validate_scan_measurements(detectors, scan)
        plane_measurements = {}
        for det, base in base_measurements.items():
            plane_measurements[det] = [abtem.Measurement(np.array(base.array, copy=True), calibrations=base.calibrations)
                                       for _ in exit_planes]

        # pre-build all transmission functions so the double-channel inner loop can index into them
        slice_cache = [t for (_, _, t) in potential.generate_transmission_functions(self.energy, max_batch=1)]
        propagator = abtem.FresnelPropagator()

        if sites is not None and not isinstance(sites, ase.Atoms):
            raise TypeError('sites must be an ase.Atoms object')
        if sites is None:
            if not hasattr(potential, 'atoms'):
                raise RuntimeError('Potential has no atoms; pass sites=atoms explicitly')
            atoms = potential.atoms
        else:
            atoms = sites
        if not np.any(atoms.numbers == transition_potential.Z):
            raise RuntimeError(f'No scattering sites matching transition potential for element '
                               f'{transition_potential.Z}')

        # a built PotentialArray only exposes per-slice thicknesses; a builder Potential a uniform thickness
        if hasattr(potential, 'slice_thicknesses'):
            slice_thicknesses = potential.slice_thicknesses
        else:
            slice_thicknesses = potential.slice_thickness
        sliced_sites = abtem.structures.SlicedAtoms(atoms, slice_thicknesses)
        sites_per_slice = [sliced_sites.get_subsliced_atoms(i, atomic_number=transition_potential.Z).atoms
                           for i in range(num_slices)]

        plane_of_slice = {p: k for k, p in enumerate(exit_planes)}

        positions = scan.get_positions()
        progress = abtem.utils.ProgressBar(total=len(positions), disable=not pbar)

        for index, position in enumerate(positions):
            waves = self.build(position)

            buffers = {}
            for det, base in base_measurements.items():
                buffers[det] = [np.zeros(base.array.shape[-2:], dtype=base.array.dtype) for _ in exit_planes]

            for i in range(num_slices):
                t = slice_cache[i]
                waves = t.transmit(waves)
                waves = propagator.propagate(waves, t.thickness)

                sites_slice = sites_per_slice[i]
                if len(sites_slice) == 0:
                    continue

                abs_threshold = transition_potential.absolute_threshold(waves, threshold=threshold)

                for sites_chunk, scattered in transition_potential.generate_scattered_waves(
                        waves, sites_slice, max_batch=scatter_max_batch, threshold=abs_threshold):
                    if len(scattered) == 0:
                        continue

                    if double_channel:
                        k = plane_of_slice.get(i)
                        if k is not None:
                            for det in detectors:
                                buffers[det][k] += self._detect(det, scattered, position).sum(axis=0)

                        if i + 1 == num_slices:
                            continue

                        for j in range(i + 1, num_slices):
                            scattered = slice_cache[j].transmit(scattered)
                            scattered = propagator.propagate(scattered, slice_cache[j].thickness)
                            k = plane_of_slice.get(j)
                            if k is not None:
                                for det in detectors:
                                    buffers[det][k] += self._detect(det, scattered, position).sum(axis=0)

                    else:
                        # single-channel: detect once (un-propagated) and add the same pattern to
                        # every exit plane at or after the scattering slice
                        k0 = bisect_left(exit_planes, i)
                        if k0 < len(exit_planes):
                            for det in detectors:
                                pattern = self._detect(det, scattered, position).sum(axis=0)
                                for k in range(k0, len(exit_planes)):
                                    buffers[det][k] += pattern

            for det in detectors:
                for k in range(len(exit_planes)):
                    scan.insert_new_measurement(plane_measurements[det][k], index, buffers[det][k])

            progress.update(1)

        progress.close()

        if len(detectors) == 1 and len(exit_planes) == 1:
            return plane_measurements[detectors[0]][0]

        measurements = []
        for det in detectors:
            measurements += plane_measurements[det]
        return measurements

    @staticmethod
    def _detect(detector, waves, position) -> np.ndarray:
        """
        Detect a batch of scattered waves incoherently with ``detector``.

        Detectors with a ``detect_at`` method (e.g. a probe-centered confocal detector) are called
        with the probe position of the waves; all other detectors are called with ``detect``. The
        batch dimension is preserved; the caller sums over it.

        Parameters
        ----------
        detector : abtem.detect.AbstractDetector
            The detector to detect the waves with.
        waves : abtem.Waves
            A batch of scattered waves with shape (n, gpts, gpts).
        position : array, shape (2,)
            The probe position of the waves.

        Returns
        -------
        numpy.ndarray
            The detected intensities with shape (n, ...).
        """
        if hasattr(detector, 'detect_at'):
            positions = np.tile(position, (waves.array.shape[0], 1))
            return detector.detect_at(waves, positions)
        return detector.detect(waves)
