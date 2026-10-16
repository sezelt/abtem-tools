import abtem
import numpy as np
from typing import Tuple, Union

__all__ = ["BinnedPixelatedDetector", "ConfocalDetector"]


def _as_pair(value):
    """Convert a scalar or pair value into a pair of floats."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return (float(value), float(value))
    if len(value) != 2:
        raise ValueError('value should be a scalar or a pair, got {}'.format(value))
    return (float(value[0]), float(value[1]))


def _as_int_pair(value):
    """Convert a scalar or pair value into a pair of integers."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value = (value, value)
    if len(value) != 2:
        raise ValueError('value should be a scalar or a pair, got {}'.format(value))
    value = (int(value[0]), int(value[1]))
    if min(value) < 1:
        raise ValueError('gpts must be at least 1, got {}'.format(value))
    return value


def _bin_boundaries(n_old, n_new, xp):
    """
    Compute the pixel boundaries for sum-binning a zero-centered (fftshifted) grid of n_old pixels into n_new bins.

    Bin j spans the input pixels [boundaries[j], boundaries[j + 1]). Both grids span the same extent, so bin j
    collects every input pixel whose center falls inside the angular extent of the j-th output pixel. When
    n_new = n_old // f with f dividing n_old, this reduces to the usual i // f binning.

    Parameters
    ----------
    n_old : int
        Number of pixels of the zero-centered input grid.
    n_new : int
        Number of pixels of the zero-centered binned grid, with 1 <= n_new <= n_old.
    xp : numpy or cupy
        Array module to build the boundary array with.

    Returns
    -------
    array of int
        Array of length n_new + 1 with the bin boundaries.
    """
    j = xp.arange(n_new + 1, dtype=xp.int64)
    # boundaries[j] = ceil((2*j*n_old - (n_old % 2)*n_new) / (2*n_new)), evaluated with exact integer
    # floor division so it works identically on numpy and cupy
    return (2 * j * n_old + (2 - n_old % 2) * n_new - 1) // (2 * n_new)


def _mean_bin_axis(array, n_old, n_new, xp):
    """
    Average the last axis of array from n_old to n_new pixels, assuming both grids are zero-centered
    (fftshifted) and span the same extent. Leading batch dimensions are preserved.

    Each output pixel is the mean of the input pixels inside its extent -- the bin's integrated signal
    divided by the number of input pixels it contains, i.e. its area. This makes the binned array an
    estimate of the intensity field on the coarser grid: a uniform input stays uniform even though the
    bins can contain different numbers of input pixels.

    Parameters
    ----------
    array : array
        Array with n_old pixels along the last axis.
    n_old : int
        Number of pixels of the input grid along the last axis.
    n_new : int
        Number of pixels of the binned grid along the last axis.
    xp : numpy or cupy
        Array module.

    Returns
    -------
    array
        Array with n_new pixels along the last axis, where each pixel is the mean of the input pixels
        that fall inside its extent.
    """
    if n_new == n_old:
        return array
    boundaries = _bin_boundaries(n_old, n_new, xp)
    cumulative = xp.concatenate((xp.zeros(array.shape[:-1] + (1,), dtype=array.dtype),
                                 xp.cumsum(array, axis=-1)), axis=-1)
    sums = cumulative[..., boundaries[1:]] - cumulative[..., boundaries[:-1]]
    counts = boundaries[1:] - boundaries[:-1]
    counts = counts.reshape((1,) * (array.ndim - 1) + (n_new,)).astype(array.dtype)
    return sums / counts


class ConfocalDetector(abtem.detect.AbstractDetector):
    """
    Detector that records the square magnitude of the wave in real space.

    Unlike the pixelated detector, which records the intensity of the Fourier-transformed exit wave function, this
    detector records the intensity at the image plane, which may be used for example for simulating confocal or
    bright-field STEM.

    Parameters
    ----------
    gpts : int or (int, int), optional
        The number of grid points of the detected image. Exactly two of gpts, extent, and sampling must be
        specified; the remaining one is inferred from the other two (extent = gpts * sampling, sampling =
        extent / gpts, gpts = round(extent / sampling)).
    extent : float or (float, float), optional
        The real space extent of the detected image [Å]. Exactly two of gpts, extent, and sampling must be
        specified; see gpts.
    sampling : float or (float, float), optional
        The pixel spacing of the detected image [Å]. Exactly two of gpts, extent, and sampling must be
        specified; see gpts.
    ctf : abtem.transfer.CTF, optional
        A contrast transfer function that is applied to the wave before detection.
    center : (float, float) or 'probe', optional
        The position [Å] in the wave on which the crop box is centered. If 'probe' (default), the crop box is
        centered on the probe position of each wave in the batch; the positions are passed together with the
        waves via detect_at, which PositionAwareProbe does automatically for each detected batch during a
        scan. If given as a pair, the crop box is centered on the given position. If None, the crop box is
        centered on the origin of the wave (so the origin sits at the center of the image); regions outside
        the wave are wrapped (periodic) or zero-padded (non-periodic).
    periodic : bool, optional
        If True (default), the wave is treated as periodic and the crop box wraps around the edges of the
        grid instead of zero-padding regions outside the wave. For probe-centered detection wrapping is the
        physically correct behavior near the edges of the scan; set to False to zero-pad instead.
    save_file : str
        The path to the file used for saving the detector output.
    """

    def __init__(self,
                 gpts: Union[int, Tuple[int, int]] = None,
                 extent: Union[float, Tuple[float, float]] = None,
                 sampling: Union[float, Tuple[float, float]] = None,
                 ctf: abtem.transfer.CTF = None,
                 center: Union[Tuple[float, float], str] = 'probe',
                 periodic: bool = True,
                 save_file: str = None):

        given = [name for name, value in (('gpts', gpts), ('extent', extent), ('sampling', sampling))
                 if value is not None]
        if len(given) != 2:
            raise ValueError('exactly two of gpts, extent, and sampling must be specified (the remaining one is '
                             'inferred from the other two); got: '
                             + (', '.join(given) if given else 'none'))

        gpts = _as_int_pair(gpts)
        extent = _as_pair(extent)
        sampling = _as_pair(sampling)

        if gpts is None:
            gpts = tuple(max(1, int(np.round(e / s))) for e, s in zip(extent, sampling))
        elif extent is None:
            extent = tuple(g * s for g, s in zip(gpts, sampling))
        else:
            sampling = tuple(e / g for e, g in zip(extent, gpts))

        if center is not None and center != 'probe':
            center = _as_pair(center)

        self._gpts = gpts
        self._extent = extent
        self._sampling = sampling
        self._ctf = ctf
        self._center = center
        self._periodic = bool(periodic)

        super().__init__(save_file=save_file)

    @property
    def gpts(self):
        return self._gpts

    @gpts.setter
    def gpts(self, value):
        self._gpts = _as_int_pair(value)
        self._sampling = tuple(e / g for e, g in zip(self._extent, self._gpts))

    @property
    def extent(self):
        return self._extent

    @extent.setter
    def extent(self, value):
        self._extent = _as_pair(value)
        self._gpts = tuple(max(1, int(np.round(e / s))) for e, s in zip(self._extent, self._sampling))

    @property
    def sampling(self):
        return self._sampling

    @sampling.setter
    def sampling(self, value):
        self._sampling = _as_pair(value)
        self._gpts = tuple(max(1, int(np.round(e / s))) for e, s in zip(self._extent, self._sampling))

    @property
    def ctf(self) -> abtem.transfer.CTF:
        return self._ctf

    @ctf.setter
    def ctf(self, value: abtem.transfer.CTF):
        self._ctf = value

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, value):
        if value is not None and value != 'probe':
            value = _as_pair(value)
        self._center = value

    @property
    def periodic(self):
        return self._periodic

    @periodic.setter
    def periodic(self, value):
        self._periodic = bool(value)

    def detect_at(self, waves, positions) -> np.ndarray:
        """
        Detect the waves with the crop box centered on the given probe positions [Å].

        The positions are passed together with the waves, so the detector holds no state: this is how
        probe-centered detection is applied to waves built manually, e.g. a single simulated probe position:

            waves = probe.multislice([position], potential)
            image = detector.detect_at(waves, [position])

        During a scan, PositionAwareProbe calls this method with the waves of each detected batch and their
        probe positions.

        Parameters
        ----------
        waves: Waves object
            The batch of wave functions to detect.
        positions : (float, float) or (N, 2) array
            The position [Å] in the wave on which the crop box is centered: a single position for a single
            wave, or one position per wave in the batch.

        Returns
        -------
            The detected images, as for detect.
        """
        if self._center != 'probe':
            raise ValueError("detect_at requires the detector to be configured with center='probe'; with a "
                             'fixed center the crop is already centered and detect(waves) needs no positions.')
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim == 1:
            positions = positions[None, :]
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError('positions should be a pair [x, y] or an (N, 2) array, '
                             'got shape {}'.format(positions.shape))

        batch = 1 if waves.array.ndim == 2 else waves.array.shape[0]
        if positions.shape[0] != batch:
            raise RuntimeError('detect_at requires one position per wave in the batch (got {} waves and '
                               '{} positions).'.format(batch, positions.shape[0]))

        xp = abtem.device.get_array_module(waves.array)
        abs2 = abtem.device.get_device_function(xp, 'abs2')

        if self._ctf is not None:
            waves = waves.apply_ctf(self._ctf)

        array = abs2(waves.array)
        gpts, extent, sampling = self._target_grid()
        return self._centered_crop(array, waves, gpts, extent, positions, xp)

    def _target_grid(self):
        """
        The target grid of the detected image. All three of gpts, extent, and sampling are fully specified at
        construction (exactly two are given, the third is inferred from the other two).

        Returns
        -------
        gpts, extent, sampling
        """
        return self._gpts, self._extent, self._sampling

    def _bilinear_nodes_and_weights(self, old_gpts, old_extent, gpts, extent, xp):
        nodes = []
        weights = []

        for n, e_old, m, e_new in zip(old_gpts, old_extent, gpts, extent):
            d_old = e_old / n
            d_new = e_new / m

            # the wave is in natural (un-shifted) ordering (index i holds the wave at physical coordinate
            # i * d_old); the image grid is centered on the origin (index j is at (j - m/2) * d_new)
            x_old = xp.arange(n, dtype=xp.float32) * d_old
            x_new = (xp.arange(m, dtype=xp.float32) - m / 2) * d_new

            distances = x_new[None] - x_old[:, None]
            distances[distances < 0.] = np.inf

            w = distances.min(0) / d_old
            w[w == np.inf] = 0.

            # points outside the support of the wave are set to zero
            outside = (x_new < x_old[0]) | (x_new > x_old[-1])
            w[outside] = 0.

            nodes.append(distances.argmin(0))
            weights.append(w)

        v, u = nodes
        vw, uw = weights
        v, u, vw, uw = xp.broadcast_arrays(v[:, None], u[None, :], vw[:, None], uw[None, :])
        return v, u, vw, uw

    def _image_calibrations(self, gpts, extent, sampling):
        """
        Calibrations of the detected image. For the legacy (non-centered) behavior the image covers the wave
        in its conventional ordering; for centered crops the image covers [-extent/2, extent/2) around the
        reference position ('probe' mode labels the axes relative to the probe position, which is the center
        of each individual image).
        """
        if self._center is None and not self._periodic:
            return abtem.measure.calibrations_from_grid(gpts, sampling, names=['x', 'y'], units='Å')

        if self._center is not None and self._center != 'probe':
            reference = self._center
        else:
            reference = (0., 0.)

        return (abtem.measure.Calibration(offset=reference[0] - extent[0] / 2, sampling=sampling[0],
                                          units='Å', name='x'),
                abtem.measure.Calibration(offset=reference[1] - extent[1] / 2, sampling=sampling[1],
                                          units='Å', name='y'))

    def allocate_measurement(self, waves, scan: abtem.scan.AbstractScan = None) -> abtem.Measurement:
        """
        Allocate a Measurement object or an hdf5 file.

        Parameters
        ----------
        waves : Waves or SMatrix object
            The wave function that will define the shape of the detected image.
        scan: Scan object
            The scan object that will define the scan dimensions the measurement.

        Returns
        -------
        Measurement object or str
            The allocated measurement or path to hdf5 file with the measurement data.
        """

        waves.grid.check_is_defined()
        waves.accelerator.check_is_defined()

        gpts, extent, sampling = self._target_grid()
        calibrations = self._image_calibrations(gpts, extent, sampling)

        if scan is None:
            scan_shape = ()
            scan_calibrations = ()
        elif isinstance(scan, tuple):
            scan_shape = scan
            scan_calibrations = (None,) * len(scan)
        else:
            scan_shape = scan.shape
            scan_calibrations = scan.calibrations

        array = np.zeros(scan_shape + gpts, dtype=np.float32)
        measurement = abtem.Measurement(array, calibrations=scan_calibrations + calibrations)
        if isinstance(self.save_file, str):
            measurement = measurement.write(self.save_file)
        return measurement

    def detect(self, waves) -> np.ndarray:
        """
        Calculate the intensity of the wave functions in real space. If requested, the result is cropped and
        resampled to the detector extent and sampling, centering the crop box on the detector center.

        For probe-centered detectors (center='probe') the probe positions must be passed together with the
        waves via detect_at, so this method raises.

        Parameters
        ----------
        waves: Waves object
            The batch of wave functions to detect.

        Returns
        -------
            Detected values. The first dimension indexes the batch size, the second and third indexes the two
            components of the real space coordinate.
        """

        if self._center == 'probe':
            raise RuntimeError("center='probe' requires the probe positions of the batch; use detect_at(waves, "
                               'positions) to pass them together with the waves, or abtem_tools.'
                               'PositionAwareProbe, which passes them to the detector automatically for each '
                               'detected batch.')

        xp = abtem.device.get_array_module(waves.array)
        abs2 = abtem.device.get_device_function(xp, 'abs2')

        if self._ctf is not None:
            waves = waves.apply_ctf(self._ctf)

        array = abs2(waves.array)

        gpts, extent, sampling = self._target_grid()

        if self._center is None and not self._periodic:
            if gpts != waves.gpts:
                interpolate_bilinear = abtem.device.get_device_function(xp, 'interpolate_bilinear')
                v, u, vw, uw = self._bilinear_nodes_and_weights(waves.gpts, waves.extent, gpts, extent, xp)
                array = interpolate_bilinear(array, v, u, vw, uw)
                # interpolate_bilinear clamps its gather to the array boundary, so pixels outside the
                # wave's support (whose weights are zeroed) still receive the boundary value; zero them
                # to give true zero padding. The bounds mirror _bilinear_nodes_and_weights exactly.
                inside = []
                for m, e_new, n, e_old in zip(gpts, extent, waves.gpts, waves.extent):
                    x_new = (xp.arange(m, dtype=xp.float32) - m / 2) * (e_new / m)
                    x_old_max = (xp.arange(n, dtype=xp.float32) * (e_old / n))[-1]
                    inside.append(((x_new >= 0) & (x_new <= x_old_max)).astype(xp.float32))
                mask = inside[0][:, None] * inside[1][None, :]
                if array.ndim == 3:
                    mask = mask[None, ...]
                array = array * mask
            return array

        center = (0., 0.) if self._center is None else self._center
        batch = array.shape[0] if array.ndim == 3 else 1
        centers = np.tile(np.asarray(center, dtype=np.float64), (batch, 1))
        return self._centered_crop(array, waves, gpts, extent, centers, xp)

    def _centered_crop(self, array, waves, gpts, extent, centers, xp):
        """
        Crop and resample the intensity array (batch, n0, n1) to the target grid, centering the crop box on
        the given positions (batch, 2) [Å] in the wave. The wave array is in natural (un-shifted) ordering,
        so a position p maps to index p/d. Regions outside the wave are wrapped around (periodic) or
        zero-padded.
        """
        # single-probe batches are squeezed to 2D by the wave builder; normalize to a batch dimension
        squeezed = array.ndim == 2
        if squeezed:
            array = array[None, ...]

        n0, n1 = waves.gpts
        d_old = (waves.extent[0] / n0, waves.extent[1] / n1)
        d_new = (extent[0] / gpts[0], extent[1] / gpts[1])

        i0 = xp.arange(gpts[0], dtype=xp.float32)
        i1 = xp.arange(gpts[1], dtype=xp.float32)

        # physical coordinate of each image pixel relative to the crop center
        # (image axis 0 <-> x = centers[:, 0], image axis 1 <-> y = centers[:, 1])
        x0 = (xp.asarray(centers[:, 0], dtype=xp.float32)[:, None]
              - extent[0] / 2 + i0[None, :] * d_new[0])  # (batch, m0)
        x1 = (xp.asarray(centers[:, 1], dtype=xp.float32)[:, None]
              - extent[1] / 2 + i1[None, :] * d_new[1])  # (batch, m1)

        # the wave array is in natural (un-shifted) ordering, so physical coordinate 0 is at index 0 and index
        # i holds the wave at physical coordinate i * d_old (mod extent). The source index of a pixel at
        # physical coordinate x is therefore (x / d_old) modulo the number of grid points, with the fractional
        # part giving the bilinear weight; this is the same in both modes.
        s0 = x0 / d_old[0]
        f0 = s0 - xp.floor(s0)
        v0 = (xp.floor(s0) % n0).astype(xp.int32)
        w0 = (v0 + 1) % n0
        s1 = x1 / d_old[1]
        f1 = s1 - xp.floor(s1)
        v1 = (xp.floor(s1) % n1).astype(xp.int32)
        w1 = (v1 + 1) % n1

        if self._periodic:
            valid0 = valid1 = None
        else:
            # zero-pad pixels outside the wave's physical extent [0, extent)
            valid0 = (s0 >= 0) & (s0 < n0)
            valid1 = (s1 >= 0) & (s1 < n1)

        ib = xp.arange(array.shape[0])[:, None, None]
        a0 = v0[:, :, None]
        a1 = w0[:, :, None]
        b0 = v1[:, None, :]
        b1 = w1[:, None, :]
        fa = f0[:, :, None]
        fb = f1[:, None, :]

        out = ((1 - fa) * (1 - fb) * array[ib, a0, b0]
               + (1 - fa) * fb * array[ib, a0, b1]
               + fa * (1 - fb) * array[ib, a1, b0]
               + fa * fb * array[ib, a1, b1])
        if valid0 is not None:
            out = out * valid0[:, :, None] * valid1[:, None, :]
        return out[0] if squeezed else out


class BinnedPixelatedDetector(abtem.PixelatedDetector):
    """
    Pixelated detector that records diffraction patterns binned into a coarser pixel grid.

    Like abtem.PixelatedDetector, but each recorded pixel is the mean of the far-field pixels that fall
    inside its angular extent -- the bin's integrated signal divided by the number of contributing
    pixels, i.e. its area -- instead of recording one pixel per far-field grid point. Averaging a range
    of pixels, rather than sampling points, avoids the aliasing artifacts that point sampling causes
    when resampling diffraction data onto a coarser grid, and the normalization by the number of
    contributing pixels keeps a uniform far field uniform even when the bins contain different pixel
    counts. The binned array is therefore an estimate of the intensity field on the coarser grid, not
    an integrated flux; multiply by the binned pixel area (sampling squared) to recover fluxes.

    The size of the binned grid is specified with exactly one of
    binfactor : int
        An integer binning factor. The far-field size of each axis is divided by this factor (integer
        division). When the factor divides the far-field size, each binned pixel is the mean of exactly
        binfactor x binfactor far-field pixels and the binned angular sampling is binfactor times the
        far-field sampling.
    gpts : int or (int, int)
        The exact number of pixels of the binned patterns. The binned grid spans the same angular extent
        as the far field (cropped to max_angle), so its angular sampling is the far-field sampling scaled
        by the ratio of the far-field to binned grid size, which in general is not an integer; the
        far-field pixels are assigned to bins by their angular position.

    If neither binfactor nor gpts is given, binfactor defaults to 1 and the detector behaves like an
    unbinned abtem.PixelatedDetector. In 'complex' mode the complex amplitudes are coherently averaged.

    Parameters
    ----------
    max_angle : float or str
        The maximum angle of the diffraction patterns. Either a number in radians or one of the strings
        'limit', 'valid', or 'full'.
    mode : str
        'intensity' to record the intensity, 'complex' to record the complex far field.
    save_file : str
        If set, the measurement is written to this hdf5 file.
    binfactor : int
        Integer binning factor (see above).
    gpts : int or (int, int)
        Number of pixels of the binned patterns (see above).
    """

    def __init__(self, max_angle: Union[str, float] = 'valid', mode='intensity', save_file: str = None,
                 binfactor: int = None, gpts: Union[int, Tuple[int, int]] = None):
        if (binfactor is not None) and (gpts is not None):
            raise ValueError('specify either binfactor or gpts, not both')
        if binfactor is not None:
            if (not isinstance(binfactor, (int, np.integer))) or binfactor < 1:
                raise ValueError('binfactor must be a positive integer, got {}'.format(binfactor))
            binfactor = int(binfactor)
        elif gpts is not None:
            gpts = _as_int_pair(gpts)
        else:
            binfactor = 1
        self._binfactor = binfactor
        self._gpts = gpts
        super().__init__(max_angle=max_angle, resample=False, mode=mode, save_file=save_file)

    @property
    def binfactor(self):
        return self._binfactor

    @property
    def gpts(self):
        return self._gpts

    def _binned_gpts(self, gpts_old):
        """
        Compute the pixel count of the binned patterns from the pixel count of the far field.

        Parameters
        ----------
        gpts_old : tuple of int
            The pixel count of the far field before binning.

        Returns
        -------
        tuple of int
            The pixel count of the binned patterns.
        """
        if self._gpts is not None:
            gpts_new = self._gpts
        else:
            gpts_new = (gpts_old[0] // self._binfactor, gpts_old[1] // self._binfactor)
        for old, new in zip(gpts_old, gpts_new):
            if new < 1:
                raise ValueError('binning would leave fewer than one pixel per axis ({} -> {}); '
                                 'choose a smaller binning'.format(gpts_old, gpts_new))
            if new > old:
                raise ValueError('the binned grid ({}) must not be larger than the far field ({}); '
                                 'binning only coarsens the grid, use abtem.PixelatedDetector with '
                                 "resample='uniform' to resample to a finer grid".format(gpts_new, gpts_old))
        return gpts_new

    def allocate_measurement(self, waves, scan: abtem.scan.AbstractScan = None) -> abtem.Measurement:
        """
        Allocate a Measurement object or an hdf5 file.

        Parameters
        ----------
        waves : Waves or SMatrix object
            The wave function that will define the shape of the diffraction patterns.
        scan: Scan object
            The scan object that will define the scan dimensions the measurement.

        Returns
        -------
        Measurement object or str
            The allocated measurement or path to hdf5 file with the measurement data.
        """

        waves.grid.check_is_defined()
        waves.accelerator.check_is_defined()
        abtem.detect.check_max_angle_exceeded(waves, self.max_angle)

        gpts_old = waves.downsampled_gpts(self.max_angle)
        gpts = self._binned_gpts(gpts_old)

        # The binned grid spans the same angular extent as the far field, so its angular sampling is the
        # far-field sampling scaled by the ratio of the far-field to binned grid size.
        new_angular_sampling = (gpts_old[0] / gpts[0] * waves.angular_sampling[0],
                                gpts_old[1] / gpts[1] * waves.angular_sampling[1])

        sampling = (1 / new_angular_sampling[0] / gpts[0] * waves.wavelength * 1000,
                    1 / new_angular_sampling[1] / gpts[1] * waves.wavelength * 1000)

        calibrations = abtem.detect.calibrations_from_grid(gpts,
                                              sampling,
                                              names=['alpha_x', 'alpha_y'],
                                              units='mrad',
                                              scale_factor=waves.wavelength * 1000,
                                              fourier_space=True)

        if scan is None:
            scan_shape = ()
            scan_calibrations = ()
        elif isinstance(scan, tuple):
            scan_shape = scan
            scan_calibrations = (None,) * len(scan)
        else:
            scan_shape = scan.shape
            scan_calibrations = scan.calibrations

        if self._mode == 'intensity':
            array = np.zeros(scan_shape + gpts, dtype=np.float32)
        elif self._mode == 'complex':
            array = np.zeros(scan_shape + gpts, dtype=np.complex64)
        else:
            raise ValueError()

        measurement = abtem.Measurement(array, calibrations=scan_calibrations + calibrations)
        if isinstance(self.save_file, str):
            measurement = measurement.write(self.save_file)
        return measurement

    def detect(self, waves) -> np.ndarray:
        """
        Calculate the far field intensity of the wave functions and average it into the binned grid. The
        output is cropped to include the non-suppressed frequencies from the antialiased 2D fourier
        spectrum.

        Parameters
        ----------
        waves: Waves object
            The batch of wave functions to detect.

        Returns
        -------
            Detected values. The first dimension indexes the batch size, the second and third indexes the
            two components of the spatial frequency.
        """

        xp = abtem.device.get_array_module(waves.array)
        abs2 = abtem.device.get_device_function(xp, 'abs2')

        waves = waves.far_field(max_angle=self.max_angle)

        if self._mode == 'intensity':
            array = abs2(waves.array)
        elif self._mode == 'complex':
            array = waves.array
        else:
            raise ValueError()

        array = xp.fft.fftshift(array, axes=(-2, -1))

        # Mean-bin the y axis (last), then the x axis (via transposes)
        gpts_old = array.shape[-2:]
        gpts_new = self._binned_gpts(gpts_old)
        array = _mean_bin_axis(array, gpts_old[1], gpts_new[1], xp)
        array = _mean_bin_axis(array.swapaxes(-2, -1), gpts_old[0], gpts_new[0], xp).swapaxes(-2, -1)

        return array