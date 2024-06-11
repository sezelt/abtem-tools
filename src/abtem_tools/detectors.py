import abtem
import numpy as np

__all__ = ["BinnedPixelatedDetector"]

class BinnedPixelatedDetector(abtem.PixelatedDetector):
    """
    just like abtem.PixelatedDetector but also accepts an extra argument for binning of the patterns
    binfactor: set binning level
    """
    def __init__(self, *args, **kwargs):
        self._binning = kwargs.pop('binfactor',1)
        super().__init__(*args, **kwargs)

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

        gpts = waves.downsampled_gpts(self.max_angle)
        # print(f"Old gpts: {gpts}")
        # gpts, new_angular_sampling = self._resampled_gpts(gpts, angular_sampling=waves.angular_sampling)
        gpts = gpts[0] // self._binning, gpts[1]//self._binning
        new_angular_sampling = waves.angular_sampling[0] * self._binning, waves.angular_sampling[1] * self._binning
        # print(f"New gpts: {gpts}")

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

    def bin2D(self, array, factor, xp, dtype=np.float32):
        """
        Bin a 2D ndarray by binfactor.
    
        Args:
            array (2D numpy array):
            factor (int): the binning factor
            dtype (numpy dtype): datatype for binned array. default is numpy default for
                np.zeros()
    
        Returns:
            the binned array
        """
        x, y = array.shape[-2:]
        binx, biny = x // factor, y // factor
        xx, yy = binx * factor, biny * factor
    
        # Make a binned array on the device
        binned_ar = xp.zeros((array.shape[0], binx, biny), dtype=dtype)
        array = array.astype(dtype)
    
        # Collect pixel sums into new bins
        for ix in range(factor):
            for iy in range(factor):
                binned_ar += array[:, 0 + ix:xx + ix:factor, 0 + iy:yy + iy:factor]
        return binned_ar

    def detect(self, waves) -> np.ndarray:
        """
        Calculate the far field intensity of the wave functions. The output is cropped to include the non-suppressed
        frequencies from the antialiased 2D fourier spectrum.

        Parameters
        ----------
        waves: Waves object
            The batch of wave functions to detect.

        Returns
        -------
            Detected values. The first dimension indexes the batch size, the second and third indexes the two components
            of the spatial frequency.
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

        # bin the array
        array = self.bin2D(array, self._binning, xp, dtype=array.dtype)
        
        return array