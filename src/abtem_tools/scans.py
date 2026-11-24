import abtem
import numpy as np

__all__ = ["RotatedGridScan", "NoisyGridScan", "DriftingGridScan"]


class RotatedGridScan(abtem.GridScan):
    def __init__(self, *args, rotation=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rotation = rotation

    def get_positions(self):
        x = np.linspace(
            self.start[0], self.end[0], self.gpts[0], endpoint=self.grid.endpoint[0]
        )
        y = np.linspace(
            self.start[1], self.end[1], self.gpts[1], endpoint=self.grid.endpoint[1]
        )
        x, y = np.meshgrid(x, y, indexing="ij")

        c_x = (self.end[0] + self.start[0]) / 2
        c_y = (self.end[1] + self.start[1]) / 2

        xc = x - c_x
        yc = y - c_y

        xr = np.cos(self.rotation) * xc - np.sin(self.rotation) * yc
        yr = np.sin(self.rotation) * xc + np.cos(self.rotation) * yc

        x = xr + c_x
        y = yr + c_y

        return np.stack((np.reshape(x, (-1,)), np.reshape(y, (-1,))), axis=1)

    def add_to_mpl_plot(self, *args, **kwargs):
        return abtem.scan.PositionScan.add_to_mpl_plot(self, *args, **kwargs)


class NoisyGridScan(abtem.GridScan):
    def __init__(self, *args, sigma=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma = sigma / np.sqrt(2) # scale so the magnitude is averaged to sigma, not the components

        x = np.linspace(self.start[0], self.end[0], self.gpts[0], endpoint=self.grid.endpoint[0])
        y = np.linspace(self.start[1], self.end[1], self.gpts[1], endpoint=self.grid.endpoint[1])
        x, y = np.meshgrid(x, y, indexing='ij')

        x += np.random.normal(loc=0.0, scale=self.sigma, size=x.shape)
        y += np.random.normal(loc=0.0, scale=self.sigma, size=y.shape)

        self._xy = x,y

    def get_positions(self):
        x, y = self._xy
        
        return np.stack((np.reshape(x, (-1,)),
                         np.reshape(y, (-1,))), axis=1)

    def add_to_mpl_plot(self, *args, **kwargs):
        return abtem.scan.PositionScan.add_to_mpl_plot(self, *args, **kwargs)


class DriftingGridScan(abtem.GridScan):
    def __init__(self, *args, dwell_time=100e-6, drift_velocity=(0.15, 0.10), drift_noise=(0.0,0.0), **kwargs):
        """
        GridScan with sample drift. dwell_time is in [s]; drift_velocity and drift_noise are in [A/s].
        Typical values for different systems can be found in Goodge et al Microscopy and Microanalysis 26(3):439-446 2020
        """
        super().__init__(*args, **kwargs)

        x = np.linspace(self.start[0], self.end[0], self.gpts[0], endpoint=self.grid.endpoint[0])
        y = np.linspace(self.start[1], self.end[1], self.gpts[1], endpoint=self.grid.endpoint[1])
        x, y = np.meshgrid(x, y, indexing='ij')

        instant_velocity_x = np.random.normal(loc=drift_velocity[0], scale=drift_noise[0], size=x.shape)
        instant_velocity_y = np.random.normal(loc=drift_velocity[1], scale=drift_noise[1], size=y.shape)

        drift_x = np.cumsum(dwell_time * instant_velocity_x.ravel()).reshape(instant_velocity_x.shape)
        drift_y = np.cumsum(dwell_time * instant_velocity_y.ravel()).reshape(instant_velocity_y.shape)

        x += drift_x
        y += drift_y

        self._xy = x,y

    def get_positions(self):
        x, y = self._xy
        
        return np.stack((np.reshape(x, (-1,)),
                         np.reshape(y, (-1,))), axis=1)

    def add_to_mpl_plot(self, *args, **kwargs):
        return abtem.scan.PositionScan.add_to_mpl_plot(self, *args, **kwargs)
