import abtem
import numpy as np

__all__ = ["RotatedGridScan"]

class RotatedGridScan(abtem.GridScan):
    def __init__(self, *args, rotation=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rotation = rotation

    def get_positions(self):
        x = np.linspace(self.start[0], self.end[0], self.gpts[0], endpoint=self.grid.endpoint[0])
        y = np.linspace(self.start[1], self.end[1], self.gpts[1], endpoint=self.grid.endpoint[1])
        x, y = np.meshgrid(x, y, indexing='ij')

        c_x = (self.end[0] + self.start[0]) / 2
        c_y = (self.end[1] + self.start[1]) / 2

        xc = x - c_x
        yc = y - c_y

        xr = np.cos(self.rotation) * xc - np.sin(self.rotation) * yc
        yr = np.sin(self.rotation) * xc + np.cos(self.rotation) * yc

        x = xr + c_x
        y = yr + c_y
        
        return np.stack((np.reshape(x, (-1,)),
                         np.reshape(y, (-1,))), axis=1)

    def add_to_mpl_plot(self, *args, **kwargs):
        return abtem.scan.PositionScan.add_to_mpl_plot(self, *args, **kwargs)