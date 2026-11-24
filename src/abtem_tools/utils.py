from skimage.color import lab2rgb
from matplotlib.colors import Normalize
import warnings

__all__ = ["complex_to_Lab", "get_N_colors"]

# make nicely spaced colors:
def complex_to_Lab(
    im, amin=None, amax=None, gamma=1.0, L_scale=100, ab_scale=64, uniform_L=None
):
    Lab = np.zeros(im.shape + (3,), dtype=np.float64)
    angle = np.angle(im)

    L = Normalize(vmin=amin, vmax=amax, clip=True)(np.abs(im)) ** gamma
    L = Normalize()(L)

    # attempt at polynomial saturation
    # ab_prescale = 4*L - 4*L*L
    ab_prescale = 0.5

    Lab[..., 0] = uniform_L or L * L_scale
    Lab[..., 1] = np.cos(angle) * ab_scale * ab_prescale
    Lab[..., 2] = np.sin(angle) * ab_scale * ab_prescale

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rgb = lab2rgb(Lab)

    return rgb

def get_N_colors(N):
    ar = np.exp(2.0j*np.pi*np.arange(N)/N)
    return np.squeeze(complex_to_Lab(
        np.atleast_2d(ar),
        ab_scale=256,
        uniform_L=64,
    ))