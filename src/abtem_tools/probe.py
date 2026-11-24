from abtem import Probe
from abtem.device import get_array_module, asnumpy
from abtem.base_classes import cached_method
from scipy.interpolate import RegularGridInterpolator

class ElectrostaticProbe(Probe):
    """
    Construct a Probe using an input potential array to define the probe phase.

    `projected_electrostatic_potential` is a 2D numpy array containing the projected electrostatic potential
        (in Volt-angstrom) of a phase plate, with np.nan indicating areas blocked by the aperture.
    `potential_array_semiangle_max` is the semiangle corresponding to the edges of the input
        array. To generate the probe, the input array will be interpolated on
        the grid of the Probe, and all areas outside the array are assumed blocked
    """
    def __init__(self, *args, projected_electrostatic_potential=None, potential_array_semiangle_max=None, **kwargs):
        super().__init__(*args, **kwargs)

        # convert potential into phase shift before storing
        self._phase_array = projected_electrostatic_potential * self._accelerator.sigma
        self._array_semiangle_max = potential_array_semiangle_max / 1000.

    @cached_method('_ctf_cache')
    def _evaluate_ctf(self):
        alpha, phi = self.get_scattering_angles()

        xp = get_array_module(alpha)

        # the Wave coordinates are scaled to the (-1,1) range for the interpolator
        # normalized by the semiangle corresponding to the input array so that
        # values outside the array are masked (blocked by aperture)
        qx, qy = alpha * xp.cos(phi) / self._array_semiangle_max, alpha * xp.sin(phi) / self._array_semiangle_max
        qx, qy = asnumpy(qx), asnumpy(qy)

        interp = RegularGridInterpolator(
            (np.linspace(-1,1,self._phase_array.shape[0]), np.linspace(-1,1,self._phase_array.shape[1])),
            self._phase_array,
            method='nearest',
            bounds_error=False,
            fill_value=np.nan,
        )
        phi_interp = interp(np.stack([qx,qy],axis=2))

        aperture = 1.0 - np.isnan(phi_interp)
        phase = np.nan_to_num(phi_interp)

        # Also add the phase (but not anything else!) from the base class CTF
        ctf_phase = asnumpy(self._ctf.evaluate_chi(alpha, phi))

        array = xp.array( aperture * np.exp(1.0j * (phase + ctf_phase)), dtype=xp.complex64 )

        return array