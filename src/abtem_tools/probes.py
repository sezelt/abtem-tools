import abtem
from typing import List, Sequence, Union

__all__ = ["PositionAwareProbe"]


class PositionAwareProbe(abtem.Probe):
    """
    A probe that passes the probe positions of each detected batch to the detectors.

    This probe behaves identically to abtem.Probe except that, for each batch of exit waves, the probe
    positions of the waves in the batch are passed to every detector that defines a ``detect_at(waves,
    positions)`` method. Detectors that use the positions (such as ConfocalDetector with ``center='probe'``)
    can then place each detected image on the correct probe position.

    Detectors that do not define ``detect_at`` are detected with the usual ``detect`` method, so this probe
    can be used with a mixed list of detectors.
    """

    def scan(self,
             scan: abtem.scan.AbstractScan,
             detectors: Union[abtem.detect.AbstractDetector, Sequence[abtem.detect.AbstractDetector]],
             potential: abtem.potentials.AbstractPotential,
             measurements: Union[abtem.Measurement, dict] = None,
             max_batch: int = None,
             pbar: bool = True) -> Union[abtem.Measurement, List[abtem.Measurement]]:
        """
        Raster scan the probe across the potential and record a measurement for each detector, passing the
        probe positions of each batch to any detector that defines a ``detect_at`` method.
        """

        self.grid.match(potential.grid)
        self.grid.check_is_defined()

        detectors = self._validate_detectors(detectors)
        measurements = self.validate_scan_measurements(detectors, scan, measurements)

        positions = scan.get_positions()

        for indices, exit_probes in self.generate_probes(scan, potential, max_batch, pbar):
            for detector, measurement in measurements.items():
                if hasattr(detector, 'detect_at'):
                    new_entries = detector.detect_at(exit_probes, positions[indices])
                else:
                    new_entries = detector.detect(exit_probes)
                new_entries = new_entries / potential.num_frozen_phonon_configs
                scan.insert_new_measurement(measurement, indices, new_entries)

        measurements = list(measurements.values())
        if len(measurements) == 1:
            return measurements[0]
        else:
            return measurements
