__all__ = ["Converter", "Subref", "get_converter"]


# standard library
from dataclasses import dataclass
from logging import getLogger


# dependencies
import xarray as xr


# constants
LOGGER = getLogger(__name__)


@dataclass(frozen=True)
class Subref:
    """Estimated subreflector parameters of the Nobeyama 45m telescope.

    Args:
        dX: Estimated offset (in mm) from the X cylinder position
            optimized for the gravity deformation correction.
        dZ: Estimated offset (in mm) from the Z cylinder positions (Z1 = Z2)
            optimized for the gravity deformation correction.
        m0: Estimated expansion coefficient (in mm) of the Zernike polynomial Z(2, 0)
        m1: Estimated expansion coefficient (in mm) of the Zernike polynomial Z(1, -1).

    """

    dX: float
    dZ: float
    m0: float
    m1: float


@dataclass
class Converter:
    """EPL-to-subref parameter converter for the Nobeyama 45m telescope..

    Args:
        gain_dX: Propotional gain for the estimated dX.
        gain_dZ: Propotional gain for the estimated dZ.
        last: Last estimated subreflector parameters.

    """

    gain_dX: float = 0.1
    gain_dZ: float = 0.1
    last: Subref = Subref(dX=0.0, dZ=0.0, m0=0.0, m1=0.0)

    def __call__(self, epl: xr.DataArray, epl_cal: xr.DataArray, /) -> Subref:
        """Convert EPL to subreflector parameters.

        Args:
            epl: EPL to be converted (feed; in m)
                with the telescope state information at that time.
            epl_cal: EPL at calibration (feed; in m; must be zero)
                with the telescope state information at that time.

        Returns:
            Estimated subreflector parameters.

        """
        # current = Subref(dX=..., dZ=..., m0=..., m1=...)

        # if not condition_1:
        #     return self.on_failure()

        # if not condition_2:
        #     return self.on_failure()

        # if not condition_3:
        #     return self.on_failure()

        # return self.on_success(current)

    def on_success(self, estimated: Subref, /) -> Subref:
        """Return the estimated subreflector parameters and update the last."""
        self.last = estimated
        return estimated

    def on_failure(self) -> Subref:
        """Return the last subreflector parameters without updating."""
        return self.last


def get_converter(gain_dX: float = 0.1, gain_dZ: float = 0.1, /) -> Converter:
    """Get an EPL-to-subref parameter converter for the Nobeyama 45m telescope.

    Args:
        gain_dX: Propotional gain for the estimated dX.
        gain_dZ: Propotional gain for the estimated dZ.

    Returns:
        EPL-to-subref parameter converter.

    """
    return Converter(gain_dX=gain_dX, gain_dZ=gain_dZ)
