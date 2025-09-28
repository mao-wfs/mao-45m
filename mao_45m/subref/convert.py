__all__ = [
    "Converter",
    "Subref",
    "get_converter",
    "get_homologous_epl",
    "get_measurement_matrix",
    "get_zernike_matrix",
]


# standard library
from dataclasses import dataclass
from logging import getLogger
from os import PathLike


# dependencies
import numpy as np
import pandas as pd
import xarray as xr
from poppy.zernike import zernike


# constants
LOGGER = getLogger(__name__)
NRO45M_DIAMETER = 45.0  # m


@dataclass(frozen=True)
class Subref:
    """Estimated subreflector parameters of the Nobeyama 45m telescope.

    Args:
        dX: Estimated offset (in m) from the X cylinder position
            optimized for the gravity deformation correction.
        dZ: Estimated offset (in m) from the Z cylinder positions (Z1 = Z2)
            optimized for the gravity deformation correction.
        m0: Estimated expansion coefficient of the Zernike polynomial Z(2, 0)
        m1: Estimated expansion coefficient of the Zernike polynomial Z(1, -1).

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

    def __call__(self, epl: xr.DataArray, epl_cal: xr.DataArray, /) -> Subref:  # type: ignore
        """Convert EPL to subreflector parameters.

        Args:
            epl: EPL to be converted (in m; feed)
                with the telescope state information at that time.
            epl_cal: EPL at calibration (in m; feed; must be zero)
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


def get_homologous_epl(
    feed_model: PathLike[str] | str,
    /,
    *,
    elevation_step: float = 0.01,
) -> xr.DataArray:
    """Get the homologous EPL (in m; feed x elevation) from given feed model.

    Args:
        feed_model: Path to the feed model CSV file.
        elevation_step: Elevation step size (in deg) for calculation.

    Returns:
        Homologous EPL (in m; feed x elevation).

    """
    df = pd.read_csv(feed_model, comment="#", index_col=0, skipinitialspace=True)

    a = xr.DataArray(
        df["homologous_EPL_A"],
        dims="feed",
        coords={"feed": df.index},
        attrs={"units": "m"},
    )
    b = xr.DataArray(
        df["homologous_EPL_B"],
        dims="feed",
        coords={"feed": df.index},
        attrs={"units": "deg"},
    )
    c = xr.DataArray(
        df["homologous_EPL_C"],
        dims="feed",
        coords={"feed": df.index},
        attrs={"units": "m"},
    )
    elevation = xr.DataArray(
        data := np.arange(0, 90.0 + elevation_step, elevation_step),
        dims="elevation",
        coords={"elevation": data},
        attrs={"units": "deg"},
    )

    with xr.set_options(keep_attrs=True):
        return a * np.cos(np.deg2rad(elevation + b)) + c


def get_inv(X: xr.DataArray, /) -> xr.DataArray:
    """Get the inverse of given two-dimensional DataArray."""
    return X.copy(data=np.linalg.inv(X.data.T)).T


def get_measurement_matrix(feed_model: PathLike[str] | str, /) -> xr.DataArray:
    """Get the measurement matrix (M; feed x drive) from given feed model.

    Args:
        feed_model: Path to the feed model CSV file.

    Returns:
        Measurement matrix (M; feed x drive).

    """
    df = pd.read_csv(feed_model, comment="#", index_col=0, skipinitialspace=True)

    return xr.DataArray(
        [df["EPL_over_dX"], df["EPL_over_dZ"]],
        dims=["drive", "feed"],
        coords={
            "drive": ["X", "Z"],
            "feed": df.index,
        },
        name="M",
    ).T


def get_zernike_matrix(feed_model: PathLike[str] | str, /) -> xr.DataArray:
    """Get the Zernike polynomial matrix (Z; feed x drive) from given feed model.

    Args:
        feed_model: Path to the feed model CSV file.

    Returns:
        Zernike polynomial matrix (Z; feed x drive).

    """
    df = pd.read_csv(feed_model, comment="#", index_col=0, skipinitialspace=True)
    rho = df["position_radius"] / (NRO45M_DIAMETER / 2)
    theta = np.deg2rad(df["position_angle"])

    return xr.DataArray(
        [
            zernike(1, -1, rho=rho, theta=theta, noll_normalize=False),
            zernike(2, 0, rho=rho, theta=theta, noll_normalize=False),
        ],
        dims=("drive", "feed"),
        coords={
            "drive": ["X", "Z"],
            "feed": df.index,
            "zernike": ("drive", ["1,-1", "2,0"]),
        },
        name="Z",
    ).T
