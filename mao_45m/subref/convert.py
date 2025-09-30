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
from functools import cached_property
from logging import getLogger
from os import PathLike


# dependencies
import numpy as np
import pandas as pd
import xarray as xr
from poppy.zernike import zernike
from typing_extensions import Self


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
        G: Homologous EPL (G; feed x elevation; in m).
        M: Measurement matrix (M; feed x drive).
        Z: Zernike polynomial matrix (Z; feed x drive).
        gain_dX: Propotional gain for the estimated dX.
        gain_dZ: Propotional gain for the estimated dZ.
        range_ddX: Absolute range for ddX (in m).
        range_ddZ: Absolute range for ddZ (in m).
        last: Last estimated subreflector parameters.

    """

    G: xr.DataArray
    M: xr.DataArray
    Z: xr.DataArray
    gain_dX: float = 0.1
    gain_dZ: float = 0.1
    range_ddX: tuple[float, float] = (0.00005, 0.000375)  # m
    range_ddZ: tuple[float, float] = (0.00005, 0.000300)  # m
    last: Subref = Subref(dX=0.0, dZ=0.0, m0=0.0, m1=0.0)

    @classmethod
    def from_feed_model(
        cls,
        feed_model: PathLike[str] | str,
        gain_dX: float = 0.1,
        gain_dZ: float = 0.1,
        range_ddX: tuple[float, float] = (0.00005, 0.000375),  # m
        range_ddZ: tuple[float, float] = (0.00005, 0.000300),  # m
        last: Subref = Subref(dX=0.0, dZ=0.0, m0=0.0, m1=0.0),
    ) -> Self:
        """Create an EPL-to-subref converter from given feed model.

        Args:
            feed_model: Path to the feed model CSV file.
            gain_dX: Propotional gain for the estimated dX.
            gain_dZ: Propotional gain for the estimated dZ.
            range_ddX: Absolute range for ddX (in m).
            range_ddZ: Absolute range for ddZ (in m).
            last: Last estimated subreflector parameters.

        """
        return cls(
            G=get_homologous_epl(feed_model),
            M=get_measurement_matrix(feed_model),
            Z=get_zernike_matrix(feed_model),
            gain_dX=gain_dX,
            gain_dZ=gain_dZ,
            range_ddX=range_ddX,
            range_ddZ=range_ddZ,
            last=last,
        )

    @cached_property
    def inv_ZTZ_ZT(self) -> xr.DataArray:
        """Pre-calculated (Z^T Z)^-1 Z^T (drive x feed)."""
        Z_ = self.Z.rename(drive="drive_")
        return get_inv(Z_ @ self.Z) @ Z_.T

    @cached_property
    def inv_ZTM_ZT(self) -> xr.DataArray:
        """Pre-calculated (Z^T M)^-1 Z^T (drive x feed)."""
        Z_ = self.Z.rename(drive="drive_")
        return get_inv(Z_ @ self.M) @ Z_.T

    def __call__(self, epl: xr.DataArray, epl_cal: xr.DataArray, /) -> Subref:
        """Convert EPL to subreflector parameters.

        Args:
            epl: EPL to be converted (in m; feed)
                with the telescope state information at that time.
            epl_cal: EPL at calibration (in m; feed; must be zero)
                with the telescope state information at that time.

        Returns:
            Estimated subreflector parameters.

        """
        depl = (
            epl
            - epl_cal
            - self.G.interp(elevation=epl.elevation)
            + self.G.interp(elevation=epl_cal.elevation)
        )
        m = self.inv_ZTZ_ZT @ depl
        d = self.inv_ZTM_ZT @ depl

        current = Subref(
            dX=-self.gain_dX * float(d.sel(drive="X")),
            dZ=-self.gain_dZ * float(d.sel(drive="Z")),
            m0=float(m.sel(drive="X")),
            m1=float(m.sel(drive="Z")),
        )

        return self.on_success(current)

    def on_success(self, estimated: Subref, /) -> Subref:
        """Return the estimated subreflector parameters and update the last."""
        self.last = estimated
        return estimated

    def on_failure(self) -> Subref:
        """Return the last subreflector parameters without updating."""
        return self.last


def get_converter(
    feed_model: PathLike[str] | str,
    gain_dX: float = 0.1,
    gain_dZ: float = 0.1,
    range_ddX: tuple[float, float] = (0.00005, 0.000375),  # m
    range_ddZ: tuple[float, float] = (0.00005, 0.000300),  # m
    /,
) -> Converter:
    """Get an EPL-to-subref parameter converter for the Nobeyama 45m telescope.

    Args:
        feed_model: Path to the feed model CSV file.
        gain_dX: Propotional gain for the estimated dX.
        gain_dZ: Propotional gain for the estimated dZ.
        range_ddX: Absolute range for ddX (in m).
        range_ddZ: Absolute range for ddZ (in m).

    Returns:
        EPL-to-subref parameter converter.

    """
    return Converter.from_feed_model(
        feed_model=feed_model,
        gain_dX=gain_dX,
        gain_dZ=gain_dZ,
        range_ddX=range_ddX,
        range_ddZ=range_ddZ,
    )


def get_homologous_epl(
    feed_model: PathLike[str] | str,
    /,
    *,
    elevation_step: float = 0.01,
) -> xr.DataArray:
    """Get the homologous EPL (G; feed x elevation; in m) from given feed model.

    Args:
        feed_model: Path to the feed model CSV file.
        elevation_step: Elevation step size (in deg) for calculation.

    Returns:
        Homologous EPL (G; feed x elevation; in m).

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
        return (a * np.sin(np.deg2rad(elevation - b)) + c).rename("G")


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
