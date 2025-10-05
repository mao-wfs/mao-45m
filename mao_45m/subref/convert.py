__all__ = [
    "Converter",
    "Subref",
    "get_converter",
    "get_homologous_epl",
    "get_measurement_matrix",
]


# standard library
from dataclasses import dataclass, replace
from functools import cached_property
from logging import getLogger
from os import PathLike


# dependencies
import numpy as np
import pandas as pd
import xarray as xr
from ..cosmos import ABSMAX_DX, ABSMAX_DZ
from ..utils import to_timedelta


# constants
LOGGER = getLogger(__name__)
SECOND = np.timedelta64(1, "s")


@dataclass(frozen=True)
class Subref:
    """Estimated subreflector parameters of the Nobeyama 45m telescope.

    Args:
        dX: Estimated offset (in m) from the X cylinder position
            optimized for the gravity deformation correction.
        dZ: Estimated offset (in m) from the Z cylinder positions (Z1 = Z2)
            optimized for the gravity deformation correction.
        m0: Expansion coefficient in the X direction.
        m1: Expansion coefficient in the Z direction.
        time: Time (in UTC) of the estimated EPL.

    """

    dX: float
    dZ: float
    m0: float
    m1: float
    time: np.datetime64 | None


@dataclass
class Converter:
    """EPL-to-subref parameter converter for the Nobeyama 45m telescope..

    Args:
        G: Homologous EPL (G; feed x elevation; in m).
        M: Measurement matrix (M; feed x drive).
        control_period: Control period (float in s or string with units).
        epl_interval_tolerance: Acceptable fraction of EPL time interval
            relative to the control period (0.1 means +/- 10% allowance).
        integral_gain_dX: Integral gain for the estimated dX.
        integral_gain_dZ: Integral gain for the estimated dZ.
        proportional_gain_dX: Propotional gain for the estimated dX.
        proportional_gain_dZ: Propotional gain for the estimated dZ.
        range_ddX: Absolute range for ddX (in m).
        range_ddZ: Absolute range for ddZ (in m).
        last: Last estimated subreflector parameters.

    """

    G: xr.DataArray
    M: xr.DataArray
    control_period: np.timedelta64 | str | float = "0.5 s"
    epl_interval_tolerance: float = 0.1
    integral_gain_dX: float = 0.1
    integral_gain_dZ: float = 0.1
    proportional_gain_dX: float = 0.1
    proportional_gain_dZ: float = 0.1
    range_ddX: tuple[float, float] = (0.00005, 0.000375)  # m
    range_ddZ: tuple[float, float] = (0.00005, 0.000300)  # m
    last: Subref = Subref(dX=0.0, dZ=0.0, m0=0.0, m1=0.0, time=None)

    @cached_property
    def inv_MTM_MT(self) -> xr.DataArray:
        """Pre-calculated (M^T M)^-1 M^T (drive x feed)."""
        M_ = self.M.rename(drive="drive_")
        return get_inv(M_ @ self.M) @ M_.T

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
        m = self.inv_MTM_MT @ depl
        m0 = float(m.sel(drive="X"))
        m1 = float(m.sel(drive="Z"))
        tc = float(to_timedelta(self.control_period) / SECOND)

        dX = (
            self.last.dX
            - self.integral_gain_dX * tc * m0
            - self.proportional_gain_dX * (m0 - self.last.m0)
        )
        dZ = (
            self.last.dZ
            - self.integral_gain_dZ * tc * m1
            - self.proportional_gain_dZ * (m1 - self.last.m1)
        )
        current = Subref(dX=dX, dZ=dZ, m0=m0, m1=m1, time=epl.time.data)

        if abs(dX) > ABSMAX_DX:
            LOGGER.warning(f"{dX=} is out of range (|dX| <= {ABSMAX_DX}).")
            return self.on_failure(current)

        if abs(dZ) > ABSMAX_DZ:
            LOGGER.warning(f"{dZ=} is out of range (|dZ| <= {ABSMAX_DZ}).")
            return self.on_failure(current)

        if abs(ddX := dX - self.last.dX) < self.range_ddX[0]:
            LOGGER.warning(f"{ddX=} is out of range (|ddX| >= {self.range_ddX[0]}).")
            return self.on_failure(current)

        if abs(ddX) > self.range_ddX[1]:
            LOGGER.warning(f"{ddX=} is out of range (|ddX| <= {self.range_ddX[1]}).")
            return self.on_failure(current)

        if abs(ddZ := dZ - self.last.dZ) < self.range_ddZ[0]:
            LOGGER.warning(f"{ddZ=} is out of range (|ddZ| >= {self.range_ddZ[0]}).")
            return self.on_failure(current)

        if abs(ddZ) > self.range_ddZ[1]:
            LOGGER.warning(f"{ddZ=} is out of range (|ddZ| <= {self.range_ddZ[1]}).")
            return self.on_failure(current)

        if self.last.time is not None:
            dt = (current.time - self.last.time) / SECOND  # type: ignore

            if dt < (dt_min := tc * (1 - self.epl_interval_tolerance)):
                LOGGER.warning(f"{dt=} is out of range (dt >= {dt_min}).")
                return self.on_failure(current)

            if dt > (dt_max := tc * (1 + self.epl_interval_tolerance)):
                LOGGER.warning(f"{dt=} is out of range (dt <= {dt_max}).")
                return self.on_failure(current)

        return self.on_success(current)

    def on_success(self, current: Subref, /) -> Subref:
        """Replace the last subreflector parameters with current one."""
        self.last = current
        return self.last

    def on_failure(self, current: Subref, /) -> Subref:
        """Replace the time of the last subreflector parameters with current one."""
        self.last = replace(self.last, time=current.time)
        return self.last


def get_converter(
    *,
    control_period: np.timedelta64 | str | float = "0.5 s",
    epl_interval_tolerance: float = 0.1,
    feed_model: PathLike[str] | str,
    integral_gain_dX: float = 0.1,
    integral_gain_dZ: float = 0.1,
    proportional_gain_dX: float = 0.1,
    proportional_gain_dZ: float = 0.1,
    range_ddX: tuple[float, float] = (0.00005, 0.000375),  # m
    range_ddZ: tuple[float, float] = (0.00005, 0.000300),  # m
) -> Converter:
    """Get an EPL-to-subref parameter converter for the Nobeyama 45m telescope.

    Args:
        control_period: Control period (float in s or string with units).
        epl_interval_tolerance: Acceptable fraction of EPL time interval
            relative to the control period (0.1 means +/- 10% allowance).
        feed_model: Path to the feed model CSV file.
        integral_gain_dX: Integral gain for the estimated dX.
        integral_gain_dZ: Integral gain for the estimated dZ.
        proportional_gain_dX: Proportional gain for the estimated dX.
        proportional_gain_dZ: Proportional gain for the estimated dZ.
        range_ddX: Absolute range for ddX (in m).
        range_ddZ: Absolute range for ddZ (in m).

    Returns:
        EPL-to-subref parameter converter.

    """
    return Converter(
        G=get_homologous_epl(feed_model),
        M=get_measurement_matrix(feed_model),
        control_period=control_period,
        epl_interval_tolerance=epl_interval_tolerance,
        proportional_gain_dX=proportional_gain_dX,
        proportional_gain_dZ=proportional_gain_dZ,
        integral_gain_dX=integral_gain_dX,
        integral_gain_dZ=integral_gain_dZ,
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
