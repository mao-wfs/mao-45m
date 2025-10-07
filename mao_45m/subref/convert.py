__all__ = [
    "Converter",
    "get_converter",
    "get_epl_offsets",
    "get_homologous_epl",
    "get_integral_gain",
    "get_measurement_matrix",
    "get_proportional_gain",
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
from ..cosmos import ABSMAX_DX, ABSMAX_DZ
from ..utils import to_timedelta


# constants
LOGGER = getLogger(__name__)
SECOND = np.timedelta64(1, "s")


@dataclass
class Converter:
    """EPL-to-subref control converter for the Nobeyama 45m telescope..

    Args:
        G: Homologous EPL (G; feed x elevation; in m).
        K_I: Integral gain (K_I; feed).
        K_P: Proportional gain (K_I; feed).
        M: Measurement matrix (M; feed x drive).
        control_period: Control period (float in s or string with units).
        epl_interval_tolerance: Acceptable fraction of EPL time interval
            relative to the control period (0.1 means +/- 10% allowance).
        range_ddX: Absolute range for ddX (in m).
        range_ddZ: Absolute range for ddZ (in m).
        last: Last estimated subreflector parameters.

    """

    G: xr.DataArray
    M: xr.DataArray
    K_I: xr.DataArray
    K_P: xr.DataArray
    control_period: np.timedelta64 | str | float = "0.5 s"
    epl_interval_tolerance: float = 0.1
    range_ddX: tuple[float, float] = (0.00005, 0.000375)  # m
    range_ddZ: tuple[float, float] = (0.00005, 0.000300)  # m
    last: xr.DataArray | None = None

    @cached_property
    def inv_MTM_MT(self) -> xr.DataArray:
        """Pre-calculated (M^T M)^-1 M^T (drive x feed)."""
        M_ = self.M.rename(drive="drive_")
        return get_inv(M_ @ self.M) @ M_.T

    def __call__(
        self,
        epl: xr.DataArray,
        epl_cal: xr.DataArray,
        /,
        *,
        epl_offset: xr.DataArray | None,
    ) -> xr.DataArray:
        """Convert EPL to subreflector control (u; drive; in m).

        Args:
            epl: EPL to be converted (feed; in m)
                with the telescope state information at that time.
            epl_cal: EPL at calibration (feed; in m; must be zero)
                with the telescope state information at that time.
            epl_offset: EPL offset to be added to the EPL (feed; in m).

        Returns:
            Estimated subreflector control.

        """
        depl: xr.DataArray = (
            epl
            - epl_cal.data
            - self.G.interp(elevation=epl.elevation.data)
            + self.G.interp(elevation=epl_cal.elevation.data)
        )

        LOGGER.info(
            "Raw".ljust(10)
            + ", ".join(
                f"EPL({feed})={1e3 * epl.sel(feed=feed):+.3f}mm"
                for feed in epl.feed.data
            )
        )
        LOGGER.info(
            "Corrected".ljust(10)
            + ", ".join(
                f"EPL({feed})={1e3 * depl.sel(feed=feed):+.3f}mm"
                for feed in depl.feed.data
            )
        )

        if epl_offset is None:
            m = self.inv_MTM_MT @ depl
        else:
            m = self.inv_MTM_MT @ (depl + epl_offset.drop_vars("time"))

            LOGGER.info(
                "Offset".ljust(10)
                + ", ".join(
                    f"EPL({feed})={1e3 * epl_offset.sel(feed=feed):+.3f}mm"
                    for feed in epl_offset.feed.data
                )
            )

        tc = float(to_timedelta(self.control_period) / SECOND)

        if self.last is None:
            if (self.K_I == 0).all():
                u: xr.DataArray = (
                    (-self.K_P * m)
                    .assign_coords(m=m.assign_attrs(units="m"))
                    .assign_attrs(units="m")
                    .rename("u")
                )

            else:
                u: xr.DataArray = (
                    (-self.K_I * tc * m - self.K_P * m)
                    .assign_coords(m=m.assign_attrs(units="m"))
                    .assign_attrs(units="m")
                    .rename("u")
                )

            if abs(dX := float(u.sel(drive="X"))) > ABSMAX_DX:
                LOGGER.warning(f"{dX=} is out of range (|dX| <= {ABSMAX_DX}).")
                return self.on_failure(u)

            if abs(dZ := float(u.sel(drive="Z"))) > ABSMAX_DZ:
                LOGGER.warning(f"{dZ=} is out of range (|dZ| <= {ABSMAX_DZ}).")
                return self.on_failure(u)

            return self.on_success(u)

        if (self.K_I == 0).all():
            u: xr.DataArray = (
                (-self.K_P * m)
                .assign_coords(m=m.assign_attrs(units="m"))
                .assign_attrs(units="m")
                .rename("u")
            )

        else:
            u: xr.DataArray = (
                (self.last - self.K_I * tc * m - self.K_P * (m - self.last.m))
                .assign_coords(m=m.assign_attrs(units="m"))
                .assign_attrs(units="m")
                .rename("u")
            )
        dt = (u.time - self.last.time) / SECOND

        if abs(dX := float(u.sel(drive="X"))) > ABSMAX_DX:
            LOGGER.warning(f"{dX=} is out of range (|dX| <= {ABSMAX_DX}).")
            return self.on_failure(u)

        if abs(dZ := float(u.sel(drive="Z"))) > ABSMAX_DZ:
            LOGGER.warning(f"{dZ=} is out of range (|dZ| <= {ABSMAX_DZ}).")
            return self.on_failure(u)

        if abs(ddX := dX - float(self.last.sel(drive="X"))) < self.range_ddX[0]:
            LOGGER.warning(f"{ddX=} is out of range (|ddX| >= {self.range_ddX[0]}).")
            return self.on_failure(u)

        if abs(ddX) > self.range_ddX[1]:
            LOGGER.warning(f"{ddX=} is out of range (|ddX| <= {self.range_ddX[1]}).")
            return self.on_failure(u)

        if abs(ddZ := dZ - float(self.last.sel(drive="Z"))) < self.range_ddZ[0]:
            LOGGER.warning(f"{ddZ=} is out of range (|ddZ| >= {self.range_ddZ[0]}).")
            return self.on_failure(u)

        if abs(ddZ) > self.range_ddZ[1]:
            LOGGER.warning(f"{ddZ=} is out of range (|ddZ| <= {self.range_ddZ[1]}).")
            return self.on_failure(u)

        if dt < (dt_min := tc * (1 - self.epl_interval_tolerance)):
            LOGGER.warning(f"{dt=} is out of range (dt >= {dt_min}).")
            return self.on_failure(u)

        if dt > (dt_max := tc * (1 + self.epl_interval_tolerance)):
            LOGGER.warning(f"{dt=} is out of range (dt <= {dt_max}).")
            return self.on_failure(u)

        return self.on_success(u)

    def on_success(self, current: xr.DataArray, /) -> xr.DataArray:
        """Replace the last subreflector control with current one."""
        LOGGER.info(
            ", ".join(
                f"u({drive})={1e3 * current.sel(drive=drive):+.3f}mm"
                for drive in current.drive.data
            )
        )
        LOGGER.info(
            ", ".join(
                f"m({drive})={1e3 * current.m.sel(drive=drive):+.3f}mm"
                for drive in current.drive.data
            )
        )
        self.last = current
        return current

    def on_failure(self, current: xr.DataArray, /) -> xr.DataArray:
        """Replace the last subreflector control's time with current one."""
        if self.last is None:
            m_zero = xr.zeros_like(current.m)
            u_zero = xr.zeros_like(current)
            return self.on_success(u_zero.assign_coords(m=m_zero))
        else:
            return self.on_success(self.last.assign_coords(time=current.time))


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
    """Get an EPL-to-subref control converter for the Nobeyama 45m telescope.

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
        EPL-to-subref control converter.

    """
    return Converter(
        G=get_homologous_epl(feed_model),
        M=get_measurement_matrix(feed_model),
        K_I=get_integral_gain(integral_gain_dX, integral_gain_dZ),
        K_P=get_proportional_gain(proportional_gain_dX, proportional_gain_dZ),
        control_period=control_period,
        epl_interval_tolerance=epl_interval_tolerance,
        range_ddX=range_ddX,
        range_ddZ=range_ddZ,
    )


def get_epl_offsets(epl_offsets: PathLike[str] | str, /) -> xr.DataArray:
    """Get the EPL offsets (time x feed; in m) from given CSV file."""
    df = pd.read_csv(epl_offsets, comment="#", index_col=0, skipinitialspace=True)
    df = df.set_index(pd.to_timedelta(df.index, unit="s"))

    return (
        df.to_xarray()
        .to_dataarray(dim="feed")
        .assign_attrs(units="m")
        .rename("epl_offsets")
        .T
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


def get_integral_gain(dX: float, dZ: float, /) -> xr.DataArray:
    """Get the integral gain (K_I; feed) from given values."""
    return xr.DataArray(
        data=[dX, dZ],
        dims="drive",
        coords={"drive": ["X", "Z"]},
        name="K_I",
    )


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


def get_proportional_gain(dX: float, dZ: float, /) -> xr.DataArray:
    """Get the proportional gain (K_P; feed) from given values."""
    return xr.DataArray(
        data=[dX, dZ],
        dims="drive",
        coords={"drive": ["X", "Z"]},
        name="K_P",
    )
