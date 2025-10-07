__all__ = [
    "Converter",
    "get_converter",
    "get_homologous_epl",
    "get_integral_gain",
    "get_measurement_matrix",
    "get_proportional_gain",
    "get_anti_windup_gain",
    "saturate_subref_control",
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
        K_a: Anti-windup gain (K_a; feed).
        M: Measurement matrix (M; feed x drive).
        control_period: Control period (float in s or string with units).
        epl_interval_tolerance: Acceptable fraction of EPL time interval
            relative to the control period (0.1 means +/- 10% allowance).
        range_dX: Absolute range for dX (in m).
        range_dZ: Absolute range for dZ (in m).
        range_ddX: Absolute range for ddX (in m).
        range_ddZ: Absolute range for ddZ (in m).
        last: Last estimated subreflector parameters.

    """

    G: xr.DataArray
    M: xr.DataArray
    K_I: xr.DataArray
    K_P: xr.DataArray
    K_a: xr.DataArray
    control_period: np.timedelta64 | str | float = "0.5 s"
    epl_interval_tolerance: float = 0.1
    range_dX: tuple[float, float] = (-0.038, 0.038)  # m
    range_dZ: tuple[float, float] = (-0.019, 0.019)  # m
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
        epl_offset: xr.DataArray | None,
        /,
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

        if epl_offset is None:
            m = self.inv_MTM_MT @ depl
        else:
            m = self.inv_MTM_MT @ (depl + epl_offset)
        tc = float(to_timedelta(self.control_period) / SECOND)

        if self.last is None:
            if (self.K_I == 0).all():  # P control
                u: xr.DataArray = (
                    (-self.K_P * m)
                    .assign_coords(m=m.assign_attrs(units="m"))
                    .assign_attrs(units="m")
                    .rename("u")
                )

            else:  # PI control with anti-windup
                v: xr.DataArray = -tc * m
                u_tmp: xr.DataArray = self.K_I * v - self.K_P * m
                u: xr.DataArray = (
                    (saturate_subref_control(u_tmp, self.range_dX, self.range_dZ))
                    .assign_coords(
                        {
                            "m": m.assign_attrs(units="m"),
                            "v": v.assign_attrs(units="m*s"),
                            "u_tmp": u_tmp.assign_attrs(units="m"),
                        }
                    )
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

        if (self.K_I == 0).all():  # P control
            u: xr.DataArray = (
                (-self.K_P * m)
                .assign_coords(m=m.assign_attrs(units="m"))
                .assign_attrs(units="m")
                .rename("u")
            )

        else:  # PI control with anti-windup
            v: xr.DataArray = (
                self.last.v - tc * m - tc * self.K_a * (self.last - self.last.u_tmp)
            )
            u_tmp = self.K_I * v - self.K_P * m
            u: xr.DataArray = (
                (saturate_subref_control(u_tmp, self.range_dX, self.range_dZ))
                .assign_coords(
                    {
                        "m": m.assign_attrs(units="m"),
                        "v": v.assign_attrs(units="m*s"),
                        "u_tmp": u_tmp.assign_attrs(units="m"),
                    }
                )
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
        self.last = current
        return current

    def on_failure(self, current: xr.DataArray, /) -> xr.DataArray:
        """Replace the last subreflector control's time with current one."""
        if self.last is None:
            if (self.K_I == 0).all():  # P control
                m_zero = xr.zeros_like(current.m)
                u_zero = xr.zeros_like(current)
                self.last = u_zero.assign_coords(m=m_zero)
            else:  # PI control with anti-windup
                m_zero = xr.zeros_like(current.m)
                v_zero = xr.zeros_like(current.v)
                u_tmp_zero = xr.zeros_like(current.u_tmp)
                u_zero = xr.zeros_like(current)
                self.last = u_zero.assign_coords(
                    {"m": m_zero, "v": v_zero, "u_tmp": u_tmp_zero}
                )
        else:
            self.last = self.last.assign_coords(time=current.time)

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
    anti_windup_gain_gain_dX: float = 10,
    anti_windup_gain_gain_dZ: float = 10,
    range_dX: tuple[float, float] = (-0.038, 0.038),  # m
    range_dZ: tuple[float, float] = (-0.019, 0.019),  # m
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
        anti_windup_gain_gain_dX: Anti-windup gain for the estimated dX.
        anti_windup_gain_gain_dZ: Anti-windup gain for the estimated dZ.
        range_dX: Absolute range for dX (in m).
        range_dZ: Absolute range for dZ (in m).
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
        K_a=get_anti_windup_gain(anti_windup_gain_gain_dX, anti_windup_gain_gain_dZ),
        control_period=control_period,
        epl_interval_tolerance=epl_interval_tolerance,
        range_dX=range_dX,
        range_dZ=range_dZ,
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


def get_anti_windup_gain(dX: float, dZ: float, /) -> xr.DataArray:
    """Get the anti-windup gain (K_a; feed) from given values."""
    return xr.DataArray(
        data=[dX, dZ],
        dims="drive",
        coords={"drive": ["X", "Z"]},
        name="K_a",
    )


def saturate_subref_control(
    u_tmp: xr.DataArray, range_dX: tuple[float, float], range_dZ: tuple[float, float]
) -> xr.DataArray:
    """Saturation function to limit the subreflector control within the range."""
    u = u_tmp.copy()
    if float(u.sel(drive="X")) < range_dX[0]:
        u.loc[dict(drive="X")] = range_dX[0]

    if float(u.sel(drive="X")) > range_dX[1]:
        u.loc[dict(drive="X")] = range_dX[1]

    if float(u.sel(drive="Z")) < range_dZ[0]:
        u.loc[dict(drive="Z")] = range_dZ[0]

    if float(u.sel(drive="Z")) > range_dZ[1]:
        u.loc[dict(drive="Z")] = range_dZ[1]

    return u
