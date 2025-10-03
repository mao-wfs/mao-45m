__all__ = [
    "Converter",
    "Subref",
    "get_converter",
    "get_homologous_epl",
    "get_measurement_matrix",
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
from typing_extensions import Self


# constants
LOGGER = getLogger(__name__)
NRO45M_DIAMETER = 45.0  # m
SOFTWARE_LIMIT_DZ = 0.049  # m
SOFTWARE_LIMIT_DX = 0.118  # m


@dataclass(frozen=True)
class Subref:
    """Estimated subreflector parameters of the Nobeyama 45m telescope.

    Args:
        dX: Estimated offset (in m) from the X cylinder position
            optimized for the gravity deformation correction.
        dZ: Estimated offset (in m) from the Z cylinder positions (Z1 = Z2)
            optimized for the gravity deformation correction.
        m0: 展開係数のX軸方向成分.
        m1: 展開係数のZ軸方向成分.
        time: EPL推定の最終時刻.

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
        proportional_gain_dX: Propotional gain for the estimated dX.
        proportional_gain_dZ: Propotional gain for the estimated dZ.
        integral_gain_dX: Integral gain for the estimated dX.
        integral_gain_dZ: Integral gain for the estimated dZ.
        range_ddX: Absolute range for ddX (in m).
        range_ddZ: Absolute range for ddZ (in m).
        Tc: Time constant (in s).
        time_threshold: 現在と直前ループのEPL推定最終時刻の時刻差判定で使用する閾値.
        last: Last estimated subreflector parameters.

    """

    G: xr.DataArray
    M: xr.DataArray
    proportional_gain_dX: float = 0.1
    proportional_gain_dZ: float = 0.1
    integral_gain_dX: float = 0.1
    integral_gain_dZ: float = 0.1
    range_ddX: tuple[float, float] = (0.00005, 0.000375)  # m
    range_ddZ: tuple[float, float] = (0.00005, 0.000300)  # m
    Tc: float = 0.250  # s
    time_threshold: float = 0.5  # s (適当)
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
        if (self.last.time != None) and (epl.time - self.last.time) / np.timedelta64(
            1, "s"
        ) >= self.time_threshold:
            LOGGER.warning(f"Time difference exceeds threshold.")
            return self.on_failure(epl)  # 異常発生時のEPL時刻

        depl = (
            epl
            - epl_cal
            - self.G.interp(elevation=epl.elevation)
            + self.G.interp(elevation=epl_cal.elevation)
        )
        m = self.inv_MTM_MT @ depl

        current = Subref(
            dX=self.last.dX
            - self.integral_gain_dX * self.Tc * float(m.sel(drive="X"))
            - self.proportional_gain_dX * (float(m.sel(drive="X")) - self.last.m0),
            dZ=self.last.dZ
            - self.integral_gain_dZ * self.Tc * float(m.sel(drive="Z"))
            - self.proportional_gain_dZ * (float(m.sel(drive="Z")) - self.last.m1),
            m0=float(m.sel(drive="X")),
            m1=float(m.sel(drive="Z")),
            time=epl.time.values,  # epl.timeがnp.datetime64 (UTC)だと仮定
        )

        if not (
            -SOFTWARE_LIMIT_DX < current.dX < SOFTWARE_LIMIT_DX
            or -SOFTWARE_LIMIT_DZ < current.dZ < SOFTWARE_LIMIT_DZ
        ):
            LOGGER.warning(f"Software limit reached.")
            return self.on_failure(epl)

        if not (
            self.range_ddX[0] < np.abs(current.dX - self.last.dX) < self.range_ddX[1]
        ):
            return self.on_failure(epl)

        if not (
            self.range_ddZ[0] < np.abs(current.dZ - self.last.dZ) < self.range_ddZ[1]
        ):
            return self.on_failure(epl)

        return self.on_success(current)

    def on_success(self, estimated: Subref, /) -> Subref:
        """Return the estimated subreflector parameters and update the last."""
        self.last = estimated
        return estimated

    def on_failure(self, epl) -> Subref:
        """Return the last subreflector parameters without updating."""
        self.last = Subref(
            dX=self.last.dX,
            dZ=self.last.dZ,
            m0=self.last.m0,
            m1=self.last.m1,
            time=epl.time.values,
        )
        return self.last


def get_converter(
    feed_model: PathLike[str] | str,
    proportional_gain_dX: float = 0.1,
    proportional_gain_dZ: float = 0.1,
    integral_gain_dX: float = 0.1,
    integral_gain_dZ: float = 0.1,
    range_ddX: tuple[float, float] = (0.00005, 0.000375),  # m
    range_ddZ: tuple[float, float] = (0.00005, 0.000300),  # m
    Tc: float = 0.5,  # s
    time_threshold: float = 0.5,  # s (適当)
    /,
) -> Converter:
    """Get an EPL-to-subref parameter converter for the Nobeyama 45m telescope.

    Args:
        feed_model: Path to the feed model CSV file.
        proportional_gain_dX: Propotional gain for the estimated dX.
        proportional_gain_dZ: Propotional gain for the estimated dZ.
        integral_gain_dX: Integral gain for the estimated dX.
        integral_gain_dZ: Integral gain for the estimated dZ.
        range_ddX: Absolute range for ddX (in m).
        range_ddZ: Absolute range for ddZ (in m).
        Tc: Time constant (in s).
        time_threshold: 現在と直前ループのEPL推定最終時刻の時刻差判定で使用する閾値.

    Returns:
        EPL-to-subref parameter converter.

    """
    return Converter(
        G=get_homologous_epl(feed_model),
        M=get_measurement_matrix(feed_model),
        proportional_gain_dX=proportional_gain_dX,
        proportional_gain_dZ=proportional_gain_dZ,
        integral_gain_dX=integral_gain_dX,
        integral_gain_dZ=integral_gain_dZ,
        range_ddX=range_ddX,
        range_ddZ=range_ddZ,
        Tc=Tc,
        time_threshold=time_threshold,
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
