__all__ = ["control"]


# standard library
from collections.abc import Sequence
from logging import getLogger
from os import PathLike
from time import sleep


# dependencies
import numpy as np
from ndtools import Range
from tqdm import tqdm
from .convert import get_converter as get_subref_converter
from ..cosmos import get_cosmos
from ..epl.convert import get_aggregated, get_converter as get_epl_converter
from ..vdif import FRAMES_PER_SAMPLE
from ..vdif.convert import get_samples
from ..vdif.receive import get_frames
from ..utils import take, to_timedelta


# constants
LOGGER = getLogger(__name__)
SECOND = np.timedelta64(1, "s")


def control(
    *,
    # options for the feed information
    feed_model: PathLike[str] | str,
    feed_origin: str,
    feed_pattern: Sequence[str] | str,
    # options for the EPL estimates
    cal_interval: str | float = "10 s",
    freq_binning: int = 8,
    freq_range: tuple[float, float] = (19.5e9, 22.5e9),  # Hz
    integ_per_sample: str | float = "0.01 s",
    integ_per_epl: str | float = "0.5 s",
    # options for the subref control
    dry_run: bool = False,
    gain_dX: float = 0.1,
    gain_dZ: float = 0.1,
    range_ddX: tuple[float, float] = (0.00005, 0.000375),  # m
    range_ddZ: tuple[float, float] = (0.00005, 0.000300),  # m
    # options for network connection
    cosmos_host: str = "127.0.0.1",
    cosmos_port: int = 11111,
    cosmos_safe: bool = True,
    vdif_group: str = "239.0.0.1",
    vdif_port: int = 22222,
    # option for display and logging
    status: bool = True,
) -> None:
    """Control the subreflector of the Nobeyama 45m telescope by MAO."""
    # define the frame size for each EPL estimate
    dt_epl = to_timedelta(integ_per_epl)
    dt_sample = to_timedelta(integ_per_sample)
    frame_size = FRAMES_PER_SAMPLE * int(dt_epl / dt_sample)

    # create the EPL and subref converters
    get_epl = get_epl_converter(cal_interval)
    get_subref = get_subref_converter(
        feed_model,
        gain_dX,
        gain_dZ,
        range_ddX,
        range_ddZ,
    )

    with (
        tqdm(disable=not status, unit="EPL") as bar,
        get_cosmos(host=cosmos_host, port=cosmos_port, safe=cosmos_safe) as cosmos,
        get_frames(frame_size * 2, group=vdif_group, port=vdif_port) as frames,
    ):
        # wait until enough frames are buffered
        while len(frames.get(frame_size)) != frame_size:
            sleep(dt_epl / SECOND)

        try:
            while True:
                with take(dt_epl / SECOND):
                    # get the current telescope state
                    state = cosmos.receive_state()

                    # get the current VDIF samples (time x chan)
                    samples = get_samples(frames.get(frame_size + FRAMES_PER_SAMPLE))

                    # get the aggregated data (feed x freq)
                    aggregated = get_aggregated(
                        samples,
                        elevation=state.elevation,
                        feed_pattern=feed_pattern,
                        feed_origin=feed_origin,
                        freq_binning=freq_binning,
                        freq_range=Range(*freq_range),
                    )

                    # estimate the EPL (in m; feed)
                    epl, epl_cal = get_epl(aggregated)

                    # estimate the current subref parameters
                    subref = get_subref(epl, epl_cal)

                    # send the subref parameters to COSMOS
                    if not dry_run:
                        cosmos.send_subref(dX=subref.dX, dZ=subref.dZ)

                    # update the progress bar
                    bar.update(1)
        except KeyboardInterrupt:
            LOGGER.warning("Control interrupted by user.")
