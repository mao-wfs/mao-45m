__all__ = ["control"]


# standard library
from logging import getLogger
from time import sleep


# dependencies
import numpy as np
from ndtools import Range
from tqdm import tqdm
from .convert import get_converter as get_subref_converter
from ..cosmos import get_cosmos
from ..epl.convert import get_aggregated, get_converter as get_epl_converter
from ..vdif import FRAMES_PER_SAMPLE
from ..vdif.receive import get_frames
from ..vdif.convert import get_samples
from ..utils import take


# constants
LOGGER = getLogger(__name__)


def control(
    *,
    feed_model: str,
    feed_origin: str,
    feed_pattern: str,
    # options for the EPL estimates
    cal_interval: int = 30,  # s
    freq_binning: int = 8,
    freq_min: float = 19.5e9,  # Hz
    freq_max: float = 22.5e9,  # Hz
    integ_per_sample: float = 0.01,  # s
    integ_per_epl: float = 0.5,  # s
    # options for the subref control
    dry_run: bool = False,
    gain_dX: float = 0.1,
    gain_dZ: float = 0.1,
    # options for network connection
    cosmos_host: str = "127.0.0.1",
    cosmos_port: int = 11111,
    vdif_group: str = "239.0.0.1",
    vdif_port: int = 22222,
    # option for display and logging
    status: bool = True,
) -> None:
    """Control the subreflector of the Nobeyama 45m telescope by MAO."""
    # define the frame size for each EPL estimate
    frame_size = FRAMES_PER_SAMPLE * int(integ_per_epl / integ_per_sample)

    # create the EPL and subref converters
    get_epl = get_epl_converter(cal_interval)
    get_subref = get_subref_converter(feed_model, gain_dX, gain_dZ)

    with (
        tqdm(disable=not status, unit="EPL") as bar,
        get_cosmos(cosmos_host, cosmos_port) as cosmos,
        get_frames(frame_size * 2, group=vdif_group, port=vdif_port) as frames,
    ):
        # wait until enough frames are buffered
        while len(frames.get(frame_size)) != frame_size:
            sleep(integ_per_epl)

        try:
            while True:
                with take(integ_per_epl):
                    # get the current telescope state
                    state = cosmos.receive_state()

                    # get the current VDIF samples (time x chan)
                    samples = get_samples(frames.get(frame_size + FRAMES_PER_SAMPLE))

                    # get the aggregated data (feed x freq)
                    aggregated = get_aggregated(
                        samples,
                        elevation=state.elevation,
                        feed_pattern=tuple(feed_pattern),
                        feed_origin=np.datetime64(feed_origin),
                        freq_binning=freq_binning,
                        freq_range=Range(freq_min, freq_max),
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
