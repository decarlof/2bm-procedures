"""
Piezosystem Jena NV200/D NET — triggered step mode using the nv200 Python library.

Loads up to 1024 positions into the arbitrary waveform generator buffer.
Each rising edge on the TRG IN connector (pin 3 of the I/O D-Sub, TTL 0/3.3-5V)
advances the actuator to the next position.

Requirements:
    pip install nv200 numpy

Usage:
    1. Stop the EPICS IOC (only one Telnet connection allowed at a time).
    2. Run this script.
    3. Each TTL rising edge on TRG IN steps to the next position.
    4. Press Enter to stop and restore manual control, then restart the IOC.

Notes:
    - Requires a sensor-equipped actuator for closed loop (closed_loop=True).
    - Positions must be within the actuator's closed-loop stroke range.
    - The library's WaveformUnit.POSITION converts µm values to % of range internally.
"""

import argparse
import asyncio
import numpy as np

from nv200.nv200_device import NV200Device
from nv200.waveform_generator import WaveformGenerator, WaveformUnit
from nv200.shared_types import TransportType, PidLoopMode, TriggerInFunction
from nv200.telnet_protocol import TelnetProtocol


async def _progress(done: int, total: int) -> None:
    if done % 128 == 0 or done == total:
        print(f'    {done}/{total}', flush=True)


async def setup_triggered_step(
    dev: NV200Device,
    positions: list[float] | None = None,
    n: int = 256,
    closed_loop: bool = True,
    random: bool = False,
    label: str = '',
) -> WaveformGenerator:
    """
    Configure the NV200 so each rising edge on TRG IN moves to the next position.

    Parameters
    ----------
    dev : NV200Device
        Connected device instance.
    positions : list of float or None
        Positions in µm (closed-loop physical units). Max 1024 values.
        If None, n evenly spaced positions spanning the full stroke are used.
    n : int
        Number of positions to generate when positions is None.
    closed_loop : bool
        True for closed loop (requires sensor), False for open loop.
    label : str
        Optional label for console output.

    Returns
    -------
    WaveformGenerator
        The running waveform generator (call wg.stop() to halt).
    """
    if label:
        print(f'--- {label} ---')

    # Set control mode
    await dev.pid.set_mode(PidLoopMode.CLOSED_LOOP if closed_loop else PidLoopMode.OPEN_LOOP)

    # Read device stroke limits
    posmin, posmax = await dev.get_position_range()
    print(f'  Actuator stroke: {posmin} … {posmax} µm')

    if positions is None:
        if random:
            positions = list(np.random.uniform(posmin, posmax, n))
            print(f'  Auto-generated {n} random positions.')
        else:
            positions = list(np.linspace(posmin, posmax, n))
            print(f'  Auto-generated {n} evenly-spaced positions.')

    # Validate against device stroke
    out_of_range = [p for p in positions if p < posmin or p > posmax]
    if out_of_range:
        raise ValueError(
            f'{len(out_of_range)} position(s) outside actuator range '
            f'[{posmin}, {posmax}]: e.g. {out_of_range[0]}'
        )

    n_pos = len(positions)
    if n_pos > WaveformGenerator.NV200_WAVEFORM_BUFFER_SIZE:
        raise ValueError(
            f'Maximum {WaveformGenerator.NV200_WAVEFORM_BUFFER_SIZE} positions, got {n_pos}'
        )

    # Load positions into the waveform buffer.
    # WaveformUnit.POSITION: values in µm; the library converts to % of range internally.
    print(f'  Loading {n_pos} positions into buffer...', flush=True)
    wg = WaveformGenerator(dev)
    await wg.set_waveform_buffer(
        buffer=positions,
        unit=WaveformUnit.POSITION,
        on_progress=_progress,
    )

    # Set loop range: iterate over all loaded positions, starting at index 0
    await wg.configure_waveform_loop(
        start_index=0,
        loop_start_index=0,
        loop_end_index=n_pos - 1,
    )

    # Trigger input: each rising edge advances the waveform index by one step
    await dev.set_trigger_function(TriggerInFunction.WAVEFORM_STEP)

    # Start waveform generator (cycles=0 → run indefinitely, driven by trigger)
    await wg.start(cycles=0, start_index=0)

    pos = await dev.get_current_position()
    print(f'  Running. {n_pos} positions loaded. Current position: {pos:.3f} µm')
    return wg, positions


async def main():
    parser = argparse.ArgumentParser(description='NV200 triggered step mode')
    parser.add_argument('--random', action='store_true',
                        help='Use random positions instead of evenly-spaced (default: linspace)')
    parser.add_argument('--n', type=int, default=256,
                        help='Number of positions to load (default: 256, max: 1024)')
    args = parser.parse_args()

    IP_X = '10.54.113.126'
    IP_Y = '10.54.113.125'

    print(f'Connecting to X ({IP_X})...', flush=True)
    dev_x = NV200Device(transport=TelnetProtocol(host=IP_X))
    await dev_x.connect(auto_adjust_comm_params=True)
    print(f'Connecting to Y ({IP_Y})...', flush=True)
    dev_y = NV200Device(transport=TelnetProtocol(host=IP_Y))
    await dev_y.connect(auto_adjust_comm_params=True)

    wg_x = wg_y = None
    try:
        wg_x, pos_x = await setup_triggered_step(dev_x, n=args.n, closed_loop=True, random=args.random, label='X axis')
        wg_y, pos_y = await setup_triggered_step(dev_y, n=args.n, closed_loop=True, random=args.random, label='Y axis')

        np.savetxt('positions_x.txt', pos_x, fmt='%.4f', header='X positions (um)')
        np.savetxt('positions_y.txt', pos_y, fmt='%.4f', header='Y positions (um)')
        print(f'Positions saved to positions_x.txt and positions_y.txt')

        print('\nRunning. Each rising edge on TRG IN (I/O connector pin 3) steps to the next position.')
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: input('Press Enter to stop...\n'))
    finally:
        print('Stopping...')
        for wg, dev in [(wg_x, dev_x), (wg_y, dev_y)]:
            if wg is not None:
                try:
                    await wg.stop()
                except Exception:
                    pass
            try:
                await dev.set_trigger_function(TriggerInFunction.DISABLED)
                await dev.close()
            except Exception:
                pass
        print('Stopped. Manual control restored.')


if __name__ == '__main__':
    asyncio.run(main())
