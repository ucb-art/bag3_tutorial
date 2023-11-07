# BSD 3-Clause License

# Copyright (c) 2018, Regents of the University of California
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union, cast
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from bag.simulation.cache import SimulationDB, DesignInstance
from bag.simulation.measure import MeasurementManager
from bag.simulation.data import SimData, AnalysisType
from bag.concurrent.util import GatherHelper

from bag3_testbenches.measurement.data.tran import EdgeType
from bag3_testbenches.measurement.digital.util import setup_digital_tran
from bag3_testbenches.measurement.tran.digital import DigitalTranTB
from bag3_testbenches.measurement.ac.base import ACTB

class InvACMeas(MeasurementManager):
    """A simple inverter measurement manager.

    This class measures the input-output delay of an inverter, as well as the output rise/fall times.
    A rail-to-rail 50% duty-cycle square wave is used as the input.
    Average power is estimated using numpy.trapz.

    Notes
    -----
    The specification dictionary has the following entries in addition to the default ones:

    sim_params : Mapping[str, float]
        Required entries are listed below.

        t_per : float
            The period of the input stimulus.
        t_rf : float
            The rise/fall time of the input.
        c_load : float
            The load capacitance at the output.

    sim_envs : Sequence[str]
        The sequence of corners/simulation environments to characterize over.

    plot : bool
        Optional. True to plot time-domain waveforms for debug. Defaults to False.
    """

    def commit(self):
        self.plot: bool = self.specs.get('plot', False)

    async def async_measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                        dut: Optional[DesignInstance],
                                        harnesses: Optional[Sequence[DesignInstance]] = None) -> Mapping[str, Any]:
        sim_envs: Sequence[str] = self.specs['sim_envs']
        assert len(sim_envs) == 1
        # TODO: Exercise for the reader: extend features for multiple sim_envs    

        # Create input stimulus
        src_list = [
            dict(type='vsin', value='v_in', conns={'PLUS': 'in', 'MINUS': 'in_dc'}),
            dict(type='vdc', value='v_dc', conns={'PLUS': 'in_dc', 'MINUS': 'VSS'})
        ]

        # Create output loading
        load_list = [
            dict(pin='out', type='cap', value='c_load'),
        ]

        # Set up tbm_specs
        sup_src_name = ACTB.sup_src_name('VDD')
        sup_pwr_sig_name = f'{sup_src_name}:pwr'
        tb_params = dict(
            src_list=src_list,
            load_list=load_list,
            sim_envs=sim_envs,
            save_outputs=['in', 'out', sup_pwr_sig_name]  # supply source saved for power measurement
        )
        tbm_specs = self.specs['tbm_specs']
        tbm_specs.update(dict(dut_pins=dut.pin_names, **tb_params))

        # Create testbench manager and simulate
        tbm: ACTB = cast(ACTB, self.make_tbm(ACTB, tbm_specs))
        sim_results = await sim_db.async_simulate_tbm_obj(name, sim_dir, dut, tbm, tb_params, harnesses=harnesses)
        data = sim_results.data
        data.open_analysis(AnalysisType.AC)

        # Compute measurements
        freq = data['freq']
        vin = data['in']
        vout = data['out']

        # Compute DC gain
        gain = np.abs(vout / vin)[0]
        gain_dc = gain[0]
        gain_dB = 20 * np.log10(gain)

        # Compute phase
        phase = np.arctan(np.imag(vout/vin) / np.real(vout/vin))[0]
        phase_deg = phase / np.pi * 180

        # Compute 3dB BW
        gain_3dB = gain_dc * np.sqrt(1/2)
        gain_3dB_dB = 20 * np.log10(gain_3dB)
        idx_3dB = np.argwhere(gain >= gain_3dB)[-1]
        freq_3dB = freq[idx_3dB]

        # Plot results to a PDF
        with PdfPages(str(sim_dir / 'ac.pdf')) as pdf:
            fig = plt.figure()
            plt.subplot(211)
            plt.semilogx(freq, gain_dB, label='Gain [V/V]')
            plt.plot(freq_3dB, [gain_3dB_dB], 'o', label=f'3dB Corner = {freq_3dB[0]/1e9:0.2f}GHz')
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Gain [V/V]")
            plt.grid()
            plt.legend()

            plt.subplot(212)
            plt.semilogx(freq, phase_deg, label='Phase [deg]')
            plt.plot(freq_3dB, [phase_deg[idx_3dB]], 'o', label=f'3dB Corner = {freq_3dB[0]/1e9:0.2f}GHz')
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Phase [deg]")
            plt.grid()
            plt.legend()

            plt.suptitle("Bode Plots")
            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)

        # Return resuls
        return dict(
            sim_data=sim_results.data,
            gain=gain_dc,
            freq_3dB=freq_3dB,
        )

    @staticmethod
    def plot_results(results: Sequence[Mapping[str, Any]], plot_dir: Path) -> None:
        """Plots and saves results into a PDF.

        Parameters
        ----------
        results : Sequence[Mapping[str, Any]]
            The measurement results.
        plot_dir : Path
            The directory to save plots.
        """
        sim_data: Mapping[str, SimData] = results['sim_data']
        data0 = next(iter(sim_data.values()))
        swp_names = data0.sweep_params[1:-1]  # remove corner and time axes
        swp_shape = data0.data_shape[:-1]
        fvecs = {sim_env: np.reshape(data['freq'], data.data_shape) for sim_env, data in sim_data.items()}

        with PdfPages(str(plot_dir / 'ac.pdf')) as pdf:
            for i in np.ndindex(swp_shape):
                fig, [ax0, ax1] = plt.subplots(2, 1, constrained_layout=True)
                sfx = ', '.join([f'{k} = {data0.get_param_value(k)[i]}' for k in swp_names])
                fig.suptitle(f'Inverter Input & Output, {sfx}')
                ax0.set(xlabel='Time (s)', ylabel='in (V)')
                ax1.set(xlabel='Time (s)', ylabel='out (V)')
                ax0.grid()
                ax1.grid()
                for sim_env, data in sim_data.items():
                    time = fvecs[sim_env][i]
                    ax0.plot(time, data['in'][i], label=sim_env)
                    ax1.plot(time, data['out'][i], label=sim_env)
                ax0.legend()
                ax1.legend()
                pdf.savefig()
                plt.close(fig)


class InvMeas(MeasurementManager):
    """A simple inverter measurement manager.

    This class measures the input-output delay of an inverter, as well as the output rise/fall times.
    A rail-to-rail 50% duty-cycle square wave is used as the input.
    Average power is estimated using numpy.trapz.

    Notes
    -----
    The specification dictionary has the following entries in addition to the default ones:

    sim_params : Mapping[str, float]
        Required entries are listed below.

        t_per : float
            The period of the input stimulus.
        t_rf : float
            The rise/fall time of the input.
        c_load : float
            The load capacitance at the output.

    sim_envs : Sequence[str]
        The sequence of corners/simulation environments to characterize over.

    plot : bool
        Optional. True to plot time-domain waveforms for debug. Defaults to False.
    """

    def commit(self):
        self.plot: bool = self.specs.get('plot', False)

    async def async_measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                        dut: Optional[DesignInstance],
                                        harnesses: Optional[Sequence[DesignInstance]] = None) -> Mapping[str, Any]:
        sim_envs: Sequence[str] = self.specs['sim_envs']
        num_sim_envs = len(sim_envs)

        # Characterize across corners in separate simulations
        helper = GatherHelper()
        for sim_env in sim_envs:
            helper.append(self.async_meas_pvt(name, sim_dir / sim_env, sim_db, dut, harnesses, sim_env))

        meas_results = await helper.gather_err()

        # Combine measurement results from different corners into one data structure
        results = {'sim_envs': sim_envs}
        for meas_k in meas_results[0]:
            meas_vals = [meas_results[idx][meas_k] for idx in range(num_sim_envs)]
            if meas_k == 'sim_data':
                results[meas_k] = dict(zip(sim_envs, meas_vals))
            else:
                results[meas_k] = np.concatenate(meas_vals)

        # Optional plotting
        if self.plot:
            self.plot_results(results, sim_dir)

        return results

    async def async_meas_pvt(self, name: str, sim_dir: Path, sim_db: SimulationDB, dut: Optional[DesignInstance],
                             harnesses: Optional[Sequence[DesignInstance]], sim_env: str) \
            -> Mapping[str, Union[np.ndarray, SimData]]:
        """A coroutine that performs measurement for one corner.

        Parameters
        ----------
        name : str
            Name of this measurement.
        sim_dir : Path
            Simulation directory.
        sim_db : SimulationDB
            The simulation database object.
        dut : Optional[DesignInstance]
            The DUT to measure.
        harnesses : Optional[Sequence[DesignInstance]]
            The list of DUT and harnesses to measure.
        sim_env : str
            The simulation environment.

        Returns
        -------
        output : Mapping[str, Union[np.ndarray, SimData]]
            The measurement results. Contains the following entries:

            sim_data : SimData
                The raw simulation data.
            delay_rise : np.ndarray
                The input-to-output delay for a low to high transition at the output.
            delay_fall : np.ndarray
                The input-to-output delay for a high to low transition at the output.
            trans_rise : np.ndarray
                The output rise time.
            trans_fall : np.ndarray
                The output fall time.
            pwr_avg : np.ndarray
                The average power dissipation.
        """
        # Create input stimulus
        pulse_list = [dict(pin='in', tper='t_per', tpw='t_per/2', trf='t_rf')]

        # Create output loading
        load_list = [dict(pin='out', type='cap', value='c_load')]

        # Set up tbm_specs
        sup_src_name = DigitalTranTB.sup_src_name('VDD')
        sup_pwr_sig_name = f'{sup_src_name}:pwr'
        tb_params = dict(
            pulse_list=pulse_list,
            load_list=load_list,
            sim_envs=[sim_env],
            save_outputs=['in', 'out', sup_pwr_sig_name]  # supply source saved for power measurement
        )
        tbm_specs, tb_params = setup_digital_tran(self.specs, dut, **tb_params)
        tbm_specs['sim_params']['t_sim'] = '2*t_per'

        # Create testbench manager and simulate
        tbm: DigitalTranTB = cast(DigitalTranTB, self.make_tbm(DigitalTranTB, tbm_specs))
        sim_results = await sim_db.async_simulate_tbm_obj(name, sim_dir, dut, tbm, tb_params, harnesses=harnesses)
        data = sim_results.data
        data.open_analysis(AnalysisType.TRAN)

        # Compute measurements
        delay_rise = tbm.calc_delay(data, 'in', 'out', EdgeType.FALL, EdgeType.RISE)
        delay_fall = tbm.calc_delay(data, 'in', 'out', EdgeType.RISE, EdgeType.FALL)
        trans_rise = tbm.calc_trf(data, 'out', True)
        trans_fall = tbm.calc_trf(data, 'out', False)
        pwr_avg = np.full_like(delay_rise, np.nan)
        time = np.reshape(data['time'], data.data_shape)
        for i in np.ndindex(pwr_avg.shape):
            tvec = time[i]
            pwr = -data[sup_pwr_sig_name][i]
            # filter NaNs
            mask = ~np.isnan(tvec)
            tvec, pwr = tvec[mask], pwr[mask]
            pwr_avg[i] = np.trapz(pwr, x=tvec) / (tvec.max() - tvec.min())

        return dict(
            sim_data=sim_results.data,
            delay_rise=delay_rise,
            delay_fall=delay_fall,
            trans_rise=trans_rise,
            trans_fall=trans_fall,
            pwr_avg=pwr_avg
        )

    @staticmethod
    def plot_results(results: Mapping[str, Any], plot_dir: Path) -> None:
        """Plots and saves time-domain waveforms into a PDF.

        Parameters
        ----------
        results : Mapping[str, Any]
            The measurement results.
        plot_dir : Path
            The directory to save plots.
        """
        sim_data: Mapping[str, SimData] = results['sim_data']
        data0 = next(iter(sim_data.values()))
        swp_names = data0.sweep_params[1:-1]  # remove corner and time axes
        swp_shape = data0.data_shape[:-1]
        tvecs = {sim_env: np.reshape(data['time'], data.data_shape) for sim_env, data in sim_data.items()}

        with PdfPages(str(plot_dir / 'tran.pdf')) as pdf:
            for i in np.ndindex(swp_shape):
                fig, [ax0, ax1] = plt.subplots(2, 1, constrained_layout=True)
                sfx = ', '.join([f'{k} = {data0.get_param_value(k)[i]}' for k in swp_names])
                fig.suptitle(f'Inverter Input & Output, {sfx}')
                ax0.set(xlabel='Time (s)', ylabel='in (V)')
                ax1.set(xlabel='Time (s)', ylabel='out (V)')
                ax0.grid()
                ax1.grid()
                for sim_env, data in sim_data.items():
                    time = tvecs[sim_env][i]
                    ax0.plot(time, data['in'][i], label=sim_env)
                    ax1.plot(time, data['out'][i], label=sim_env)
                ax0.legend()
                ax1.legend()
                pdf.savefig()
                plt.close(fig)
