import numpy as np
import finufft
from collections import namedtuple
from quartical.gains.conversion import no_op, trig_to_angle
from quartical.gains.parameterized_gain import ParameterizedGain
from quartical.gains.delay_and_tec.kernel import (
    delay_and_tec_solver,
    delay_and_tec_params_to_gains
)
from quartical.gains.general.flagging import (
    apply_gain_flags_to_gains,
    apply_param_flags_to_params
)
from quartical.gains.general.generics import compute_corrected_residual
# from numba import njit


# @njit
# def bar(string):
#     pass


# Overload the default measurement set inputs to include the frequencies.
ms_inputs = namedtuple(
    'ms_inputs', ParameterizedGain.ms_inputs._fields + ('CHAN_FREQ',)
)


class DelayAndTec(ParameterizedGain):

    solver = staticmethod(delay_and_tec_solver)
    ms_inputs = ms_inputs

    native_to_converted = (
        (1, (no_op,)),
        (1, (no_op,))
    )
    converted_to_native = (
        (1, no_op),
        (1, no_op)
    )
    converted_dtype = np.float64
    native_dtype = np.float64

    def __init__(self, term_name, term_opts):

        super().__init__(term_name, term_opts)

    @classmethod
    def _make_freq_map(cls, chan_freqs, chan_widths, freq_interval):
        # Overload gain mapping construction - we evaluate it in every channel.
        return np.arange(chan_freqs.size, dtype=np.int32)

    @classmethod
    def make_param_names(cls, correlations):

        # TODO: This is not dasky, unlike the other functions. Delayed?
        parameterisable = ["XX", "YY", "RR", "LL"]

        param_corr = [c for c in correlations if c in parameterisable]

        template = ("tec_{}", "delay_{}")

        return [n.format(c) for c in param_corr for n in template]


    def init_term(self, term_spec, ref_ant, ms_kwargs, term_kwargs):
        """Initialise the gains (and parameters)."""

        gains, gain_flags, params, param_flags = super().init_term(
            term_spec, ref_ant, ms_kwargs, term_kwargs
        )

        # Convert the parameters into gains.
        delay_and_tec_params_to_gains(
            params,
            gains,
            ms_kwargs["CHAN_FREQ"],
            term_kwargs[f"{self.name}_param_freq_map"],
        )

        if self.load_from or not self.initial_estimate:

            apply_param_flags_to_params(param_flags, params, 0)
            apply_gain_flags_to_gains(gain_flags, gains)

            return gains, gain_flags, params, param_flags

        data = ms_kwargs["DATA"]  # (row, chan, corr)
        flags = ms_kwargs["FLAG"]  # (row, chan)
        a1 = ms_kwargs["ANTENNA1"]
        a2 = ms_kwargs["ANTENNA2"]
        chan_freq = ms_kwargs["CHAN_FREQ"]
        row_map = ms_kwargs["ROW_MAP"]
        row_weights = ms_kwargs["ROW_WEIGHTS"]
        t_map = term_kwargs[f"{term_spec.name}_time_map"]
        f_map = term_kwargs[f"{term_spec.name}_param_freq_map"]
        _, n_chan, n_ant, n_dir, n_corr = gains.shape


        #what about dir_maps?
        # dir_maps = np.zeros(1, dtype=np.int32)
        dir_maps = (term_kwargs[f"{term_spec.name}_dir_map"],)

        # We only need the baselines which include the ref_ant.
        sel = np.where((a1 == ref_ant) | (a2 == ref_ant))
        a1 = a1[sel]
        a2 = a2[sel]
        t_map = t_map[sel]
        data = data[sel]
        flags = flags[sel]

        data[flags == 1] = 0  # Ignore UV-cut, otherwise there may be no est.

        utint = np.unique(t_map)
        ufint = np.unique(f_map)


        if n_corr == 1:
            n_paramt = 1 #number of parameters in TEC
            n_paramk = 1 #number of parameters in delay
        elif n_corr in (2, 4):
            n_paramt = 2
            n_paramk = 2
        else:
            raise ValueError("Unsupported number of correlations.")

        n_param = params.shape[-1]
        assert n_param == n_paramk + n_paramt

        for ut in utint:
            sel = np.where((t_map == ut) & (a1 != a2))
            ant_map_pq = np.where(a1[sel] == ref_ant, a2[sel], 0)
            ant_map_qp = np.where(a2[sel] == ref_ant, a1[sel], 0)
            ant_map = ant_map_pq + ant_map_qp

            ref_data = np.zeros((n_ant, n_chan, n_corr), dtype=np.complex128)
            counts = np.zeros((n_ant, n_chan), dtype=int)
            np.add.at(
                ref_data,
                ant_map,
                data[sel]
            )
            np.add.at(
                counts,
                ant_map,
                flags[sel] == 0
            )
            np.divide(
                ref_data,
                counts[:, :, None],
                where=counts[:, :, None] != 0,
                out=ref_data
            )

            for uf in ufint:

                fsel = np.where(f_map == uf)[0]
                sel_n_chan = fsel.size
                ##in inverse frequency domain
                invfreq = 1./chan_freq

                fsel_data = ref_data[:, fsel]
                valid_ant = fsel_data.any(axis=(1, 2))

                #Initialise array to contain delay estimates
                delay_est = np.zeros((n_ant, n_paramk), dtype=np.float64)
                delay_est, fft_arrk, fft_freqk = self.initial_estimates(
                    fsel_data, delay_est, chan_freq, valid_ant, type="k"
                )

                tec_est = np.zeros((n_ant, n_paramt), dtype=np.float64)
                tec_est, fft_arrt, fft_freqt = self.initial_estimates(
                    fsel_data, tec_est, invfreq, valid_ant, type="t"
                )

                
                # path00 = "/home/russeeawon/testing/thesis_figures/expt10_tandd/"
                # path00 = "/home/russeeawon/testing/thesis_figures/expt10_tandd_solved/"
                # path00 = "/home/russeeawon/testing/thesis_figures/expt11_solvingdelay/"
                # path00 = "/home/russeeawon/testing/thesis_figures/expt11_solvingdelayb/"
                # path00 = "/home/russeeawon/testing/thesis_figures/expt11_solvingtec/"
                # path00 = "/home/russeeawon/testing/thesis_figures/expt11_solvingtecb/"
                # path00 = "/home/russeeawon/testing/thesis_figures/expt12_tandd/"
                # path00 = "/home/russeeawon/testing/thesis_figures/expt12_tandd_solved/"
                # path00 = "/home/russeeawon/testing/thesis_figures/expt13_solvingdelay/"
                # path00 = "/home/russeeawon/testing/thesis_figures/expt13_solvingdelayb/"
                # path00 = "/home/russeeawon/testing/thesis_figures/expt13_solvingtec/"
                # path00 = "/home/russeeawon/testing/thesis_figures/expt13_solvingtecb/"

                #Tweaking the selection criteria
                # path00 = "/home/russeeawon/testing/thesis_figures/expt13_solvingdelay_altered/"
                path00 = "/home/russeeawon/testing/thesis_figures/expt13_solvingdelayb_altered/"


                path01 = ""

                path0 = path00+path01
                np.save(path0+"delayest0.npy", delay_est)
                np.save(path0+"delay_fftarr0.npy", fft_arrk)
                np.save(path0+"delay_fft_freq0.npy", fft_freqk)
                np.save(path0+"tecest0.npy", tec_est)
                np.save(path0+"tec_fftarr0.npy", fft_arrt)
                np.save(path0+"tec_fft_freq0.npy", fft_freqt)


                #Selecting the dominant peak and setting the other parameter to zero.
                for t, p, q in zip(t_map[sel], a1[sel], a2[sel]):
                    if p == ref_ant:
                        if n_corr == 1:
                            if np.max(np.abs(fft_arrk[q, :, 0])**2) > np.max(np.abs(fft_arrt[q, :, 0])**2):
                                #delay is dominant >> only assign delay
                                params[t, uf, q, 0, 0] = 0
                                params[t, uf, q, 0, 1] = -delay_est[q]
                            else:
                                #tec is dominant >> only assign tec
                                params[t, uf, q, 0, 0] = -tec_est[q]
                                params[t, uf, q, 0, 1] = 0
                        elif n_corr > 1:
                            if np.max(np.abs(fft_arrk[q, :, 0])**2) > np.max(np.abs(fft_arrt[q, :, 0])**2):
                                #only assign delay
                                params[t, uf, q, 0, 0] = 0
                                params[t, uf, q, 0, 1] = -delay_est[q, 0]
                            else:
                                #only assign tec
                                params[t, uf, q, 0, 0] = -tec_est[q, 0]
                                params[t, uf, q, 0, 1] = 0
                            
                            if np.max(np.abs(fft_arrk[q, :, 1])**2) > np.max(np.abs(fft_arrt[q, :, 1])**2):
                                #only assign delay
                                params[t, uf, q, 0, 2] = 0
                                params[t, uf, q, 0, 3] = -delay_est[q, 1]
                            else:
                                #only assign tec
                                params[t, uf, q, 0, 2] = -tec_est[q, 1]
                                params[t, uf, q, 0, 3] = 0

                    else:
                        if n_corr == 1:
                            if np.max(np.abs(fft_arrk[p, :, 0])**2) > np.max(np.abs(fft_arrt[p, :, 0])**2):
                                #delay is dominant >> only assign delay
                                params[t, uf, p, 0, 0] = 0
                                params[t, uf, p, 0, 1] = delay_est[p]
                            else:
                                #tec is dominant >> only assign tec
                                params[t, uf, p, 0, 0] = tec_est[p]
                                params[t, uf, p, 0, 1] = 0
                        elif n_corr > 1:
                            if np.max(np.abs(fft_arrk[p, :, 0])**2) > np.max(np.abs(fft_arrt[p, :, 0])**2):
                                #only assign delay
                                params[t, uf, p, 0, 0] = 0
                                params[t, uf, p, 0, 1] = delay_est[p, 0]
                            else:
                                #only assign tec
                                params[t, uf, p, 0, 0] = tec_est[p, 0]
                                params[t, uf, p, 0, 1] = 0
                            
                            if np.max(np.abs(fft_arrk[p, :, 1])**2) > np.max(np.abs(fft_arrt[p, :, 1])**2):
                                #only assign delay
                                params[t, uf, p, 0, 2] = 0
                                params[t, uf, p, 0, 3] = delay_est[p, 1]
                            else:
                                #only assign tec
                                params[t, uf, p, 0, 2] = tec_est[p, 1]
                                params[t, uf, p, 0, 3] = 0

        delay_and_tec_params_to_gains(
            params,
            gains,
            ms_kwargs["CHAN_FREQ"],
            term_kwargs[f"{self.name}_param_freq_map"],
        )

        #Save the midway gains
        np.save(path0+"gains0.npy", gains)


        # gain_tuple spans from the different gain types, here we are only \ 
        # considering one gain type (delay_and_tec).
        gain_tuple = (gains,)
        #tuples required for time and frequency maps 
        corrected_data = compute_corrected_residual(
            data, gain_tuple, a1, a2, (t_map,), (f_map,), dir_maps, row_map, row_weights, n_corr
        )

        #A second round of estimation
        for ut in utint:
            sel = np.where((t_map == ut) & (a1 != a2))
            ant_map_pq = np.where(a1[sel] == ref_ant, a2[sel], 0)
            ant_map_qp = np.where(a2[sel] == ref_ant, a1[sel], 0)
            ant_map = ant_map_pq + ant_map_qp

            ref_data = np.zeros((n_ant, n_chan, n_corr), dtype=np.complex128)
            counts = np.zeros((n_ant, n_chan), dtype=int)
            np.add.at(
                ref_data,
                ant_map,
                corrected_data[sel]
            )
            np.add.at(
                counts,
                ant_map,
                flags[sel] == 0
            )
            np.divide(
                ref_data,
                counts[:, :, None],
                where=counts[:, :, None] != 0,
                out=ref_data
            )

            for uf in ufint:

                fsel = np.where(f_map == uf)[0]
                sel_n_chan = fsel.size
                ##in inverse frequency domain
                invfreq = 1./chan_freq

                fsel_data = ref_data[:, fsel]
                valid_ant = fsel_data.any(axis=(1, 2))

                #Initialise array to contain delay estimates
                delay_est = np.zeros((n_ant, n_paramk), dtype=np.float64)
                delay_est, fft_arrk, fft_freqk = self.initial_estimates(
                    fsel_data, delay_est, chan_freq, valid_ant, type="k"
                )

                tec_est = np.zeros((n_ant, n_paramt), dtype=np.float64)
                tec_est, fft_arrt, fft_freqt = self.initial_estimates(
                    fsel_data, tec_est, invfreq, valid_ant, type="t"
                )

                np.save(path0+"delayest1.npy", delay_est)
                np.save(path0+"delay_fftarr1.npy", fft_arrk)
                np.save(path0+"delay_fft_freq1.npy", fft_freqk)
                np.save(path0+"tecest1.npy", tec_est)
                np.save(path0+"tec_fftarr1.npy", fft_arrt)
                np.save(path0+"tec_fft_freq1.npy", fft_freqt)

                #select again!
                #Attempting to tweak the peak selection for the previously non-dominant peak
                for t, p, q in zip(t_map[sel], a1[sel], a2[sel]):
                    if p == ref_ant:
                        if n_corr == 1:
                            if np.max(np.abs(fft_arrk[q, :, 0])**2) > np.max(np.abs(fft_arrt[q, :, 0])**2):
                                #delay is dominant >> only assign delay
                                params[t, uf, q, 0, 0] = -tec_est[q]
                                params[t, uf, q, 0, 1] = 0
                            else:
                                #tec is dominant >> only assign tec
                                params[t, uf, q, 0, 0] = 0
                                params[t, uf, q, 0, 1] = -delay_est[q]
                        elif n_corr > 1:
                            if np.max(np.abs(fft_arrk[q, :, 0])**2) > np.max(np.abs(fft_arrt[q, :, 0])**2):
                                #only assign delay
                                params[t, uf, q, 0, 0] = 0
                                params[t, uf, q, 0, 1] = -delay_est[q, 0]
                            else:
                                #only assign tec
                                params[t, uf, q, 0, 0] = -tec_est[q, 0]
                                params[t, uf, q, 0, 1] = 0
                            
                            if np.max(np.abs(fft_arrk[q, :, 1])**2) > np.max(np.abs(fft_arrt[q, :, 1])**2):
                                #only assign delay
                                params[t, uf, q, 0, 2] = 0
                                params[t, uf, q, 0, 3] = -delay_est[q, 1]
                            else:
                                #only assign tec
                                params[t, uf, q, 0, 2] = -tec_est[q, 1]
                                params[t, uf, q, 0, 3] = 0

                    else:
                        if n_corr == 1:
                            if np.max(np.abs(fft_arrk[p, :, 0])**2) > np.max(np.abs(fft_arrt[p, :, 0])**2):
                                #delay is dominant >> only assign delay
                                params[t, uf, p, 0, 0] = tec_est[p]
                                params[t, uf, p, 0, 1] = 0
                            else:
                                #tec is dominant >> only assign tec
                                params[t, uf, p, 0, 0] = 0
                                params[t, uf, p, 0, 1] = delay_est[p]
                        elif n_corr > 1:
                            if np.max(np.abs(fft_arrk[p, :, 0])**2) > np.max(np.abs(fft_arrt[p, :, 0])**2):
                                #only assign delay
                                params[t, uf, p, 0, 0] = 0
                                params[t, uf, p, 0, 1] = delay_est[p, 0]
                            else:
                                #only assign tec
                                params[t, uf, p, 0, 0] = tec_est[p, 0]
                                params[t, uf, p, 0, 1] = 0
                            
                            if np.max(np.abs(fft_arrk[p, :, 1])**2) > np.max(np.abs(fft_arrt[p, :, 1])**2):
                                #only assign delay
                                params[t, uf, p, 0, 2] = 0
                                params[t, uf, p, 0, 3] = delay_est[p, 1]
                            else:
                                #only assign tec
                                params[t, uf, p, 0, 2] = tec_est[p, 1]
                                params[t, uf, p, 0, 3] = 0


        apply_param_flags_to_params(param_flags, params, 0)
        apply_gain_flags_to_gains(gain_flags, gains)

        delay_and_tec_params_to_gains(
            params,
            gains,
            ms_kwargs["CHAN_FREQ"],
            term_kwargs[f"{self.name}_param_freq_map"],
        )

        return gains, gain_flags, params, param_flags


    def initial_estimates(self, fsel_data, est_arr, freq, valid_ant, type="k"):
        """
        This function return the set of initial estimates for each param in params.
        type is either k (delay) or t (tec).

        """
        
        n_ant, n_param = est_arr.shape

        dfreq = np.abs(freq[-2] - freq[-1])
        #Maximum reconstructable delta
        max_delta = (2*np.pi)/ dfreq
        nyq_rate = 1./ (2*(freq.max() - freq.min()))
        nbins = int(max_delta/ nyq_rate)

        if type == "k":
            fft_freq = np.fft.fftfreq(nbins, dfreq)
            fft_freq = np.fft.fftshift(fft_freq)

            #when not using finufft
            # fft_arr = np.abs(
            #     np.fft.fft(fsel_data, n=nbins, axis=1)
            # )
            # fft_arr = np.fft.fftshift(fft_arr, axes=1)
        elif type == "t":
            ##factor for rescaling frequency
            ffactor = 1 #1e8
            freq *= ffactor
            fft_freq = np.linspace(0.5*-max_delta, 0.5*max_delta, nbins)
        else:
            raise TypeError("Unsupported parameter type.")

        fft_arr = np.zeros((n_ant, nbins, n_param), dtype=fsel_data.dtype)

        for i in range(n_param):
            if i == 0:
                datak = fsel_data[:, :, 0]
            elif i == 1:
                datak = fsel_data[:, :, -1]
            else:
                raise ValueError("Unsupported number of parameters for delay.")

            vis_finufft = finufft.nufft1d3(
                2 * np.pi * freq,
                datak,
                fft_freq,
                eps=1e-6,
                isign=-1
            )
            fft_arr[:, :, i] = vis_finufft
            est_arr[:, i] = fft_freq[np.argmax(np.abs(vis_finufft), axis=1)]
        
        est_arr[~valid_ant] = 0

        return est_arr, fft_arr, fft_freq
