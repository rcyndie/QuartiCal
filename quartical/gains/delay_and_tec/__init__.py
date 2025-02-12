import numpy as np
import finufft
from collections import namedtuple
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
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
        n_tint = utint.size
        n_fint = ufint.size
        # NOTE: This determines the number of subintervals which are used to 
        # estimate the delay and tec values. More subintervals will typically
        # yield better estimates at the cost of SNR.
        n_subint = 2

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

        ctz_tec = np.empty((n_tint, n_fint, n_subint, n_ant))
        ctz_delay = np.empty((n_tint, n_fint, n_subint, n_ant))
        gradients = np.empty((n_tint, n_fint, n_subint))

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
                fsel_nchan = fsel.size
                ##in inverse frequency domain
                fsel_chan = chan_freq[fsel]
                invfreq = 1./fsel_chan

                fsel_data = ref_data[:, fsel]
                valid_ant = fsel_data.any(axis=(1, 2))

                subint_stride = int(np.ceil(fsel_nchan / n_subint))

                for i, si in enumerate(range(0, fsel_nchan, subint_stride)):

                    si_sel = slice(si, si + subint_stride)

                    subint_data = fsel_data[:, si_sel]
                    subint_freq = fsel_chan[si_sel]
                    subint_ifreq = 1/subint_freq

                    gradients[ut, uf, i], _ = np.polyfit(
                        subint_freq, subint_ifreq, deg=1
                    )

                    #Initialise array to contain delay and tec estimates
                    delay_est = np.zeros((n_ant, n_paramk), dtype=np.float64)
                    delay_est, fft_arrk, fft_freqk = self.initial_estimates(
                        subint_data, delay_est, subint_freq, valid_ant, type="k"
                    )

                # tec_est = np.zeros((n_ant, n_paramt), dtype=np.float64)
                # tec_est, fft_arrt, fft_freqt = self.initial_estimates(
                #     fsel_data, tec_est, invfreq, valid_ant, type="t"
                # )

                # pst = (fft_arrt * fft_arrt.conj()).real
                # for ai in range(n_ant):
                #     ctz = cumulative_trapezoid(pst[ai, :, 0], fft_freqt)
                #     median_i = np.argwhere(ctz >= ctz.max()/2)[0]
                #     ctz_tec[ut, uf, ai] = fft_freqt[median_i]

                    psk = (fft_arrk * fft_arrk.conj()).real
                    for ai in range(n_ant):
                        ctz = cumulative_trapezoid(psk[ai, :, 0], fft_freqk)
                        median_i = np.argwhere(ctz >= ctz.max()/2)[0]
                        ctz_delay[ut, uf, i, ai] = fft_freqk[median_i]

                # if ut == 0:

                #     true_tec = np.array(
                #         [
                #             0.00000000e+00,
                #             -2.48157553e+10,
                #             9.91282870e+09,
                #             -5.69428194e+10,
                #             -4.89735646e+10,
                #             3.66740214e+09,
                #             -6.64857439e+10
                #         ]
                #     )
                #     true_delay = np.array(
                #         [
                #             0.00000000e+00,
                #             -7.45331380e-09,
                #             6.68659373e-09,
                #             3.22179062e-09,
                #             -2.40328479e-09,
                #             -1.91728836e-08,
                #             -1.91232591e-08
                #         ]
                #     )

                #     plotsel = slice(fft_freqt.size//2-1000, fft_freqt.size//2+1000)

                #     TEC = (ctz_delay[ut, 0] - ctz_delay[ut, 1])/(gradients[ut, 0] - gradients[ut, 1])
                #     K = ctz_delay[ut, 0] - gradients[ut, 0] * TEC

                #     plt.figure()
                #     for foo in range(fft_arrk.shape[0]):
                #         plt.plot(fft_freqt[plotsel], pst[foo, plotsel, 0])
                #         plt.axvline(-true_tec[foo])
                #         plt.axvline(-(true_delay[foo] / gradients[ut, uf] + true_tec[foo]), c="r")
                #         plt.axvline(ctz_tec[ut, uf, foo], c="k")
                #         plt.axvline(TEC[foo], c="g")
                #         plt.title("PS - TEC")
                #         if uf == 1:
                #             plt.show()
                #         else:
                #             plt.clf()

                #     plt.figure()
                #     for foo in range(fft_arrk.shape[0]):
                #         plt.plot(fft_freqk[plotsel], psk[foo, plotsel, 0])
                #         plt.axvline(-true_delay[foo])
                #         plt.axvline(-(true_tec[foo] * gradients[ut, uf] + true_delay[foo]), c="r")
                #         plt.axvline(ctz_delay[ut, uf, foo], c="k")
                #         plt.axvline(K[foo], c="g")
                #         plt.title("PS - Clock")
                #         if uf == 1:
                #             plt.show()
                #         else:
                #             plt.clf()

        tec_numerator = np.diff(ctz_delay, axis=2)
        tec_denominator = np.diff(gradients, axis=2)[..., None]
        tec_est = (tec_numerator / tec_denominator)
        delay_est = ctz_delay[:, :, :-1] - gradients[..., :-1, None] * tec_est

        tec_est = tec_est.mean(axis=2)
        delay_est = delay_est.mean(axis=2)

        tec_est[:, :, ref_ant] = 0
        delay_est[:, :, ref_ant] = 0

        tec_est[:, :, ref_ant:] = -tec_est[:, :, ref_ant:]
        delay_est[:, :, ref_ant:] = -delay_est[:, :, ref_ant:]

        # TODO: Handle the correlation axis throughout the estimates.
        params[:, :, :, 0, 0] = tec_est
        params[:, :, :, 0, 1] = delay_est

        delay_and_tec_params_to_gains(
            params,
            gains,
            ms_kwargs["CHAN_FREQ"],
            term_kwargs[f"{self.name}_param_freq_map"],
        )

        # gain_tuple spans from the different gain types, here we are only \
        # considering one gain type (delay_and_tec).
        gain_tuple = (gains,)
        #tuples required for time and frequency maps
        corrected_data = compute_corrected_residual(
            data, gain_tuple, a1, a2, (t_map,), (term_kwargs[f"{term_spec.name}_freq_map"],), \
            dir_maps, row_map, row_weights, n_corr
        )

        plt.figure()
        _sel = np.where((t_map == 0) & (a1 != a2))
        for foo in range(len(_sel[0])):
            plt.plot(np.angle(data[_sel][foo,:,0]))
        plt.title("Uncorrected")

        plt.figure()
        _sel = np.where((t_map == 0) & (a1 != a2))
        for foo in range(len(_sel[0])):
            plt.plot(np.angle(corrected_data[_sel][foo,:,0]))
        plt.title("Partial")
        plt.show()

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
        max_delta = 1/ dfreq
        nyq_rate = 1./ (2*(freq.max() - freq.min()))
        nbins = int(max_delta/ nyq_rate)

        if type == "k":
            nbins = max(2 * freq.size, 4096)  # Need adequate samples.
            fft_freq = np.fft.fftfreq(nbins, dfreq)
            fft_freq = np.fft.fftshift(fft_freq)
            #when not using finufft
            # fft_arr = np.abs(
            #     np.fft.fft(fsel_data, n=nbins, axis=1)
            # )
            # fft_arr = np.fft.fftshift(fft_arr, axes=1)
        elif type == "t":
            nbins = max(2 * freq.size, 4096)
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
