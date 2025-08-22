import numpy as np
import finufft
from scipy.signal import medfilt
from scipy.ndimage import median_filter
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


    def init_term(self, term_spec, ref_ant, ms_kwargs, term_kwargs, meta=None):
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


        #Initialise array to contain delay and tec estimates
        #shape (ntint, nant, ??) (only one delay and one TEC per correlation)
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


        delay_arr = np.zeros((params.shape[0], n_ant, n_paramk), dtype=np.float64)
        tec_arr = np.zeros((params.shape[0], n_ant, n_paramt), dtype=np.float64)


        #Choose number of subbands (split the entire bandwidth accordingly).
        n_subint = 2

        
        #Evaluate the number of channels per subband.
        chan_per_subint = int(np.ceil(n_chan / n_subint))


        #Also, initialise the gradients array.        
        subint_est = np.empty((n_tint, n_fint, n_subint, n_ant, n_corr))
        gradients = np.empty((n_subint))



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
                fsel_chan = chan_freq[fsel]

                #Used to normalise freq
                scale_factor = 1e9
                fsel_chan *= 1/scale_factor

                fsel_data = ref_data[:, fsel]
                valid_ant = fsel_data.any(axis=(1, 2))


                nonzero_count = np.count_nonzero(fsel_data, axis=(1, 2))


                #Set threshold on the number of nonzero entries along channels
                threshold0 = 0.6 #20% of visibilities are zero >> flagged
                param_flag_sel = np.where(nonzero_count<= threshold0*fsel_data.shape[1]*fsel_data.shape[2])
                param_flags[ut, uf, param_flag_sel, :] = 1
                gain_flags[ut, :, param_flag_sel, :] = 1

                #Do not flag the ref_ant.
                param_flags[ut, uf, ref_ant, :] = 0
                gain_flags[ut, :, ref_ant, :] = 0

                #number of channels per subband
                subint_stride = int(np.ceil(fsel_nchan / n_subint))


                for i, si in enumerate(range(0, fsel_nchan, subint_stride)):
                    si_sel = slice(si, si + subint_stride)
                    subint_data = fsel_data[:, si_sel]

                    # # NOTE: Collapse correlation axis when term is scalar.
                    # if self.scalar:
                    #     subint_data[..., :] = subint_data.sum(
                    #         axis=-1, keepdims=True
                    #     )

                    subint_freq = fsel_chan[si_sel]
                    dfreq = np.abs(subint_freq[-2]-subint_freq[-1])


                    #estimate-resolution determines the accuracy of how close do we want the initial estimates \
                    #to be close to the correct wrap.
                    est_resolution = 0.001
                    max_n_wrap = 1 / (2 * dfreq) * (subint_freq[-1] - subint_freq[0])
                    nbins = int((2 * max_n_wrap) / est_resolution)
                    fft_freq = np.fft.fftfreq(nbins, dfreq)



                    for p in range(n_ant):
                        corr_indices = [0] if n_corr == 1 else [0, n_corr - 1]
    
                        for c in corr_indices:
                            datac = subint_data[p, :, c]

                            fft_arr = np.fft.fft(datac, axis=0, n=nbins)
                            #shape of subint est array << subint_est = np.empty((n_tint, n_fint, n_subint, n_ant, n_corr))
                            subint_est[ut, uf, i, p, c] = fft_freq[np.argmax(np.abs(fft_arr), axis=0)]



                    # Zero the reference antenna/antennas without data.
                    subint_est[ut, uf, :, ~valid_ant] = 0


                    #Also, calculate the gradients generated when fitting the line 1/f = fm + c
                    gradients[i], _ = np.polyfit(subint_freq, 1/subint_freq, deg=1)



                #Obtain optimal value of delay and TEC in the least-squares sense.
                A = np.ones((n_subint, 2), dtype=np.float64)
                A[:, 1] = gradients
                # ATAinv = np.linalg.inv(A.T @ A)
                
                #A is found to be singular; use the following instead where lambda helps to regularise the problem
                # lambda = 1e-6
                ATAinv = np.linalg.inv(A.T @ A + 1e-6*np.eye(A.shape[1]))
                ATAinvAT = ATAinv @ A.T


                for p in range(n_ant):
                    for k in range(n_paramk):
                        if k == 0:
                            c = 0
                        elif k == 1:
                            c = -1
                        b = subint_est[ut, uf, :, p, c]
                        x = np.matmul(ATAinvAT, b[..., None])[..., 0]  # Remove trailing dim.


                        #Store in respective arrays.
                        #shape of delay_arr << np.zeros((params.shape[0], n_ant, n_paramt), dtype=np.float64)
                        delay_arr[ut, p, k] = x[0]/ scale_factor
                        tec_arr[ut, p, k] = x[1] * scale_factor
                    


        for p, q in zip(a1[sel], a2[sel]):
            if p == ref_ant:
                if n_corr == 1:
                    params[:, :, q, 0, 1] = -delay_arr[:, q]
                    params[:, :, q, 0, 0] = -tec_arr[:, q]

                elif n_corr > 1:
                    params[:, :, q, 0, 1] = -delay_arr[:, q, 0][:, np.newaxis]
                    params[:, :, q, 0, 0] = -tec_arr[:, q, 0][:, np.newaxis]

                    params[:, :, q, 0, 3] = -delay_arr[:, q, 1][:, np.newaxis]
                    params[:, :, q, 0, 2] = -tec_arr[:, q, 1][:, np.newaxis]

            else:
                if n_corr == 1:
                    params[:, :, p, 0, 1] = delay_arr[:, p]
                    params[:, :, p, 0, 0] = tec_arr[:, p]

                elif n_corr > 1:
                    params[:, :, p, 0, 1] = delay_arr[:, p, 0][:, np.newaxis]
                    params[:, :, p, 0, 0] = tec_arr[:, p, 0][:, np.newaxis]

                    params[:, :, p, 0, 3] = delay_arr[:, p, 1][:, np.newaxis]
                    params[:, :, p, 0, 2] = tec_arr[:, p, 1][:, np.newaxis]


    
        delay_and_tec_params_to_gains(
            params,
            gains,
            ms_kwargs["CHAN_FREQ"],
            term_kwargs[f"{self.name}_param_freq_map"],
        )


        return gains, gain_flags, params, param_flags
