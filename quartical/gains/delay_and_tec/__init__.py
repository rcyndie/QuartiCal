import numpy as np
<<<<<<< HEAD
import finufft
from scipy.signal import medfilt
from scipy.ndimage import median_filter
=======
>>>>>>> upstream/main
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
<<<<<<< HEAD
from quartical.gains.general.generics import compute_corrected_residual


=======
>>>>>>> upstream/main

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


        #Before any parameter assignment, create an array to store the dominant peak selection.
        params_assigned = np.zeros(params.shape, dtype=np.int32)

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

                
                nonzero_count = np.count_nonzero(fsel_data, axis=(1, 2))

                #Set threshold on the number of nonzero entries along channels
                threshold0 = 0.6 #20% of visibilities are zero >> flagged
                param_flag_sel = np.where(nonzero_count<= threshold0*fsel_data.shape[1]*fsel_data.shape[2])
                param_flags[ut, uf, param_flag_sel, :] = 1
                gain_flags[ut, :, param_flag_sel, :] = 1

                #Do not flag the ref_ant.
                param_flags[ut, uf, ref_ant, :] = 0
                gain_flags[ut, :, ref_ant, :] = 0


                #Initialise array to contain delay and tec estimates
                delay_est = np.zeros((n_ant, n_paramk), dtype=np.float64)
                delay_est, fft_arrk, fft_freqk = self.initial_estimates(
                    fsel_data, delay_est, chan_freq, valid_ant, type="k"
                    )

                tec_est = np.zeros((n_ant, n_paramt), dtype=np.float64)
                tec_est, fft_arrt, fft_freqt = self.initial_estimates(
                    fsel_data, tec_est, invfreq, valid_ant, type="t"
                    )

                #preference coef towards TEC
                diff_tol = 0.


                #Array of zeros and assign to 1 when selecting peak.
                #Selecting the dominant peak and letting the other parameter as zero.
                for t, p, q in zip(t_map[sel], a1[sel], a2[sel]):
                    if p == ref_ant:
                        if n_corr == 1:
                            if np.max(np.abs(fft_arrk[q, :, 0])**2) > (1. + diff_tol) * np.max(np.abs(fft_arrt[q, :, 0])**2):
                                #delay is dominant >> only assign delay
                                params[t, uf, q, 0, 1] = -delay_est[q]
                                params_assigned[t, uf, q, 0, 1] = 1
                            else:
                                #tec is dominant >> only assign tec
                                params[t, uf, q, 0, 0] = -tec_est[q]
                                params_assigned[t, uf, q, 0, 0] = 1
                        elif n_corr > 1:
                            if np.max(np.abs(fft_arrk[q, :, 0])**2) > (1. + diff_tol) * np.max(np.abs(fft_arrt[q, :, 0])**2):
                                #only assign delay
                                params[t, uf, q, 0, 1] = -delay_est[q, 0]
                                params_assigned[t, uf, q, 0, 1] = 1
                            else:
                                #only assign tec
                                params[t, uf, q, 0, 0] = -tec_est[q, 0]
                                params_assigned[t, uf, q, 0, 0] = 1
                            
                            if np.max(np.abs(fft_arrk[q, :, 1])**2) > (1. + diff_tol) * np.max(np.abs(fft_arrt[q, :, 1])**2):
                                #only assign delay
                                params[t, uf, q, 0, 3] = -delay_est[q, 1]
                                params_assigned[t, uf, q, 0, 3] = 1
                            else:
                                #only assign tec
                                params[t, uf, q, 0, 2] = -tec_est[q, 1]
                                params_assigned[t, uf, q, 0, 2] = 1

                    else:
                        if n_corr == 1:
                            if np.max(np.abs(fft_arrk[p, :, 0])**2) > (1. + diff_tol) * np.max(np.abs(fft_arrt[p, :, 0])**2):
                                #delay is dominant >> only assign delay
                                params[t, uf, p, 0, 1] = delay_est[p]
                                params_assigned[t, uf, p, 0, 1] = 1
                            else:
                                #tec is dominant >> only assign tec
                                params[t, uf, p, 0, 0] = tec_est[p]
                                params_assigned[t, uf, p, 0, 0] = 1
                        elif n_corr > 1:
                            if np.max(np.abs(fft_arrk[p, :, 0])**2) > (1. + diff_tol) * np.max(np.abs(fft_arrt[p, :, 0])**2):
                                #only assign delay
                                params[t, uf, p, 0, 1] = delay_est[p, 0]
                                params_assigned[t, uf, p, 0, 1] = 1
                            else:
                                #only assign tec
                                params[t, uf, p, 0, 0] = tec_est[p, 0]
                                params_assigned[t, uf, p, 0, 0] = 1

                            if np.max(np.abs(fft_arrk[p, :, 1])**2) > (1. + diff_tol) * np.max(np.abs(fft_arrt[p, :, 1])**2):
                                #only assign delay
                                params[t, uf, p, 0, 3] = delay_est[p, 1]
                                params_assigned[t, uf, p, 0, 3] = 1
                            else:
                                #only assign tec
                                params[t, uf, p, 0, 2] = tec_est[p, 1]
                                params_assigned[t, uf, p, 0, 2] = 1

            
                # path00 = "/home/russeeawon/testing/791314_expts/expt2/"
                # path00 = "/home/russeeawon/testing/791314_expts/expt3/"
                # path00 = "/home/russeeawon/testing/791314_expts/expt4/"
                # path00 = "/home/russeeawon/testing/791516_expts/expt2/"
                path00 = "/home/russeeawon/testing/2002459_expts/expt2/"



                path01 = ""

                path0 = path00+path01


                np.save(path0+"delayest0_t{}.npy".format(ut), params[0, 0, :, 0, 1])
                np.save(path0+"delay_fftarr0_t{}.npy".format(ut), fft_arrk)
                np.save(path0+"delay_fft_freq0_t{}.npy".format(ut), fft_freqk)
                np.save(path0+"tecest0_t{}.npy".format(ut), params[0, 0, :, 0, 0])
                np.save(path0+"tec_fftarr0_t{}.npy".format(ut), fft_arrt)
                np.save(path0+"tec_fft_freq0_t{}.npy".format(ut), fft_freqt)


        delay_and_tec_params_to_gains(
            params,
            gains,
            ms_kwargs["CHAN_FREQ"],
            term_kwargs[f"{self.name}_param_freq_map"],
        )

<<<<<<< HEAD
        #Save the midway gains
        np.save(path0+"gains0.npy", gains)
        # np.save(path0+"data0.npy", data)
        np.save(path0+"params0.npy", params)

        
        # gain_tuple spans from the different gain types, here we are only \ 
        # considering one gain type (delay_and_tec).
        gain_tuple = (gains,)
        #tuples required for time and frequency maps 
        corrected_data = compute_corrected_residual(
            data, gain_tuple, a1, a2, (t_map,), (term_kwargs[f"{term_spec.name}_freq_map"],), \
            dir_maps, row_map, row_weights, n_corr
        )

        np.save(path0+"data1.npy", corrected_data)

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


                #Initialise array to contain delay and tec estimates
                
                delay_est = np.zeros((n_ant, n_paramk), dtype=np.float64)
                delay_est, fft_arrk, fft_freqk = self.initial_estimates(
                    fsel_data, delay_est, chan_freq, valid_ant, type="k"
                    )

                tec_est = np.zeros((n_ant, n_paramt), dtype=np.float64)
                tec_est, fft_arrt, fft_freqt = self.initial_estimates(
                    fsel_data, tec_est, invfreq, valid_ant, type="t"
                    )


                # select again!
                # Attempting to tweak the peak selection for the previously non-dominant peak
                for t, p, q in zip(t_map[sel], a1[sel], a2[sel]):
                    if p == ref_ant:
                        if n_corr == 1:
                            if params_assigned[t, uf, q, 0, 1] == 1: #delay was selected initially
                                #now select tec
                                params[t, uf, q, 0, 0] = -tec_est[q]
                            else:
                                params[t, uf, q, 0, 1] = -delay_est[q]

                        elif n_corr > 1:
                            if params_assigned[t, uf, q, 0, 1] == 1: #delay was selected initially
                                params[t, uf, q, 0, 0] = -tec_est[q, 0]
                            else:
                                params[t, uf, q, 0, 1] = -delay_est[q, 0]

                            
                            if params_assigned[t, uf, q, 0, 3] == 1: #delay was selected initially
                                params[t, uf, q, 0, 2] = -tec_est[q, 1]
                            else:
                                params[t, uf, q, 0, 3] = -delay_est[q, 1]

                    else:
                        if n_corr == 1:
                            if params_assigned[t, uf, p, 0, 1] == 1: #delay was selected initially
                                params[t, uf, p, 0, 0] = tec_est[p]
                            else:
                                params[t, uf, p, 0, 1] = delay_est[p]

                        elif n_corr > 1:
                            if params_assigned[t, uf, p, 0, 1] == 1:
                                params[t, uf, p, 0, 0] = tec_est[p, 0]
                            else:
                                params[t, uf, p, 0, 1] = delay_est[p, 0]
                            
                            if params_assigned[t, uf, p, 0, 3] == 1:
                                params[t, uf, p, 0, 2] = tec_est[p, 1]
                            else:
                                params[t, uf, p, 0, 3] = delay_est[p, 1]


                np.save(path0+"delayest1_t{}.npy".format(ut), params[0, 0, :, 0, 1])
                np.save(path0+"delay_fftarr1_t{}.npy".format(ut), fft_arrk)
                np.save(path0+"delay_fft_freq1_t{}.npy".format(ut), fft_freqk)
                np.save(path0+"tecest1_t{}.npy".format(ut), params[0, 0, :, 0, 0])
                np.save(path0+"tec_fftarr1_t{}.npy".format(ut), fft_arrt)
                np.save(path0+"tec_fft_freq1_t{}.npy".format(ut), fft_freqt)



        # for p in range(n_ant):
            #Check for any outliers along time axis.
            # params[:, 0, p, 0] = self.apply_outlier_filter(params[:, 0, p, 0], sel_filter="lw_up_lim", interp=False)
            # params[:, 0, p, 0] = self.apply_outlier_filter(params[:, 0, p, 0], sel_filter="lw_up_lim", interp=True)

        #     #Skip last index when assigning param_flags
        #     param_flag_sel_withfilter = np.where(np.isnan(params[:, 0, p, 0]).any(axis=1))
        #     #Flag any NaN
        #     param_flags[param_flag_sel_withfilter, 0, p, 0] = 1
        #     gain_flags[param_flag_sel_withfilter, :, p, 0] = 1


        #Do not flag the ref_ant.
        # param_flags[:, :, ref_ant, :] = 0
        # gain_flags[:, :, ref_ant, :] = 0


        #Choose a window size that must be odd.
        window_size = 11
        run_median_filter = False
        # run_median_filter = True

        if run_median_filter:
            for p in range(n_ant):
                for par in range(params.shape[-1]):
                    par_copy = params[:, 0, p, 0, par].copy()
                    # par_real = medfilt(par_copy.real, kernel_size=window_size)
                    # par_imag = medfilt(par_copy.imag, kernel_size=window_size)
                    # params[:, 0, p, 0, par] = par_real + 1j*par_imag

                    params[:, 0, p, 0, par] = median_filter(par_copy, size=window_size, mode="reflect")



        apply_param_flags_to_params(param_flags, params, 0)
        apply_gain_flags_to_gains(gain_flags, gains)

        delay_and_tec_params_to_gains(
            params,
            gains,
            ms_kwargs["CHAN_FREQ"],
            term_kwargs[f"{self.name}_param_freq_map"],
        )

        #Save as no-solve gains
        np.save(path0+"gains1.npy", gains)
        np.save(path0+"params1.npy", params)



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
            nbins = 6*nbins
            fft_freq = np.fft.fftfreq(nbins, dfreq)
            fft_freq = np.fft.fftshift(fft_freq)

            #when not using finufft
            # fft_arr = np.abs(
            #     np.fft.fft(fsel_data, n=nbins, axis=1)
            # )
            # fft_arr = np.fft.fftshift(fft_arr, axes=1)
        elif type == "t":
            nbins = 6*nbins
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

            #Normalising the data column with respect to the amplitude.
            if np.abs(datak).all() != 0:
                datak = datak/np.abs(datak)

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


    
    def select_outlier_filter(self, params, sel_filter="mad", label0="K"):
        """
        Use this function to select the outlier filter to return a mask.

        """

        if sel_filter == "lw_up_lim": #filter1
            ##Apply a lower and upper limit on the parameter values.
            if label0 == "K":
                lim = 0.1e-7
            elif label0 == "T":
                lim = 1e8
            #Return a mask that identifies outliers.
            return  np.abs(params) >= lim

        elif sel_filter == "mad": #filter2
            #Outlier sensitivity threshold (higher=less strict)
            outlier_threshold_fac = 8

            return np.abs(params-np.median(params)) >= \
                outlier_threshold_fac*np.median(np.abs(params-np.median(params)))
        else:
            return NotImplementedError
        
        return mask
    

    def apply_outlier_filter(self, params, sel_filter, interp=False):
        """
        Use this function to apply the selected filter.
        In this case, params is of shape (ntint, nparam).

        """


        params_d0 = params[:, 1]
        params_t0 = params[:, 0]

        mask_d0 = self.select_outlier_filter(params_d0, sel_filter, label0="K")
        mask_t0 = self.select_outlier_filter(params_t0, sel_filter, label0="T")

        if interp:
            delay_outlier_ind0 = np.where(mask_d0)[0]
            valid_delay_ind0 = np.where(~mask_d0)[0]
            if np.sum(mask_d0) != 0 and np.sum(valid_delay_ind0) != 0:
                params_d0[:] = \
                    np.interp(np.arange(params_d0.size).astype(np.float64), valid_delay_ind0.astype(np.float64), params_d0[valid_delay_ind0])

            tec_outlier_ind0 = np.where(mask_t0)[0]
            valid_tec_ind0 = np.where(~mask_t0)[0]
            if np.sum(mask_t0) != 0 and np.sum(valid_tec_ind0) != 0:
                params_t0[:] = \
                    np.interp(np.arange(params_t0.size).astype(np.float64), valid_tec_ind0.astype(np.float64), params_t0[valid_tec_ind0])
        
        else:
            params_d0[mask_d0] = np.nan
            params_t0[mask_t0] = np.nan
        

        if params.shape[1] == 4:
            params_d1 = params[:, 3]
            params_t1 = params[:, 2]

            mask_d1 = self.select_outlier_filter(params_d1, sel_filter, label0="K")
            mask_t1 = self.select_outlier_filter(params_t1, sel_filter, label0="T")

            if interp:
                delay_outlier_ind1 = np.where(mask_d1)[0]
                valid_delay_ind1 = np.where(~mask_d1)[0]
                if np.sum(mask_d1) != 0 and np.sum(valid_delay_ind1) != 0:
                    params_d1[:] = \
                        np.interp(np.arange(params_d1.size).astype(np.float64), valid_delay_ind1.astype(np.float64), params_d1[valid_delay_ind1])

             
                tec_outlier_ind1 = np.where(mask_t1)[0]
                valid_tec_ind1 = np.where(~mask_t1)[0]
                if np.sum(mask_t1) != 0 and np.sum(valid_tec_ind1) != 0:
                    params_t1[:] = \
                        np.interp(np.arange(params_t1.size).astype(np.float64), valid_tec_ind1.astype(np.float64), params_t1[valid_tec_ind1])


            else:
                params_d1[mask_d1] = np.nan
                params_t1[mask_t1] = np.nan

        return params
=======
        apply_param_flags_to_params(param_flags, params, 0)
        apply_gain_flags_to_gains(gain_flags, gains)

        return gains, gain_flags, params, param_flags
>>>>>>> upstream/main
