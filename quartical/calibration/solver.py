# -*- coding: utf-8 -*-
import gc
import numpy as np
from collections import namedtuple
from itertools import cycle
from quartical.gains import term_types


meta_args_nt = namedtuple("meta_args_nt", ("iters", "active_term"))


def solver_wrapper(term_spec_list, solver_opts, gain_opts, **kwargs):
    """A Python wrapper for the solvers written in Numba.

    This wrapper facilitates getting values in and out of the Numba code and
    creates a dictionary of results which can be understood by the calling
    Dask code.

    Args:
        **kwargs: A dictionary of keyword arguments. All arguments are given
                  as key: value pairs to handle variable input.

    Returns:
        results_dict: A dictionary containing the results of the solvers.
    """

    gain_tup = ()
    param_tup = ()
    flag_tup = ()
    results_dict = {}

    for (term_name, term_type, term_shape, term_pshape) in term_spec_list:
        gain = np.zeros(term_shape, dtype=np.complex128)

        # Check for initialisation data.
        if f"{term_name}_initial_gain" in kwargs:
            gain[:] = kwargs[f"{term_name}_initial_gain"]
        else:
            gain[..., (0, -1)] = 1  # Set first and last correlations to 1.

        flag = np.zeros(term_shape, dtype=np.uint8)
        param = np.zeros(term_pshape, dtype=gain.real.dtype)

        gain_tup += (gain,)
        flag_tup += (flag,)
        param_tup += (param,)

        results_dict[f"{term_name}-gain"] = gain
        results_dict[f"{term_name}-flag"] = flag
        results_dict[f"{term_name}-param"] = param
        results_dict[f"{term_name}-conviter"] = 0
        results_dict[f"{term_name}-convperc"] = 0

    kwargs["gains"] = gain_tup
    kwargs["flags"] = flag_tup
    kwargs["inverse_gains"] = tuple([np.empty_like(g) for g in gain_tup])
    kwargs["params"] = param_tup

    terms = solver_opts.terms
    iter_recipe = solver_opts.iter_recipe

    for term, iters in zip(cycle(terms), iter_recipe):

        if iters == 0:
            continue

        active_term = terms.index(term)
        term_name, term_type, _, _ = term_spec_list[active_term]

        term_type_cls = term_types[term_type]

        solver = term_type_cls.solver
        base_args_tup = term_type_cls.base_args
        term_args_tup = term_type_cls.term_args

        base_args = \
            base_args_tup(**{k: kwargs[k] for k in base_args_tup._fields})
        term_args = \
            term_args_tup(**{k: kwargs[k] for k in term_args_tup._fields})
        meta_args = meta_args_nt(iters, active_term)

        info_tup = solver(base_args,
                          term_args,
                          meta_args,
                          kwargs["corr_mode"])

        results_dict[f"{term_name}-conviter"] += \
            np.atleast_2d(info_tup.conv_iters)
        results_dict[f"{term_name}-convperc"] += \
            np.atleast_2d(info_tup.conv_perc)

    gc.collect()

    return results_dict
