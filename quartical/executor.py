# -*- coding: utf-8 -*-
# Sets up logger - hereafter import logger from Loguru.
import quartical.logging.init_logger  # noqa
from loguru import logger
from quartical.parser import parser, preprocess
from quartical.data_handling.ms_handler import (read_xds_list,
                                                write_xds_list,
                                                preprocess_xds_list)
from quartical.data_handling.model_handler import add_model_graph
from quartical.calibration.calibrate import add_calibration_graph
from quartical.flagging.flagging import finalise_flags
import time
from dask.diagnostics import ProgressBar
import dask
from dask.distributed import Client, LocalCluster


@logger.catch
def execute():
    """Runs the application."""

    opts = parser.parse_inputs()

    # Add this functionality - should check opts for problems in addition
    # to interpreting weird options. Can also raise flags for different modes
    # of operation. The idea is that all our configuration state lives in this
    # options dictionary. Down with OOP!

    # TODO: There needs to be a validation step which checks that the config is
    # possible.

    preprocess.interpret_model(opts)

    if opts.parallel_scheduler == "distributed":
        logger.info("Initializing distributed client.")
        cluster = LocalCluster(processes=opts.parallel_nworker > 1,
                               n_workers=opts.parallel_nworker,
                               threads_per_worker=opts.parallel_nthread,
                               memory_limit=0)
        client = Client(cluster) # noqa
        logger.info("Distributed client sucessfully initialized.")

    t0 = time.time()

    # Reads the measurement set using the relavant configuration from opts.
    ms_xds_list, col_kwrds = read_xds_list(opts)

    # ms_xds_list = ms_xds_list[:2]

    # Preprocess the xds_list - initialise some values and fix bad data.
    preprocessed_xds_list = preprocess_xds_list(ms_xds_list, col_kwrds, opts)

    # Model xds is a list of xdss onto which appropriate model data has been
    # assigned.
    model_xds_list = add_model_graph(preprocessed_xds_list, opts)

    gains_per_xds, post_gain_xds = \
        add_calibration_graph(model_xds_list, col_kwrds, opts)

    writable_xds = finalise_flags(post_gain_xds, col_kwrds, opts)

    writes = write_xds_list(writable_xds, col_kwrds, opts)

    import zarr
    store = zarr.DirectoryStore("cc_gains")

    for g_name, g_list in gains_per_xds.items():
        for g_ind, g in enumerate(g_list):
            g_list[g_ind] = g.to_zarr(store,
                                      mode="w",
                                      group="{}{}".format(g_name, g_ind),
                                      compute=False)

    writes = [writes] if not isinstance(writes, list) else writes

    outputs = []
    for ind in range(len(writes)):
        output = []
        for key in gains_per_xds.keys():
            output.append(gains_per_xds[key][ind])
        output.append(writes[ind])
        outputs.append(output)

    logger.success("{:.2f} seconds taken to build graph.", time.time() - t0)

    t0 = time.time()

    with ProgressBar():
        dask.compute([dask.delayed(tuple)(x) for x in outputs],
                     num_workers=opts.parallel_nthread,
                     optimize_graph=True,
                     scheduler=opts.parallel_scheduler)
    logger.success("{:.2f} seconds taken to execute graph.", time.time() - t0)

    # This code can be used to save gain xarray datasets imeediately. This is
    # much faster but requires the datasets to live in memory.

    # import zarr
    # store = zarr.DirectoryStore("cc_gains")

    # for g_name, g_list in gains.items():
    #     for g_ind, g in enumerate(g_list):
    #         g.to_zarr("{}{}".format(g_name, g_ind),
    #                   mode="w")

    # This code can be used to save gain xarray datasets using delayed. This
    # is currently slower than it should be.

    # import zarr
    # store = zarr.DirectoryStore("cc_gains")

    # for g_name, g_list in gains_per_xds.items():
    #     for g_ind, g in enumerate(g_list):
    #         g_list[g_ind] = g.to_zarr(store,
    #                                   mode="w",
    #                                   group="{}{}".format(g_name, g_ind),
    #                                   compute=False)

    # import numpy as np
    # for gain in gains["G"]:
    #     print(np.max(np.abs(gain)))
    #     np.save("example_gains.npy", gain)
    #     break
    # for gain in gains[0]["dE"]:
    #     print(np.max(np.abs(gain)))

    # dask.visualize([xds.MODEL_DATA.data for xds in model_xds_list],
    #                color='order', cmap='autumn',
    #                filename='model_order.pdf', node_attr={'penwidth': '10'})

    # dask.visualize([xds.MODEL_DATA.data for xds in model_xds_list],
    #                filename='model.pdf',
    #                optimize_graph=False)

    # dask.visualize([dask.delayed(tuple)(x) for x in outputs],
    #                color='order', cmap='autumn',
    #                filename='graph_order.pdf', node_attr={'penwidth': '10'})

    # dask.visualize([dask.delayed(tuple)(x) for x in outputs],
    #                filename='graph.pdf',
    #                optimize_graph=True)

    # dask.visualize([dask.delayed(tuple)(x[0]) for x in outputs],
    #                color='order', cmap='autumn',
    #                filename='graph_order.pdf', node_attr={'penwidth': '10'})

    # dask.visualize([dask.delayed(tuple)(x[0]) for x in outputs],
    #                filename='graph.pdf',
    #                optimize_graph=True)

    # dask.visualize([dask.delayed(tuple)([x[:2]]) for x in outputs],
    #                color='order', cmap='autumn',
    #                filename='graph_order.pdf', node_attr={'penwidth': '10'})

    # dask.visualize([dask.delayed(tuple)([x[:2]]) for x in outputs],
    #                filename='graph.pdf',
    #                optimize_graph=False)

    # dask.visualize(model_xds_list,
    #                color='order', cmap='autumn',
    #                filename='model_order.pdf', node_attr={'penwidth': '10'})

    # dask.visualize(model_xds_list,
    #                filename='model.pdf',
    #                optimize_graph=False)