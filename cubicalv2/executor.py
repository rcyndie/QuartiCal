# -*- coding: utf-8 -*-
# Sets up logger - hereafter import logger from Loguru.
import cubicalv2.logging.init_logger  # noqa
from loguru import logger
from cubicalv2.parser import parser, preprocess
from cubicalv2.data_handling import data_handler
from cubicalv2.calibration.calibrate import calibrate
from cubicalv2.data_handling.predict import predict
import dask.array as da
import time
from dask.diagnostics import ProgressBar
import dask
from dask.distributed import Client


@logger.catch
def execute():
    """Runs the application."""

    opts = parser.parse_inputs()

    # Add this functionality - should check opts for problems in addition
    # to interpreting weird options. Can also raise flags for different modes
    # of operation. The idea is that all our configuration state lives in this
    # options dictionary. Down with OOP!

    # There needs to be a validation step which checks that the config is
    # possible.

    preprocess.preprocess_opts(opts)

    if opts.parallel_scheduler == "distributed":
        client = Client(processes=False,                            # noqa
                        n_workers=opts.parallel_nworker,
                        threads_per_worker=opts.parallel_nthread)
        logger.info("Initializing distributed client.")

    # Give opts to the data handler, which returns a list of xarray data sets.

    t0 = time.time()
    data_xds = data_handler.read_ms(opts)

    predict_xds = predict(data_xds, opts) if opts._predict else data_xds

    gains_per_xds, updated_data_xds = calibrate(predict_xds, opts)

    write_columns = data_handler.write_ms(updated_data_xds, opts)
    logger.success("{:.2f} seconds taken to build graph.", time.time() - t0)

    t0 = time.time()
    with ProgressBar():
        gains, _ = da.compute(gains_per_xds,
                              write_columns,
                              num_workers=opts.parallel_nthread)
    logger.success("{:.2f} seconds taken to execute graph.", time.time() - t0)

    # dask.visualize(gains_per_xds[0], filename='graph.pdf')
