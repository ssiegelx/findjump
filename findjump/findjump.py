import os
import sys
from glob import glob
import pickle
import argparse
import time
import datetime
import gc

import numpy as np
import h5py
import pywt

from pychfpga import NameSpace, load_yaml_config
from calibration import utils

import log

from ch_util import tools, ephemeris, andata
from ch_util.fluxcat import FluxCatalog


###################################################
# default variables
###################################################

DEFAULTS = NameSpace(load_yaml_config(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                   'defaults.yaml') + ':findjump'))

LOG_FILE = os.environ.get('FINDJUMP_LOG_FILE',
           os.path.join(os.path.dirname(os.path.realpath(__file__)), 'findjump.log'))

DEFAULT_LOGGING = {
    'formatters': {
         'std': {
             'format': "%(asctime)s %(levelname)s %(name)s: %(message)s",
             'datefmt': "%m/%d %H:%M:%S"},
          },
    'handlers': {
        'stderr': {'class': 'logging.StreamHandler', 'formatter': 'std', 'level': 'DEBUG'}
        },
    'loggers': {
        '': {'handlers': ['stderr'], 'level': 'INFO'}  # root logger

        }
    }


###################################################
# auxiliary routines
###################################################

def daytime_flag(time):

    flag = np.zeros(time.size, dtype=np.bool)
    rise = ephemeris.solar_rising(time[0] - 24.0 * 3600.0, end_time=time[-1])
    for rr in rise:
        ss = ephemeris.solar_setting(rr)[0]
        flag |= ((time >= rr) & (time <= ss))

    return flag

def transit_flag(name, time, nsig=1.0, freq=400.0, pol='X'):

    flag = np.zeros(time.size, dtype=np.bool)

    if name.lower() == 'sun':

        window_sec = nsig * utils.get_window(freq, pol=pol, dec=np.radians(23.45), deg=False)

        ttrans = ephemeris.solar_transit(time[0] - 24.0 * 3600.0, time[-1] + 24.0 * 3600.0)

    else:

        src = FluxCatalog[name].skyfield

        window_sec = nsig * utils.get_window(freq, pol=pol, dec=np.radians(FluxCatalog[name].dec), deg=False)

        ttrans = ephemeris.transit_times(src, time[0] - 24.0 * 3600.0, time[-1] + 24.0 * 3600.0)

    for tt in ttrans:

        # Flag +/- window_sec around peak location
        begin = tt - window_sec
        end = tt + window_sec
        flag |= ((time >= begin) & (time <= end))

    return flag


def sliding_window(arr, window):

    # Advanced numpy tricks
    shape = arr.shape[:-1] + (arr.shape[-1]-window+1, window)
    strides = arr.strides + (arr.strides[-1],)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


def mod_max_finder(scale, coeff, search_span=0.5, threshold=None):

    # Parse input
    nscale, ntime = coeff.shape

    # Calculate modulus of wavelet transform
    mod = np.abs(coeff)

    # Flag local maxima of the modulus for each wavelet scale
    flg_mod_max = np.zeros(mod.shape, dtype=np.bool)

    for ss, sca in enumerate(scale):

        srch = max(int(search_span * sca), 1)

        slide_mod = sliding_window(mod[ss, :], 2*srch + 1)

        slc = slice(srch, ntime-srch)

        flg_mod_max[ss, slc] = np.all(mod[ss, slc, np.newaxis] >= slide_mod, axis=-1)

    # If requested, place threshold on modulus maxima
    if threshold is not None:
        flg_mod_max &= (mod > threshold)

    # Create array that only contains the modulus maxima (zero elsewhere)
    mod_max = mod * flg_mod_max.astype(np.float64)

    # Return flag and value arrays
    return flg_mod_max, mod_max


def finger_finder(scale, flag, mod_max, istart=3, do_fill=False):

    nscale, ntime = flag.shape

    icandidate = np.flatnonzero(flag[istart, :])
    ncandidate = icandidate.size

    if ncandidate == 0:
        return [None] * 6

    candidates = np.zeros((nscale, ncandidate), dtype=np.int) - 1
    candidates[istart, :] = icandidate

    isort = np.argsort(mod_max[istart, icandidate])[::-1]

    ss = istart + 1
    keep_iter = True
    while (ss < nscale) and keep_iter:

        wsearch = max(int(0.25 * scale[ss]), 1)

        ipc = list(np.flatnonzero(flag[ss, :]))

        for cc in isort:

            cand = candidates[ss-1, cc]

            if len(ipc) > 0:

                diff = [np.abs(ii - cand) for ii in ipc]

                best_match = ipc[np.argmin(diff)]

                if diff[np.argmin(diff)] <= wsearch:

                    candidates[ss, cc] = best_match

                    ipc.remove(best_match)


        iremain = np.flatnonzero(candidates[ss, :] >= 0)

        if iremain.size > 0:
            isort = iremain[np.argsort(mod_max[ss, candidates[ss, iremain]])[::-1]]
        else:
            keep_iter = False

        ss += 1

    # Fill in values below istart
    if do_fill:
        candidates[0:istart, :] = candidates[istart, np.newaxis, :]

    # Create ancillarly information
    start = np.zeros(ncandidate, dtype=np.int)
    stop = np.zeros(ncandidate, dtype=np.int)
    pdrift = np.zeros(ncandidate, dtype=np.float32)

    cmm = np.zeros((nscale, ncandidate), dtype=mod_max.dtype) * np.nan

    lbl = np.zeros((nscale, ntime), dtype=np.int) * np.nan
    lbl[flag] = -1

    for cc, index in enumerate(candidates.T):

        good_scale = np.flatnonzero(index >= 0)
        start[cc] = good_scale.min()
        stop[cc] = good_scale.max()

        pdrift[cc] = np.sqrt(np.sum((index[good_scale] - index[good_scale.min()])**2) / float(good_scale.size))

        for gw, igw in zip(good_scale, index[good_scale]):

            lbl[gw, igw] = cc
            cmm[gw, cc] = mod_max[gw, igw]


    # Return all information
    return candidates, cmm, pdrift, start, stop, lbl


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

###################################################
# main routine
###################################################

def main(config_file=None, logging_params=DEFAULT_LOGGING):

    # Setup logging
    log.setup_logging(logging_params)
    mlog = log.get_logger(__name__)

    # Set config
    config = DEFAULTS.deepcopy()
    if config_file is not None:
        config.merge(NameSpace(load_yaml_config(config_file)))

    # Set niceness
    current_niceness = os.nice(0)
    os.nice(config.niceness - current_niceness)
    mlog.info('Changing process niceness from %d to %d.  Confirm:  %d' %
                  (current_niceness, config.niceness, os.nice(0)))

    # Create output suffix
    output_suffix = config.output_suffix if config.output_suffix is not None else "jumps"

    # Calculate the wavelet transform for the following scales
    nwin = 2 * config.max_scale + 1
    nhwin = nwin // 2

    if config.log_scale:
        mlog.info("Using log scale.")
        scale = np.logspace(np.log10(config.min_scale), np.log10(nwin), num=config.num_points, dtype=np.int)
    else:
        mlog.info("Using linear scale.")
        scale = np.arange(config.min_scale, nwin, dtype=np.int)

    # Loop over acquisitions
    for acq in config.acq:

        # Find acquisition files
        all_data_files = sorted(glob(os.path.join(config.data_dir, acq, "*.h5")))
        nfiles = len(all_data_files)

        if nfiles == 0:
            continue

        mlog.info("Now processing acquisition %s (%d files)" % (acq, nfiles))

        # Determine list of feeds to examine
        dset = ['flags/inputs'] if config.use_input_flag else ()

        rdr = andata.CorrData.from_acq_h5(all_data_files, datasets=dset,
                                          apply_gain=False, renormalize=False)

        inputmap = tools.get_correlator_inputs(ephemeris.unix_to_datetime(rdr.time[0]),
                                               correlator='chime')

        # Extract good inputs
        if config.use_input_flag:
            ifeed = np.flatnonzero((np.sum(rdr.flags['inputs'][:], axis=-1, dtype=np.int) /
                                     float(rdr.flags['inputs'].shape[-1])) > config.input_threshold)
        else:
            ifeed = np.array([ii for ii, inp in enumerate(inputmap) if tools.is_chime(inp)])

        ninp = len(ifeed)

        mlog.info("Processing %d feeds." % ninp)

        # Create list of candidates
        cfreq, cinput, ctime, cindex = [], [], [], []
        jump_flag, jump_time, jump_auto = [], [], []
        ncandidate = 0

        # Determine number of files to process at once
        if config.max_num_file is None:
            chunk_size = nfiles
        else:
            chunk_size = min(config.max_num_file, nfiles)

        # Loop over chunks of files
        for chnk, data_files in enumerate(chunks(all_data_files, chunk_size)):

            mlog.info("Now processing chunk %d (%d files)" % (chnk, len(data_files)))

            # Deteremine selections along the various axes
            rdr = andata.CorrData.from_acq_h5(data_files, datasets=())

            auto_sel = np.array([ii for ii, pp in enumerate(rdr.prod) if pp[0] == pp[1]])
            auto_sel = andata._convert_to_slice(auto_sel)

            if config.time_start is None:
                ind_start = 0
            else:
                time_start = ephemeris.datetime_to_unix(datetime.datetime(*config.time_start))
                ind_start = int(np.argmin(np.abs(rdr.time - time_start)))

            if config.time_stop is None:
                ind_stop = rdr.ntime
            else:
                time_stop = ephemeris.datetime_to_unix(datetime.datetime(*config.time_stop))
                ind_stop = int(np.argmin(np.abs(rdr.time - time_stop)))

            if config.freq_physical is not None:

                if hasattr(config.freq_physical, '__iter__'):
                    freq_physical = config.freq_physical
                else:
                    freq_physical = [config.freq_physical]

                freq_sel = [np.argmin(np.abs(ff - rdr.freq)) for ff in freq_physical]
                freq_sel = andata._convert_to_slice(freq_sel)

            else:
                fstart = config.freq_start if config.freq_start is not None else 0
                fstop = config.freq_stop if config.freq_stop is not None else rdr.freq.size
                freq_sel = slice(fstart, fstop)

            # Load autocorrelations
            t0 = time.time()
            data = andata.CorrData.from_acq_h5(data_files, datasets=['vis'], start=ind_start, stop=ind_stop,
                                                           freq_sel=freq_sel, prod_sel=auto_sel,
                                                           apply_gain=False, renormalize=False)

            mlog.info("Took %0.1f seconds to load autocorrelations." % (time.time() - t0,))

            # If first chunk, save the frequencies that are being used
            if not chnk:
                all_freq = data.freq.copy()

            # If requested do not consider data during day or near bright source transits
            flag_quiet = np.ones(data.ntime, dtype=np.bool)
            if config.ignore_sun:
                flag_quiet &= ~transit_flag('sun', data.time, freq=np.min(data.freq), pol='X', nsig=1.0)

            if config.only_quiet:
                flag_quiet &= ~daytime_flag(data.time)
                for ss in ["CYG_A", "CAS_A", "TAU_A", "VIR_A"]:
                    flag_quiet &= ~transit_flag(ss, data.time, freq=np.min(data.freq), pol='X', nsig=1.0)

            # Loop over frequencies
            for ff, freq in enumerate(data.freq):

                print_cnt = 0
                mlog.info("FREQ %d (%0.2f MHz)" % (ff, freq))

                auto = data.vis[ff, :, :].real

                fractional_auto = auto * tools.invert_no_zero(np.median(auto, axis=-1, keepdims=True)) - 1.0

                # Loop over inputs
                for ii in ifeed:

                    print_cnt += 1
                    do_print = not (print_cnt % 100)

                    if do_print:
                        mlog.info("INPUT %d" % ii)
                    t0 = time.time()

                    signal = fractional_auto[ii, :]

                    # Perform wavelet transform
                    coef, freqs = pywt.cwt(signal, scale, config.wavelet_name)

                    if do_print:
                        mlog.info("Took %0.1f seconds to perform wavelet transform." % (time.time() - t0,))
                    t0 = time.time()

                    # Find local modulus maxima
                    flg_mod_max, mod_max = mod_max_finder(scale, coef, threshold=config.thresh, search_span=config.search_span)

                    if do_print:
                        mlog.info("Took %0.1f seconds to find modulus maxima." % (time.time() - t0,))
                    t0 = time.time()

                    # Find persisent modulus maxima across scales
                    candidates, cmm, pdrift, start, stop, lbl = finger_finder(scale, flg_mod_max, mod_max,
                                                                              istart=max(config.min_rise - config.min_scale, 0),
                                                                              do_fill=False)

                    if do_print:
                        mlog.info("Took %0.1f seconds to find fingers." % (time.time() - t0,))
                    t0 = time.time()

                    if candidates is None:
                        continue

                    # Cut bad candidates
                    index_good_candidates = np.flatnonzero((scale[stop] >= config.max_scale) &
                                                            flag_quiet[candidates[start, np.arange(start.size)]] &
                                                            (pdrift <= config.psigma_max))

                    ngood = index_good_candidates.size

                    if ngood == 0:
                        continue

                    mlog.info("Input %d has %d jumps" % (ii, ngood))

                    # Add remaining candidates to list
                    ncandidate += ngood

                    cfreq += [freq] * ngood
                    cinput += [ii] * ngood

                    for igc in index_good_candidates:

                        icenter = candidates[start[igc], igc]

                        cindex.append(icenter)
                        ctime.append(data.time[icenter])

                        aa = max(0, icenter - nhwin)
                        bb = min(data.ntime, icenter + nhwin + 1)

                        ncut = bb - aa

                        temp_var = np.zeros(nwin, dtype=np.bool)
                        temp_var[0:ncut] = True
                        jump_flag.append(temp_var)

                        temp_var = np.zeros(nwin, dtype=data.time.dtype)
                        temp_var[0:ncut] = data.time[aa:bb].copy()
                        jump_time.append(temp_var)

                        temp_var = np.zeros(nwin, dtype=auto.dtype)
                        temp_var[0:ncut] = auto[ii, aa:bb].copy()
                        jump_auto.append(temp_var)


            # Garbage collect
            del data
            gc.collect()

        # If we found any jumps, write them to a file.
        if ncandidate > 0:

            output_file = os.path.join(config.output_dir, "%s_%s.h5" % (acq, output_suffix))

            mlog.info("Writing %d jumps to: %s" % (ncandidate, output_file))

            # Write to output file
            with h5py.File(output_file, 'w') as handler:

                handler.attrs['files'] = all_data_files
                handler.attrs['chan_id'] = ifeed
                handler.attrs['freq'] = all_freq

                index_map = handler.create_group('index_map')
                index_map.create_dataset('jump', data=np.arange(ncandidate))
                index_map.create_dataset('window', data=np.arange(nwin))

                ax = np.array(['jump'])

                dset = handler.create_dataset('freq', data=np.array(cfreq))
                dset.attrs['axis'] = ax

                dset = handler.create_dataset('input', data=np.array(cinput))
                dset.attrs['axis'] = ax

                dset = handler.create_dataset('time', data=np.array(ctime))
                dset.attrs['axis'] = ax

                dset = handler.create_dataset('time_index', data=np.array(cindex))
                dset.attrs['axis'] = ax


                ax = np.array(['jump', 'window'])

                dset = handler.create_dataset('jump_flag', data=np.array(jump_flag))
                dset.attrs['axis'] = ax

                dset = handler.create_dataset('jump_time', data=np.array(jump_time))
                dset.attrs['axis'] = ax

                dset = handler.create_dataset('jump_auto', data=np.array(jump_auto))
                dset.attrs['axis'] = ax

        else:
            mlog.info("No jumps found for %s acquisition." % acq)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config',   help='Name of configuration file.',
                                      type=str, default=None)
    parser.add_argument('--log',      help='Name of log file.',
                                      type=str, default=LOG_FILE)

    args = parser.parse_args()

    # If calling from the command line, then send logging to log file instead of screen
    try:
        os.makedirs(os.path.dirname(args.log))
    except OSError:
        if not os.path.isdir(os.path.dirname(args.log)):
            raise

    logging_params = DEFAULT_LOGGING
    logging_params['handlers'] = {'stderr': {'class': 'logging.handlers.WatchedFileHandler',
                                             'filename': args.log, 'formatter': 'std', 'level': 'INFO'}}

    # Call main routine
    main(config_file=args.config, logging_params=logging_params)