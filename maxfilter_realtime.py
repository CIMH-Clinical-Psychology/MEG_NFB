# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 12:03:14 2021

script that contains a copy of the maxfilter functions that work in realtime

@author: Simon
"""
# Authors: Mark Wronkiewicz <wronk.mark@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#          Jussi Nurminen <jnu@iki.fi>


# License: BSD (3-clause)
from functools import wraps
# from collections import Counter, OrderedDict
# from functools import partial
# from math import factorial
# from os import path as op
# from deepdiff import DeepDiff
import numpy as np
import mne
import stimer
from mne.io.pick import (pick_types, pick_channels, pick_channels_regexp,
                      pick_info)
# from mne import __version__
# from mne.annotations import _annotations_starts_stops
# from mne.bem import _check_origin
# from mne.transforms import (_str_to_frame, _get_trans, Transform, apply_trans,
#                           _find_vector_rotation, _cart_to_sph, _get_n_moments,
#                           _sph_to_cart_partials, _deg_ord_idx, _average_quats,
#                           _sh_complex_to_real, _sh_real_to_complex, _sh_negate,
#                           quat_to_rot, rot_to_quat)
# from mne.forward import _concatenate_coils, _prep_meg_channels, _create_meg_coils
# from mne.surface import _normalize_vectors
# from mne.io.constants import FIFF, FWD
# from mne.io.meas_info import _simplify_info, Info
# from mne.io.proc_history import _read_ctc
# from mne.io.write import _generate_meas_id, DATE_NONE
# from mne.io import (_loc_to_coil_trans, _coil_trans_to_loc, BaseRaw, RawArray,
#                   Projection)
# from mne.utils import (verbose, logger, _clean_names, warn, _time_mask, _pl,
#                       _check_option, _ensure_int, _validate_type, use_log_level)
# from mne.fixes import _safe_svd, einsum, bincount
# from mne.channels.channels import _get_T1T2_mag_inds, fix_mag_coil_types
from mne.chpi import  _fit_chpi_amplitudes, _time_prefix, _fit_chpi_quat_subset
from mne.chpi import _get_hpi_initial_fit, _fit_magnetic_dipole, _check_chpi_param
from mne.io.pick import (pick_types, pick_channels, pick_channels_regexp,
                      pick_info)
from mne.forward import (_create_meg_coils,  _concatenate_coils)
from mne.dipole import _make_guesses
from mne.preprocessing.maxwell import (_sss_basis, _prep_mf_coils,
                                    _regularize_out, _get_mf_picks_fix_mags)
from mne.transforms import (apply_trans, invert_transform, _angle_between_quats,
                         quat_to_rot, rot_to_quat, _fit_matched_points,
                         _quat_to_affine)
from mne.utils import (verbose, logger, use_log_level, _check_fname, warn,
                    _validate_type, ProgressBar, _check_option)

global glob
glob = {'enable_caching' : False}

# Note: MF uses single precision and some algorithms might use
# truncated versions of constants (e.g., μ0), which could lead to small
# differences between algorithms
def cached(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not glob['enable_caching']: 
            print (f'########### NOT USING CACHE: [{func.__name__}]')
            return func(*args, **kwargs)
        if func.__name__ in glob:
            print (f'############ TAKE FROM CACHE: [{func.__name__}]')
            return glob[func.__name__]
        res = func(*args, **kwargs)
        glob[func.__name__] = res
        print(f'############ PUT INTO CACHE: [{func.__name__}]')
        return res
    return wrapper

_setup_hpi_amplitude_fitting = cached(mne.chpi._setup_hpi_amplitude_fitting)
compute_whitener = cached(mne.chpi.compute_whitener)
make_ad_hoc_cov = cached(mne.chpi.make_ad_hoc_cov)
_magnetic_dipole_field_vec = cached(mne.chpi._magnetic_dipole_field_vec)



class MaxFilterClient():
    
    def __init__(self, raw,  t_step_min=0.01, t_window='auto', ext_order=1,
                 tmin=0., tmax=None):
        
        self.t_step_min = t_step_min
        self.t_window = t_window
        self.ext_order = ext_order
        self.tmin = tmin
        self.tmax = tmax
        self.raw = raw
        self.preparation()


    def preparation(self):
        raw = self.raw.copy().crop(0,0.2)
        hpi = _setup_hpi_amplitude_fitting(raw.info, 
                                                self.t_window, 
                                                ext_order=self.ext_order)
        
        proj = hpi['proj']
        meg_picks = pick_channels(
            raw.info['ch_names'], proj['data']['col_names'], ordered=True)
        raw.info = pick_info(raw.info, meg_picks)  # makes a copy
        raw.info['projs'] = [proj]
        
        meg_coils = _concatenate_coils(_create_meg_coils(raw.info['chs'], 'accurate'))

        # Set up external model for interference suppression
        
        cov = make_ad_hoc_cov(raw.info, verbose=False)
        whitener, _ = compute_whitener(cov, raw.info, verbose=False)
        meg_picks = mne.pick_types(raw.info, meg=True)
        raw.info = pick_info(raw.info, meg_picks)  # makes a copy
        meg_coils = _concatenate_coils(_create_meg_coils(raw.info['chs'], 'accurate'))
        R = np.linalg.norm(meg_coils[0], axis=1).min()
        guesses = _make_guesses(dict(R=R, r0=np.zeros(3)), 0.01, 0., 0.005,
                                verbose=False)[0]['rr']
        fwd = _magnetic_dipole_field_vec(guesses, meg_coils, too_close='raise')
        fwd = np.dot(fwd, whitener.T)
        fwd.shape = (guesses.shape[0], 3, -1)
        fwd = np.linalg.svd(fwd, full_matrices=False)[2]
        guesses = dict(rr=guesses, whitened_fwd_svd=fwd)
        self.guesses = guesses
        self.whitener = whitener
        self.meg_coils = meg_coils
        self.hpi = hpi

    @profile
    def feed_cunk(self, raw, fit_time, fit_idx, t_step_max=1.,
                  too_close='raise', gof_limit=0.98, dist_limit=0.005):
        time_sl = fit_idx - self.hpi['n_window'] // 2
        time_sl = slice(max(time_sl, 0), time_sl + self.hpi['n_window'])
        last = self.last
        #
        # 1. Fit amplitudes for each channel from each of the N sinusoids
        #
        sin_fit = _fit_chpi_amplitudes(raw, time_sl, self.hpi)

        #%% static part def compute_chpi_locs(info, chpi_amplitudes, t_step_max=1., too_close='raise',
                          # adjust_dig=False, verbose=None):
        # Set up magnetic dipole fits
    
        # proj = sin_fits['proj']
        # meg_picks = pick_channels(
        #     raw.info['ch_names'], proj['data']['col_names'], ordered=True)
    
        # raw.info = pick_info(raw.info, meg_picks)  # makes a copy
        # raw.info['projs'] = [proj]
    
        # del meg_picks, proj
        # meg_coils = _concatenate_coils(_create_meg_coils(raw.info['chs'], 'accurate'))
    
        # # Set up external model for interference suppression
        
        # cov = make_ad_hoc_cov(raw.info, verbose=False)
        # whitener, _ = compute_whitener(cov, raw.info, verbose=False)
        
    
        # Make some location guesses (1 cm grid)
        # R = np.linalg.norm(meg_coils[0], axis=1).min()
        # guesses = _make_guesses(dict(R=R, r0=np.zeros(3)), 0.01, 0., 0.005,
        #                         verbose=False)[0]['rr']
        # logger.info('Computing %d HPI location guesses (1 cm grid in a %0.1f cm '
        #             'sphere)' % (len(guesses), R * 100))
        # fwd = self.fwd #_magnetic_dipole_field_vec(guesses, meg_coils, too_close)
        # fwd = np.dot(fwd, self.whitener.T)
        # fwd.shape = (self.guesses.shape[0], 3, -1)
        # fwd = np.linalg.svd(fwd, full_matrices=False)[2]
        # guesses = dict(rr=self.guesses, whitened_fwd_svd=fwd)
        # del fwd
        #%% def compute_chpi_locs
        
        # skip this window if bad
        if not np.isfinite(sin_fit).all():
            return
    
        # check if data has sufficiently changed
        # this part is very important, as it catches most calculations before they are actually applied
        if last['sin_fit'] is not None:  # first iteration
            corrs = np.array(
                [np.corrcoef(s, l)[0, 1]
                    for s, l in zip(sin_fit, last['sin_fit'])])
            corrs *= corrs
            # check to see if we need to continue
            if fit_time - last['coil_fit_time'] <= t_step_max - 1e-7 and \
                    (corrs > 0.98).sum() >= 3:
                # don't need to refit data
                return
    
        # update 'last' sin_fit *before* inplace sign mult
        last['sin_fit'] = sin_fit.copy()
    
        #
        # 2. Fit magnetic dipole for each coil to obtain coil positions
        #    in device coordinates
        #
        coil_fits = [_fit_magnetic_dipole(f, x0, too_close, self.whitener,
                                          self.meg_coils, self.guesses)
                     for f, x0 in zip(sin_fit, last['coil_dev_rrs'])]
        rrs, gofs, moments = zip(*coil_fits)
        # chpi_locs['times'].append(fit_time)
        # chpi_locs['rrs'].append(rrs)
        # chpi_locs['gofs'].append(gofs)
        # chpi_locs['moments'].append(moments)
        last['coil_fit_time'] = fit_time
        last['coil_dev_rrs'] = rrs
        
        # n_times = len(chpi_locs['times'])
        # shapes = dict(
        #     times=(n_times,),
        #     rrs=(n_times, n_hpi, 3),
        #     gofs=(n_times, n_hpi),
        #     moments=(n_times, n_hpi, 3),
        # )
        # for key, val in chpi_locs.items():
            # chpi_locs[key] = np.array(val, float).reshape(shapes[key])
        #%% def compute_head_pos(info, chpi_locs, dist_limit=0.005, gof_limit=0.98,
                             # adjust_dig=False, verbose=None):
    
        
        # for fit_time, this_coil_dev_rrs, g_coils in zip(
        #         *(chpi_locs[key] for key in ('times', 'rrs', 'gofs'))):
        this_coil_dev_rrs = np.array(rrs)
        g_coils = np.array(gofs)

        use_idx = np.where(g_coils >= gof_limit)[0]
    
        #
        # 1. Check number of good ones
        #
        if len(use_idx) < 3:
            msg = (_time_prefix(fit_time) + '%s/%s good HPI fits, cannot '
                   'determine the transformation (%s GOF)!'
                   % (len(use_idx), self.n_coils,
                      ', '.join('%0.2f' % g for g in g_coils)))
            warn(msg)
            return
    
        #
        # 2. Fit the head translation and rotation params (minimize error
        #    between coil positions and the head coil digitization
        #    positions) iteratively using different sets of coils.
        #
        this_quat, g, use_idx = _fit_chpi_quat_subset(
            this_coil_dev_rrs, self.hpi_dig_head_rrs, use_idx)
    
        #
        # 3. Stop if < 3 good
        #
    
        # Convert quaterion to transform
        this_dev_head_t = _quat_to_affine(this_quat)
        est_coil_head_rrs = apply_trans(this_dev_head_t, this_coil_dev_rrs)
        errs = np.linalg.norm(self.hpi_dig_head_rrs - est_coil_head_rrs, axis=1)
        n_good = ((g_coils >= gof_limit) & (errs < dist_limit)).sum()
        if n_good < 3:
            warn(_time_prefix(fit_time) + '%s/%s good HPI fits, cannot '
                 'determine the transformation (%s mm/GOF)!'
                 % (n_good, self.n_coils,
                    ', '.join(f'{1000 * e:0.1f}::{g:0.2f}'
                              for e, g in zip(errs, g_coils))))
            return
    
        # velocities, in device coords, of HPI coils
        dt = fit_time - last['quat_fit_time']
        vs = tuple(1000. * np.linalg.norm(last['coil_dev_rrs'] -
                                          this_coil_dev_rrs, axis=1) / dt)
        logger.info(_time_prefix(fit_time) +
                    ('%s/%s good HPI fits, movements [mm/s] = ' +
                    ' / '.join(['% 8.1f'] * self.n_coils))
                    % ((n_good, self.n_coils) + vs))
            
        # Log results
        # MaxFilter averages over a 200 ms window for display, but we don't
        for ii in range(self.n_coils):
            if ii in use_idx:
                start, end = ' ', '/'
            else:
                start, end = '(', ')'
            log_str = ('    ' + start +
                       '{0:6.1f} {1:6.1f} {2:6.1f} / ' +
                       '{3:6.1f} {4:6.1f} {5:6.1f} / ' +
                       'g = {6:0.3f} err = {7:4.1f} ' +
                       end)
            vals = np.concatenate((1000 * self.hpi_dig_head_rrs[ii],
                                   1000 * est_coil_head_rrs[ii],
                                   [g_coils[ii], 1000 * errs[ii]]))
            if len(use_idx) >= 3:
                if ii <= 2:
                    log_str += '{8:6.3f} {9:6.3f} {10:6.3f}'
                    vals = np.concatenate(
                        (vals, this_dev_head_t[ii, :3]))
                elif ii == 3:
                    log_str += '{8:6.1f} {9:6.1f} {10:6.1f}'
                    vals = np.concatenate(
                        (vals, this_dev_head_t[:3, 3] * 1000.))
            logger.debug(log_str.format(*vals))
    
        # resulting errors in head coil positions
        d = np.linalg.norm(self.last['quat'][3:] - this_quat[3:])  # m
        r = _angle_between_quats(self.last['quat'][:3], this_quat[:3]) / dt
        v = d / dt  # m/sec
        d = 100 * np.linalg.norm(this_quat[3:] - self.pos_0)  # dis from 1st
        logger.debug('    #t = %0.3f, #e = %0.2f cm, #g = %0.3f, '
                     '#v = %0.2f cm/s, #r = %0.2f rad/s, #d = %0.2f cm'
                     % (fit_time, 100 * errs.mean(), g, 100 * v, r, d))
        logger.debug('    #t = %0.3f, #q = %s '
                     % (fit_time, ' '.join(map('{:8.5f}'.format, this_quat))))
        quat = np.concatenate(([fit_time], this_quat, [g],
                                     [errs[use_idx].mean()], [v]))

        last['quat_fit_time'] = fit_time
        last['quat'] = this_quat
        last['coil_dev_rrs'] = this_coil_dev_rrs
        return quat


    @profile
    def headpos(self, t_step_max=1):
        """One large function that works on individual chunks of data instead
        of the entire file at once"""
        #%% def compute_chpi_amplitudes(raw, t_step_min=0.01, t_window='auto',
                                    # ext_order=1, tmin=0, tmax=None, verbose=None):
                                        
        ## IN THIS CELL RAW.INFO IS NOT CHANGED             
        raw = self.raw
        t_step_min=0.01
        t_window='auto'
        ext_order=1
        too_close='raise'
        adjust_dig=False
        tmin=0
        tmax=None 
        
        info_orig = raw.info.copy()                                

        hpi = self.hpi #_setup_hpi_amplitude_fitting(raw.info, t_window, ext_order=ext_order)
        tmin, tmax = raw._tmin_tmax_to_start_stop(tmin, tmax)
        tmin = tmin / raw.info['sfreq']
        tmax = tmax / raw.info['sfreq']
        need_win = hpi['t_window'] / 2.
        fit_idxs = raw.time_as_index(np.arange(
            tmin + need_win, tmax, t_step_min), use_rounding=True)
        logger.info('Fitting %d HPI coil locations at up to %s time points '
                    '(%0.1f sec duration)'
                    % (len(hpi['freqs']), len(fit_idxs), tmax - tmin))

        sin_fits = dict()
        sin_fits['times'] = np.round(fit_idxs + raw.first_samp -
                                     hpi['n_window'] / 2.) / raw.info['sfreq']
        sin_fits['proj'] = self.hpi['proj']
        sin_fits['slopes'] = np.empty(
            (len(sin_fits['times']),
             len(hpi['freqs']),
             len(sin_fits['proj']['data']['col_names'])))
        raw.info = raw.info
        

        chpi_locs = dict(times=[], rrs=[], gofs=[], moments=[])
        hpi_dig_dev_rrs = apply_trans(
            invert_transform(raw.info['dev_head_t'])['trans'],
            _get_hpi_initial_fit(raw.info, adjust=adjust_dig))
        # last = dict(sin_fit=None, coil_fit_time=sin_fits['times'][0] - 1,
        #             coil_dev_rrs=hpi_dig_dev_rrs, quat_fit_time=-0.1, coil_dev_rrs=coil_dev_rrs,
        #                         quat=np.concatenate([rot_to_quat(dev_head_t[:3, :3]),
        #                                               dev_head_t[:3, 3]]))
        n_hpi = len(hpi_dig_dev_rrs)
        # del hpi_dig_dev_rrs
        
        info=raw.info                         
        hpi_dig_head_rrs = _get_hpi_initial_fit(info, adjust=adjust_dig,
                                                verbose='error')
        coil_dev_rrs = apply_trans(invert_transform(info['dev_head_t']),
                                   hpi_dig_head_rrs)
        dev_head_t = info['dev_head_t']['trans']
        pos_0 = dev_head_t[:3, 3]
        # last = dict(quat_fit_time=-0.1, coil_dev_rrs=coil_dev_rrs,
        #             quat=np.concatenate([rot_to_quat(dev_head_t[:3, :3]),
        #                                  dev_head_t[:3, 3]]))
        self.last = dict(sin_fit=None, coil_fit_time=sin_fits['times'][0] - 1,
                    coil_dev_rrs=hpi_dig_dev_rrs, quat_fit_time=-0.1, 
                    # coil_dev_rrs=coil_dev_rrs,
                    quat=np.concatenate([rot_to_quat(dev_head_t[:3, :3]),
                                                      dev_head_t[:3, 3]]))
        quats = []
        
        self.pos_0 = pos_0
        self.n_coils = len(hpi_dig_head_rrs)
        self.hpi_dig_head_rrs = hpi_dig_head_rrs
        
        for mi, fit_idx in enumerate(fit_idxs):
            fit_time = sin_fits['times'][mi]
            raw_cropped=raw
            # raw_cropped = raw.copy().crop(tmax=min((fit_idx+fit_idxs[0])/1000,max(raw.times) ))
            quat = self.feed_cunk(raw_cropped, fit_time, fit_idx=fit_idx,
                                  t_step_max=t_step_max)
            if quat is not None:
                quats.append(quat)
            

        quats = np.array(quats, np.float64)
        quats = np.zeros((0, 10)) if quats.size == 0 else quats
        
        return quats
