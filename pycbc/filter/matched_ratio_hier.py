#!/home/kkacanja/.conda/envs/firhier/bin/python3.12
import logging
import numpy as np
import time
import mkl_fft

import pycbc
from pycbc.types import complex64, float32, zeros, TimeSeries, FrequencySeries
from pycbc.filter.matchedfilter import matched_filter_core
from pycbc.filter.matchedfilter_cpu import fast_multiply_analytic_cython, find_peaks_in_block_cython


def _compute_needed_blocks(windows, bad_start, N_VALID, n_samples):
    """
    Given a list of (start, stop) active windows, return:
      block_starts : sorted 1-D int64 array of block t_start values that
                     overlap at least one window
      roi_starts   : parallel int64 array — union ROI start within each block
      roi_stops    : parallel int64 array — union ROI stop  within each block

    All arithmetic is vectorised over windows; no Python loop per block.
    This is called once per unique geometry (N_VALID value) and cached by
    the caller.
    """
    STEP   = N_VALID
    wins   = np.asarray(windows, dtype=np.int64)          # (W, 2)
    v_s    = wins[:, 0]                                   # window starts
    v_e    = wins[:, 1]                                   # window stops

    # Block-index range that each window can touch
    first_blk = ((v_s - bad_start)     // STEP) * STEP
    last_blk  = ((v_e - 1 - bad_start) // STEP) * STEP

    # Collect all required block starts
    candidates = set()
    for fb, lb in zip(first_blk.tolist(), last_blk.tolist()):
        if lb < fb:
            continue
        for t in range(int(fb), int(lb) + STEP, STEP):
            if 0 <= t < n_samples:
                candidates.add(t)

    if not candidates:
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty

    block_starts = np.array(sorted(candidates), dtype=np.int64)   # (B,)

    # Vectorised ROI computation: block valid region vs each window → union
    bv_s = block_starts + bad_start                                # (B,)
    bv_e = bv_s + N_VALID                                          # (B,)

    # Broadcast (B,1) vs (1,W) → overlap arrays (B,W)
    rs = np.maximum(bv_s[:, None], v_s[None, :])
    re = np.minimum(bv_e[:, None], v_e[None, :])
    overlap = re > rs                                              # (B,W) bool

    INF = np.iinfo(np.int64).max
    roi_starts = np.where(overlap, rs, INF).min(axis=1)           # (B,)
    roi_stops  = np.where(overlap, re, 0  ).max(axis=1)           # (B,)

    # Drop blocks with no real overlap (boundary edge cases)
    valid = roi_stops > roi_starts
    return block_starts[valid], roi_starts[valid], roi_stops[valid]



class RatioMatchedFilterControl(object):
    """
    High-performance engine for hierarchical "Ratio/FIR" matched filtering.
    Uses mkl_fft for ALL FFT operations to maximize throughput and consistency.
    """

    def __init__(self, snr_threshold, delta_f,
                 high_frequency_cutoff=None, fir_fft_length=4096,
                 batch_size=64, tap_sample_rate=2048, engine_sample_rate=2048):
        self.delta_f = delta_f
        self.snr_threshold = snr_threshold
        self._window_compute_time = 0
        self._block_fft_time = 0.0
        self._block_fft_count = 0
        self.f_high = high_frequency_cutoff

        self.threshold_sq = float(snr_threshold**2)

        self.fir_fft_len = fir_fft_length
        self.batch_size = batch_size
        self.tap_sr = int(tap_sample_rate)
        self.engine_sr = int(engine_sample_rate)

        total_batch_size = batch_size * fir_fft_length
        self.temp_freq_mult = zeros(total_batch_size, dtype=complex64).data.reshape(self.batch_size, fir_fft_length)
        self.corr_output_buffer = zeros(total_batch_size, dtype=complex64).data.reshape(self.batch_size, fir_fft_length)

        # Filter preparation buffers
        self.filters_padded   = zeros(total_batch_size, dtype=complex64)
        self.filters_f_buffer = zeros(total_batch_size, dtype=complex64)

        # Direct MKL handle
        self.fft_lib = mkl_fft


    def prepare_filters(self, fir_taps, tap_counts, coarse_engine):
        """
        Compute both fine and coarse filters from a single high-res FFT pass.
        Optimized for zero-allocation memory bandwidth.
        """
        n_filters, n_taps_alloc = fir_taps.shape
        if n_taps_alloc >= self.fir_fft_len:
            raise ValueError(f"FIR Taps ({n_taps_alloc}) exceed FFT block length ({self.fir_fft_len})")

        # decimation factors
        fine_dec   = int(np.round(self.tap_sr   / self.engine_sr))    # 1
        coarse_dec = int(np.round(coarse_engine.tap_sr / coarse_engine.engine_sr))  # 4

        # High-res base
        high_res_fft_len = self.fir_fft_len * coarse_dec   # 4096 * 4 = 16384

        filters_f_fine   = np.zeros((n_filters, self.fir_fft_len),          dtype=np.complex64)
        filters_f_coarse = np.zeros((n_filters, coarse_engine.fir_fft_len), dtype=np.complex64)

        # Pre-allocate buffer once for the entire batching process
        high_res_padded = np.zeros((self.batch_size, high_res_fft_len), dtype=np.complex64)
        
        N_c = coarse_engine.fir_fft_len
        half_N = N_c // 2

        for start in range(0, n_filters, self.batch_size):
            end       = min(start + self.batch_size, n_filters)
            batch_len = end - start

            # 1. Zero out processing buffer (Fastest way to reset)
            high_res_padded[:batch_len, :] = 0.0

            tmp_taps = fir_taps[start:end]
            current_counts = tap_counts[start:end]

            # 2. Fast Time-Domain Roll (Zero Matrix Allocation)
            # Directly maps the shifted taps to avoid multi-megabyte fancy indexing
            for b in range(batch_len):
                count = current_counts[b]
                k = count // 2
                
                # Shift remainder of template to the front
                high_res_padded[b, :count - k] = tmp_taps[b, k:count]
                # Wrap front of template to the extreme back
                high_res_padded[b, -k:] = tmp_taps[b, :k]

            # 3. Execute FFT
            fft_high_res = self.fft_lib.fft(high_res_padded[:batch_len], axis=-1)

            # 4. In-Place Conjugate
            # Modifies the array in memory, saving 3 intermediate array allocations
            np.conjugate(fft_high_res, out=fft_high_res)

            # 5. Direct Slicing (using the pre-conjugated array)
            
            # Coarse: Reconstruct conjugate symmetry
            filters_f_coarse[start:end, :half_N + 1] = fft_high_res[:, :half_N + 1]
            filters_f_coarse[start:end, half_N + 1:] = fft_high_res[:, -(half_N - 1):]

            # Fine: Decimate by coarse_dec
            filters_f_fine[start:end] = fft_high_res[:, ::coarse_dec][:, :self.fir_fft_len]

        n_taps_max = int(np.max(tap_counts))
        return filters_f_fine, filters_f_coarse, n_taps_max

    def _process_segment(self, stilde, psd, ref_template, filters_f,
                        n_taps, indices, label,
                        valid_slice=None, windows=None, is_coarse=False):
        """
        Unified coarse/fine segment processing. 
        Set is_coarse=True for the coarse pass (truncates to f_high, 
        applies decimation to valid_slice and ref_snr).
        """
        if valid_slice is None:
            valid_slice = getattr(stilde, 'analyze', None)

        if windows is not None and not windows:
            return [], [], [], 0.0

        # Coarse pass truncates arrays to f_high to save compute in matched_filter_core
        if is_coarse:
            f_high_idx = int(self.f_high / psd.delta_f) + 1
            stilde_mf  = stilde[:f_high_idx]
            psd_mf     = psd[:f_high_idx]
            tmpl_mf    = ref_template[:f_high_idx]
        else:
            stilde_mf = stilde
            psd_mf    = psd
            tmpl_mf   = ref_template

        h_norm = pycbc.filter.sigmasq(
            ref_template, psd=psd,
            low_frequency_cutoff=ref_template.f_lower,
            high_frequency_cutoff=self.f_high
        )

        t1 = time.time()
        snr, _, norm = matched_filter_core(
            tmpl_mf, stilde_mf, psd_mf,
            low_frequency_cutoff=ref_template.f_lower,
            high_frequency_cutoff=self.f_high,
            h_norm=h_norm
        )
        t2 = time.time()
        logging.info(f"{label}:matched_filter_core time {t2-t1:.6f}")

        decimate = int(np.round(self.tap_sr / self.engine_sr))

        if is_coarse:
            self.ref_snr = snr.numpy() * (norm * stilde_mf.delta_t) / decimate
            t1 = time.time()
            if valid_slice is not None:
                kernel_slice = slice(
                    int(valid_slice.start // decimate),
                    int(valid_slice.stop  // decimate)
                )
                print(f"[DEBUG] coarse ref_snr len={len(self.ref_snr)}, "
                  f"valid_slice={kernel_slice}, "
                  f"decimate={decimate}")
            else:
                kernel_slice = None
            t2 = time.time()
            logging.info(f"coarse start/end loop:{t2-t1}")
        else:
            self.ref_snr  = snr.numpy() * (norm * stilde_mf.delta_t)
            kernel_slice  = valid_slice

        t1 = time.time()
        local_idxs, t_idxs, snr_vals = self._execute_blocked_kernel(
            self.ref_snr, filters_f, n_taps,
            valid_slice=kernel_slice if windows is None else None,
            windows=windows
        )
        t2 = time.time()
        logging.info(f"{label}:_execute_blocked_kernel time {t2-t1:.6f}")

        if len(local_idxs) > 0:
            return indices[local_idxs], t_idxs, snr_vals, h_norm
        else:
            return [], [], [], h_norm

    # Currently set to this for crofiling will remove for final version
    def process_segment_coarse(self, stilde, psd, ref_template, filters_f,
                            n_taps, indices, valid_slice=None, windows=None):
        return self._process_segment(
            stilde, psd, ref_template, filters_f, n_taps, indices,
            label='COARSE', valid_slice=valid_slice, windows=windows,
            is_coarse=True
        )


    def process_segment_fine(self, stilde, psd, ref_template, filters_f,
                            n_taps, indices, valid_slice=None, windows=None):
        return self._process_segment(
            stilde, psd, ref_template, filters_f, n_taps, indices,
            label='FINE', valid_slice=valid_slice, windows=windows,
            is_coarse=False
        )

    def _execute_blocked_kernel(self, data, filters_f, n_taps, valid_slice=None, windows=None):
        """
        Inner loop: Time-Blocking + Filter-Batching using mkl_fft.
        """
        tap_groups = 3
        nsizes = np.quantile(n_taps, np.linspace(0, 1, tap_groups+1)[1:]).astype(int)
        n_samples = len(data)
        n_filters = len(filters_f)
        
        N_FFT = self.fir_fft_len
        print(f"[DEBUG] n_taps raw max={n_taps.max()}, quantile nsizes={nsizes}, "
              f"N_VALID per group would be={[self.fir_fft_len - s + 1 for s in nsizes]}")    

        n_batches = (n_filters + self.batch_size - 1) // self.batch_size
        print(f"[DEBUG] n_filters={n_filters}, batch_size={self.batch_size}, "
              f"n_batches={n_batches}")

        all_f_idxs = []
        all_t_idxs = []
        all_snrs = []
        all_tstarts = []
 
        freq_mult_view = self.temp_freq_mult
        corr_out_view = self.corr_output_buffer
 
        if valid_slice:
            v_start = valid_slice.start
            v_stop = valid_slice.stop
        else:
            v_start = 0
            v_stop = n_samples
 
        block_f_cache = {}
        geometry_cache = {}
 
        # --- OUTER LOOP: Filter Batches ---
        for f_start in range(0, n_filters, self.batch_size):  
            f_end = min(f_start + self.batch_size, n_filters)
            actual_batch_size = f_end - f_start
 
            current_mult_view = freq_mult_view[:actual_batch_size]
            current_corr_view = corr_out_view[:actual_batch_size]
            
            # Valid output samples per block (Overlap-Save)
            n_taps_max = n_taps[f_start:f_end].max()
            i = np.searchsorted(nsizes, n_taps_max)
            n_taps_max = nsizes[i]
            
            N_VALID = N_FFT - n_taps_max + 1
            STEP = N_VALID
            bad_start = n_taps_max // 2
 
            # Route 1: Specific Interest Windows
            if windows is not None and len(windows) > 0:
#                d1 = time.time()
                if N_VALID not in geometry_cache:
                    geometry_cache[N_VALID] = _compute_needed_blocks(
                        windows, bad_start, N_VALID, n_samples
                    )
                block_starts, roi_starts, roi_stops = geometry_cache[N_VALID]
#                self._window_compute_time += time.time() - d1
 
                for t_start, roi_start, roi_stop in zip(block_starts, roi_starts, roi_stops):
                    roi_len = roi_stop - roi_start
                    if roi_len <= 0:
                        continue
 
                    buf_slice_start = roi_start - t_start
                    t_end = min(t_start + N_FFT, n_samples)
 
                    if t_start not in block_f_cache:
#                        _fft_t1 = time.time()
                        block_in_view = np.zeros(self.fir_fft_len, dtype=complex64)
                        block_in_view[0:t_end-t_start] = data[t_start:t_end]
                        block_f_view = self.fft_lib.fft(block_in_view)
                        block_f_cache[t_start] = block_f_view
#                        self._block_fft_time += time.time() - _fft_t1
#                        self._block_fft_count += 1
 
                    block_f_view = block_f_cache[t_start]
                    fast_multiply_analytic_cython(
                        block_f_view, filters_f[f_start:f_end], current_mult_view
                    )
                    self.fft_lib.ifft(current_mult_view, axis=-1, out=current_corr_view)
                    f_list, t_list, s_list = find_peaks_in_block_cython(
                        current_corr_view, roi_start, roi_len,
                        self.threshold_sq, f_start, input_offset=buf_slice_start
                    )
                    if f_list:
                        all_f_idxs.extend(f_list)
                        all_t_idxs.extend(t_list)
                        all_snrs.extend(s_list)
                        all_tstarts.extend([t_start] * len(s_list))
            # Route 2: Full Valid Slice Sweep — direct loop matching base code
            else:
                first_block_idx = (v_start - bad_start) // STEP
                loop_start = first_block_idx * STEP
 
                for t_start in range(loop_start, n_samples, STEP):
                    block_valid_t0 = t_start + bad_start
 
                    if block_valid_t0 >= v_stop:
                        break
                    if block_valid_t0 + N_VALID <= v_start:
                        continue
 
                    roi_start = max(v_start, block_valid_t0)
                    roi_stop  = min(v_stop, block_valid_t0 + N_VALID)
                    roi_len   = roi_stop - roi_start
                    if roi_len <= 0:
                        continue
 
                    buf_slice_start = roi_start - t_start
                    t_end = min(t_start + N_FFT, n_samples)
 
                    if t_start not in block_f_cache:
#                        _fft_t1 = time.time()
                        block_in_view = np.zeros(self.fir_fft_len, dtype=complex64)
                        block_in_view[0:t_end-t_start] = data[t_start:t_end]
                        block_f_view = self.fft_lib.fft(block_in_view)
                        block_f_cache[t_start] = block_f_view
#                        self._block_fft_time += time.time() - _fft_t1
 #                       self._block_fft_count += 1
 
                    block_f_view = block_f_cache[t_start]
                    fast_multiply_analytic_cython(
                        block_f_view, filters_f[f_start:f_end], current_mult_view
                    )
                    self.fft_lib.ifft(current_mult_view, axis=-1, out=current_corr_view)
                    f_list, t_list, s_list = find_peaks_in_block_cython(
                        current_corr_view, roi_start, roi_len,
                        self.threshold_sq, f_start, input_offset=buf_slice_start
                    )
                    if f_list:
                        all_f_idxs.extend(f_list)
                        all_t_idxs.extend(t_list)
                        all_snrs.extend(s_list)
                        all_tstarts.extend([t_start] * len(s_list)) 
 
 #       print(f"[TIMING] total _compute_needed_blocks time = {self._window_compute_time:.6f} s")
#        print(f"[TIMING] block_fft: {self._block_fft_count} FFTs, {self._block_fft_time:.6f} s total")
 
        return (np.array(all_f_idxs, dtype=np.int32), 
                np.array(all_t_idxs, dtype=np.int64), 
                np.array(all_snrs, dtype=np.complex64))
