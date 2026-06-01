import logging
import numpy as np
import mkl_fft
from pycbc.types import zeros, complex64
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
    v_s    = wins[:, 0]                                    # window starts
    v_e    = wins[:, 1]                                    # window stops

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
                 high_frequency_cutoff=None, fir_fft_length=4096, batch_size=64,tap_sample_rate=2048, engine_sample_rate=2048):
        self.delta_f = delta_f
        self.snr_threshold = snr_threshold
        self.f_high = high_frequency_cutoff
        
        self.threshold_sq = float(snr_threshold**2)
        
        self.fir_fft_len = fir_fft_length
        self.batch_size = batch_size
        self.tap_sr = int(tap_sample_rate)
        self.engine_sr = int(engine_sample_rate)

        total_batch_size = batch_size * fir_fft_length
        self.temp_freq_mult = zeros(total_batch_size, dtype=complex64).data.reshape(self.batch_size, fir_fft_length)
        self.corr_output_buffer = zeros(total_batch_size, dtype=complex64).data.reshape(self.batch_size, fir_fft_length)

        # 3. Filter Preparation Buffers
        self.filters_padded = zeros(total_batch_size, dtype=complex64)
        self.filters_f_buffer = zeros(total_batch_size, dtype=complex64)
        
        # Direct MKL handle
        self.fft_lib = mkl_fft

    def prepare_filters(self, fir_taps, tap_counts):
        """
        Prepare frequency-domain filters for a batch of taps.
        """
        n_filters, n_taps = fir_taps.shape
        if n_taps >= self.fir_fft_len:
             raise ValueError("FIR Taps (%d) exceed FFT block length (%d)" % 
                              (n_taps, self.fir_fft_len))
        
        # Calculate max tap count for validity logic
        n_taps_max = int(np.max(tap_counts))
        
        filters_f = self._fft_all_filters(fir_taps, tap_counts)
        return filters_f, n_taps_max

# Temporarily split up for the ref snr handling

    def process_segment_coarse(self, stilde, psd, ref_template, filters_f, n_taps, indices, 
                        valid_slice=None, windows=None):
        if valid_slice is None:
            valid_slice = getattr(stilde, 'analyze', None)

        if windows is not None and not windows:
            return [], [], [], 0.0

        self._diag_corr_saved = False

        import pycbc.filter
        h_norm = pycbc.filter.sigmasq(
            ref_template,
            psd=psd,
            low_frequency_cutoff=ref_template.f_lower,
            high_frequency_cutoff=self.f_high
        )

        # Calculate Reference SNR
        snr, _, norm = matched_filter_core(
            ref_template, stilde, psd=psd,
            low_frequency_cutoff=ref_template.f_lower,
            high_frequency_cutoff=self.f_high,
            h_norm=h_norm
        )
        # 2. Determine downsampling ratio (2048 / 512 = 4)
        decimate = int(np.round(self.tap_sr / self.engine_sr))
        
        # 1. Establish the fully normalized physical SNR time series (2048 Hz spacing)
        full_res_snr = snr.numpy() * (norm * stilde.delta_t)
        
        # 2. Hard-slice the physical SNR stream to 512 Hz spacing (reverting max-pooling)
        self.ref_snr = full_res_snr[::decimate]


        if valid_slice is not None:
            coarse_start = int(valid_slice.start // decimate)
            coarse_stop = int(valid_slice.stop // decimate)
            coarse_valid_slice = slice(coarse_start, coarse_stop)
        else:
            coarse_valid_slice = None

        local_idxs, t_idxs, snr_vals = self._execute_blocked_kernel(
            self.ref_snr, filters_f, n_taps,
            valid_slice=coarse_valid_slice if windows is None else None ,
            windows=windows
        )
        
        if len(local_idxs) > 0:
            global_ids = indices[local_idxs]
            return global_ids, t_idxs, snr_vals, h_norm
        else:
            return [], [], [], h_norm

    def process_segment_fine(self, stilde, psd, ref_template, filters_f, n_taps, indices, 
                        valid_slice=None, windows=None):
        if valid_slice is None:
            valid_slice = getattr(stilde, 'analyze', None)

        if windows is not None and not windows:
            return [], [], [], 0.0


        h_norm = ref_template.sigmasq(psd)

        # 2. Calculate Reference SNR
        snr, _, norm = matched_filter_core(
            ref_template, stilde, psd=psd,
            low_frequency_cutoff=ref_template.f_lower,
            high_frequency_cutoff=self.f_high,
            h_norm=h_norm
        )

        self.ref_snr = snr.numpy() * (norm * stilde.delta_t)
        local_idxs, t_idxs, snr_vals = self._execute_blocked_kernel(
            self.ref_snr, filters_f, n_taps,
            valid_slice=valid_slice if windows is None else None,
            windows=windows
        )
        
        if len(local_idxs) > 0:
            global_ids = indices[local_idxs]
            return global_ids, t_idxs, snr_vals, h_norm
        else:
            return [], [], [], h_norm


    def _fft_all_filters(self, taps, counts):
        """Helper to FFT all filters using mkl_fft."""
        n_filters, n_taps_alloc = taps.shape
        filters_f = np.zeros((n_filters, self.fir_fft_len), dtype=np.complex64)
        
        # 1. Read metadata from the bank to determine the source generation rate
        bank_sample_rate = self.tap_sr
        engine_sample_rate = self.engine_sr

        print(f"engine_sample_rate:{engine_sample_rate},bank_sample_rate:{bank_sample_rate}")
        # Alternatively, determine the downsampling factor directly:
        exact_ratio = (bank_sample_rate / engine_sample_rate)
        decimation_factor = int(np.round(exact_ratio))

        if abs(exact_ratio - decimation_factor) > 1e-5 or decimation_factor < 1:
            raise ValueError(
                f"Multi-rate Error: The bank sample rate ({self.tap_sr} Hz) must be "
                f"an exact integer multiple of the engine sample "
                f"rate ({self.engine_sr} Hz).\n"
                f"Calculated ratio was {exact_ratio:.4f}. Please use standard power-of-2 "
                f"downsampling scales (e.g., 2048/512)."
            )

        # 2. Establish the high-resolution FFT padding length to preserve delta_f
        # 512/4096 = 0.125   2048/(4*4096) = 0.125 preserving delta_f
        # 4096/4096 = 1      2048/(1/2*4096) = 1 for 4096 engine 2048 bank
        high_res_fft_len = self.fir_fft_len * decimation_factor
       # print(f"high_res_fft_len:{high_res_fft_len}")
        # Temp allocations for high-resolution processing
        high_res_padded = np.zeros((self.batch_size, high_res_fft_len), dtype=np.complex64)
        for start in range(0, n_filters, self.batch_size):
            end = min(start + self.batch_size, n_filters)
            batch_len = end - start
            
            # Zero out processing buffer for next call
            high_res_padded[:batch_len, :] = 0.0
            
            # Copy raw 2048 Hz taps into the start of the buffer
            tmp_taps = taps[start:end]
            high_res_padded[:batch_len, :n_taps_alloc] = tmp_taps
            
            # 3. Handle Variable Time-Domain Roll Logic at the native 2048 Hz rate
            current_counts = counts[start:end]
            roll_offsets = -(current_counts // 2)
            
            cols_high = np.arange(high_res_fft_len)
            rows = np.arange(batch_len)[:, None]
            shifted_cols_high = (cols_high[None, :] - roll_offsets[:, None]) % high_res_fft_len
            
            current_data = high_res_padded[:batch_len].copy()
            high_res_padded[:batch_len] = current_data[rows, shifted_cols_high]
#            print(f"len(high_res_padded):{len(high_res_padded)}")
            # 4. Transform to Frequency Domain at native resolution
            fft_high_res = self.fft_lib.fft(high_res_padded[:batch_len], axis=-1)
            # 5. Brick-Wall Frequency Slicing (Anti-Aliasing & Decimation Match)
            # Because the data engine goes up to 256Hz (the 512Hz Nyquist limit), only need the first 4096 bins of that spectrum
            fft_sliced = fft_high_res[:batch_len, :self.fir_fft_len]
            
            # 6. Conjugate & Store back into the 512 Hz buffer block
            filters_f[start:end] = np.conj(fft_sliced)
            
        return filters_f

    def _execute_blocked_kernel(self, data, filters_f, n_taps, valid_slice,
                                windows=None):
        """
        Inner loop: Time-Blocking + Filter-Batching using mkl_fft.

        Parameters
        ----------
        data        : 1-D complex64 SNR time series
        filters_f   : (n_filters, N_FFT) complex64 frequency-domain filters
        n_taps      : 1-D int array of tap counts per filter
        valid_slice : slice object (used when windows is None)
        windows     : list of (start, stop) int pairs (Pass-2 path).
                      When provided, valid_slice is ignored.
                      Block sets are precomputed per geometry and cached in
                      self._block_cache so they are reused across filter
                      batches that share the same N_VALID.

        The block_f_cache (time-block FFTs) is local to each call but shared
        across all filter batches within the call — same as original.
        For the windowed path, self._block_cache stores precomputed block
        geometry so _compute_needed_blocks is called at most once per unique
        N_VALID value per process_segment_windowed() invocation.
        """
        tap_groups = 3
        nsizes = np.quantile(n_taps, np.linspace(0, 1, tap_groups+1)[1:]).astype(int)
        n_samples = len(data)
        n_filters = len(filters_f)

        N_FFT = self.fir_fft_len

        all_f_idxs = []
        all_t_idxs = []
        all_snrs = []
        all_tstarts = []

        freq_mult_view = self.temp_freq_mult #.data.reshape(self.batch_size, N_FFT)
        corr_out_view  = self.corr_output_buffer #.data.reshape(self.batch_size, N_FFT)

        # Shared time-block FFT cache (keyed by t_start; reused across f-batches)
        block_f_cache = {}

        # Geometry cache: N_VALID → (block_starts, roi_starts, roi_stops)
        # Only populated on the windowed path; reset each call.
        geom_cache = {}

        windowed = windows is not None

        if not windowed:
            # Single-slice path
            if valid_slice:
                v_start = valid_slice.start
                v_stop  = valid_slice.stop
            else:
                v_start = 0
                v_stop  = n_samples

        # Diagnostic: full-segment corr_out timeseries for filter 0
        # Used to compare coarse vs fine normalization. Remove after validation.
        self._diag_corr_ts = np.zeros(n_samples, dtype=np.float32)

        for f_start in range(0, n_filters, self.batch_size):
            f_end             = min(f_start + self.batch_size, n_filters)
            actual_batch_size = f_end - f_start
 
            current_mult_view = freq_mult_view[:actual_batch_size]
            current_corr_view = corr_out_view[:actual_batch_size]
            filter_batch_f = filters_f[f_start:f_end]

            # temporary diagnostic - remove after
            block_abs = np.abs(current_corr_view)
#            logging.info(f"corr_out stats: max={block_abs.max():.4f}, "
#                         f"mean={block_abs.mean():.4f}, "
#                         f"threshold_sq={self.threshold_sq:.2f}")
            
            # Valid output samples per block (Overlap-Save)
            n_taps_max = n_taps[f_start:f_end].max()
            i = np.searchsorted(nsizes, n_taps_max)
            n_taps_max = int(nsizes[i])
            N_VALID = N_FFT - n_taps_max + 1
            STEP = N_VALID
            bad_start = n_taps_max // 2

            if windowed:
                # ---- Windowed path: skip dead time between windows ----
                if N_VALID not in geom_cache:
                    geom_cache[N_VALID] = _compute_needed_blocks(
                        windows, bad_start, N_VALID, n_samples)
                block_starts, roi_starts, roi_stops = geom_cache[N_VALID]

                for t_start, roi_start, roi_stop in zip(
                        block_starts.tolist(), roi_starts.tolist(), roi_stops.tolist()):
                    t_start  = int(t_start)
                    roi_start = int(roi_start)
                    roi_stop  = int(roi_stop)
                    roi_len   = roi_stop - roi_start
                    if roi_len <= 0:
                        continue

                    buf_slice_start = roi_start - t_start

                    if t_start not in block_f_cache:
                        t_end        = min(t_start + N_FFT, n_samples)
                        block_in     = np.zeros(N_FFT, dtype=complex64)
                        block_in[0:t_end - t_start] = data[t_start:t_end]
                        block_f_cache[t_start] = self.fft_lib.fft(block_in)

                    fast_multiply_analytic_cython(
                        block_f_cache[t_start], filter_batch_f, current_mult_view)
                    self.fft_lib.ifft(current_mult_view, axis=-1, out=current_corr_view)

                    # Diagnostic: capture filter-0 output for the full segment
                    # Used to compare coarse vs fine normalization. Remove after validation.
                    if f_start == 0:
                        self._diag_corr_ts[roi_start:roi_stop] = np.abs(
                            current_corr_view[0, buf_slice_start:buf_slice_start + roi_len]
                        )

                    f_list, t_list, s_list = find_peaks_in_block_cython(
                        current_corr_view, roi_start, roi_len,
                        self.threshold_sq, f_start,
                        input_offset=buf_slice_start)
                    if f_list:
                        all_f_idxs.extend(f_list)
                        all_t_idxs.extend(t_list)
                        all_snrs.extend(s_list)

            else:
                # ---- single-slice path ----
                first_block_idx = (v_start - bad_start) // STEP
                loop_start = first_block_idx * STEP

                for t_start in range(loop_start, n_samples, STEP):
                    block_valid_t0 = t_start + bad_start

                    if block_valid_t0 >= v_stop:
                        break
                    if block_valid_t0 + N_VALID <= v_start:
                        continue

                    roi_start = max(v_start, block_valid_t0)
                    roi_stop = min(v_stop, block_valid_t0 + N_VALID)
                    roi_len = roi_stop - roi_start
                    if roi_len <= 0:
                        continue

                    buf_slice_start = roi_start - t_start
                    t_end = min(t_start + N_FFT, n_samples)

                    if t_start not in block_f_cache:
                        block_in = np.zeros(N_FFT, dtype=complex64)
                        block_in[0:t_end - t_start] = data[t_start:t_end]
                        block_f_cache[t_start] = self.fft_lib.fft(block_in)

                    fast_multiply_analytic_cython(
                        block_f_cache[t_start], filter_batch_f, current_mult_view)
                    self.fft_lib.ifft(current_mult_view, axis=-1, out=current_corr_view)

                    # Diagnostic: capture filter-0 output for the full segment
                    # Used to compare coarse vs fine normalization. Remove after validation.
                    if f_start == 0:
                        self._diag_corr_ts[roi_start:roi_stop] = np.abs(
                            current_corr_view[0, buf_slice_start:buf_slice_start + roi_len]
                        )

                    f_list, t_list, s_list = find_peaks_in_block_cython(
                        current_corr_view, roi_start, roi_len,
                        self.threshold_sq, f_start,
                        input_offset=buf_slice_start)
                    if f_list:
                        all_f_idxs.extend(f_list)
                        all_t_idxs.extend(t_list)
                        all_snrs.extend(s_list)

        return (np.array(all_f_idxs, dtype=np.int32),
                np.array(all_t_idxs, dtype=np.int64),
                np.array(all_snrs,   dtype=np.complex64))
