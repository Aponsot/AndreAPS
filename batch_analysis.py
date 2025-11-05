def _merge_close_peaks_with_aic(xw, yw, model, result, noise,
                                min_sep_q, merge_min_sep_frac,
                                overlap_coef_min, aic_improve, height_frac):
    """
    Decide which peaks to zero-out using three gates:
    (1) Proximity (q or sigma scale)
    (2) Strong shape overlap
    (3) Engulf rule + Local contribution test
    Confirm with AIC if needed.
    Returns: sorted list of peak indices to kill.
    """
    peaks = extract_peaks(result)
    n = len(peaks)
    to_kill = set()

    # Pair candidates sorted by center distance
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            di = abs(peaks[j]["center"] - peaks[i]["center"])
            pairs.append((di, i, j))
    pairs.sort(key=lambda t: t[0])

    for _, i, j in pairs:
        if i in to_kill or j in to_kill:
            continue
        pi, pj = peaks[i], peaks[j]
        ci, cj = pi["center"], pj["center"]
        si, sj = abs(pi["sigma"]), abs(pj["sigma"])
        hi, hj = pi["height"], pj["height"]
        if not all(np.isfinite([ci, cj, si, sj, hi, hj])) or si <= 0 or sj <= 0:
            continue

        d = abs(cj - ci)
        close_by_resolution = (d < min_sep_q)
        close_by_sigma = (d < merge_min_sep_frac * 0.5 * (si + sj))
        ovl = _gaussian_shape_overlap(ci, si, cj, sj)
        candidate = close_by_resolution or close_by_sigma or (ovl >= overlap_coef_min)
        if not candidate:
            continue

        # Choose weaker by height (tie: smaller amplitude)
        drop = i if (hi < hj * height_frac or (abs(hi - hj) < 1e-12 and pi["amplitude"] < pj["amplitude"])) else j
        keep = j if drop == i else i
        pd, pk = (pi, pj) if drop == i else (pj, pi)

        # --- Engulf HARD rule: tiny peak fully within big peak ---
        max_s = max(si, sj)
        engulf = (ovl >= ENGULF_OVL_MIN) and (d <= ENGULF_DIST_SIGMA * max_s) and (pd["height"] <= ENGULF_HEIGHT_FRAC * pk["height"])
        if engulf:
            to_kill.add(drop)
            continue

        # --- Local contribution test (noise-aware) ---
        dloc, nloc = _local_delta_rss(xw, yw, model, result, drop, noise, win_sigma=LOCAL_WIN_SIGMA)
        if nloc > 0:
            # normalize by noise power over that window
            snr_local = dloc / (max(noise**2, 1e-16) * nloc)
            if snr_local < LOCAL_SNR_MIN:
                to_kill.add(drop)
                continue

        # --- AIC check as last resort ---
        params_refit = result.params.copy()
        params_refit[f"g{drop}_amplitude"].set(value=0.0, vary=False)
        params_refit[f"g{drop}_center"].set(vary=False)
        params_refit[f"g{drop}_sigma"].set(vary=False)
        test = model.fit(yw, params_refit, x=xw, calc_covar=False, method="least_squares", max_nfev=600)
        delta_aic = result.aic - test.aic   # positive => full model better
        if delta_aic < aic_improve:
            to_kill.add(drop)

    return sorted(to_kill)
