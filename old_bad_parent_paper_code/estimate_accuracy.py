import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm


if __name__ == "__main__":
    archive_path = "gazebasevr.zip"
    data_dir = Path("./data")
    if not data_dir.exists():
        with zipfile.ZipFile(archive_path) as zip_ref:
            zip_ref.extractall(data_dir)
    
    accuracy_output_file = Path("spatial_accuracies.csv")
    precision_output_file = Path("spatial_precisions.csv")
    if not accuracy_output_file.exists() or not precision_output_file.exists():
        # Load each S1 RAN file
        # ---------------------
        spatial_accuracies = np.full((465, 3), fill_value=np.nan, dtype=float)
        spatial_precisions = np.full((465, 3), fill_value=np.nan, dtype=float)

        files_list = list(data_dir.rglob("*S1_5_RAN.csv"))
        for f in tqdm(sorted(files_list)):
            df = pd.read_csv(f)

            # Identify subject and round
            # --------------------------
            nb_round = int(f.stem[2])
            nb_subject = int(f.stem[3:6])

            spatial_accuracies[nb_subject - 1, nb_round - 1] = 0.0
            spatial_precisions[nb_subject - 1, nb_round - 1] = 0.0

            # Minimize saccade latency
            # ------------------------
            time = df["n"].to_numpy()
            gaze = df[["x", "y"]].to_numpy()
            stim = df[["xT", "yT"]].to_numpy()

            max_latency_samples = 200  # approx. 800 ms
            s = stim[:-max_latency_samples]
            latency_errors = np.arange(max_latency_samples, dtype=float)
            latency_errors.fill(np.nan)
            for latency in range(max_latency_samples):
                g = gaze[latency:-(max_latency_samples - latency)]
                dist = np.sqrt(np.sum((g - s)**2, axis=1))
                latency_errors[latency] = np.nanmean(dist)
            
            saccade_latency = np.argmin(latency_errors)
            assert saccade_latency < max_latency_samples - 1, "Best latency is max possible"

            if saccade_latency > 0:
                t = time[:-saccade_latency]
                g = gaze[saccade_latency:]
                s = stim[:-saccade_latency]
            else:
                t = time
                g = gaze
                s = stim

            # Visualize saccade latency removal
            # plt.figure()
            # plt.subplot(3, 1, 1)
            # plt.plot(stim[:, 0], "--k")
            # plt.plot(gaze[:, 0], "-r")
            # plt.subplot(3, 1, 2)
            # plt.plot(s[:, 0], "--k")
            # plt.plot(g[:, 0], "-r")
            # plt.subplot(3, 1, 3)
            # plt.plot(np.arange(max_latency_samples), latency_errors, "-k")
            # plt.show()
            
            # Select fixation periods
            # -----------------------
            fixation_starts = np.any(s[1:] - s[:-1] != 0, axis=1)
            fixation_starts = np.insert(fixation_starts, 0, True)
            fixation_starts_idx = np.where(fixation_starts)[0]

            n_fixations = 20
            fixation_starts_idx = fixation_starts_idx[:n_fixations]

            skip_samples = 100  # approx. 400 ms
            use_samples = 125  # approx. 500 ms
            relative_indices_to_use = np.arange(use_samples) + skip_samples

            fixation_periods = fixation_starts_idx[np.newaxis, :] + relative_indices_to_use[:, np.newaxis]  # (use_samples, n_fixations)
            gfix = g[fixation_periods]  # 2D gaze position for each fixation sample (use_samples, n_fixations, 2)
            sfix = s[fixation_periods]  # 2D stimulus position for each fixation sample (use_samples, n_fixations, 2)
            tfix = t[fixation_periods]  # timestamps for each fixation sample (use_samples, n_fixations)

            # The above is a vectorized version of the following
            # for p in range(n_fixations):
            #     gp = g[fixation_periods[:, p], :]
            #     sp = s[fixation_periods[:, p], :]
            #     assert np.all(gp == gfix[:, p, :])
            #     assert np.all(sp == sfix[:, p, :])

            # Measure spatial accuracy
            # ------------------------
            dist = np.sqrt(np.sum((gfix - sfix)**2, axis=-1))
            fixation_dist = np.nanmean(dist, axis=0)

            # Aggregate measures across fixation periods
            # ------------------------------------------
            agg_dist = np.nanmean(fixation_dist)
            assert not np.isnan(agg_dist), "Mean spatial accuracy is NaN"

            # Save aggregate measures per subject and round
            # ---------------------------------------------
            spatial_accuracies[nb_subject - 1, nb_round - 1] = agg_dist

            # Repeat for spatial precision (RMS)
            # ----------------------------------
            s2s_dist = np.sqrt(np.sum(np.diff(gfix, axis=0)**2, axis=-1))
            fixation_rms = np.sqrt(np.nanmean(s2s_dist**2, axis=0))

            agg_rms = np.nanmean(fixation_rms)
            assert not np.isnan(agg_rms), "Mean spatial precision (RMS) is NaN"

            spatial_precisions[nb_subject - 1, nb_round - 1] = agg_rms

            # Manually inspect extreme outliers
            # if agg_s2s_rms > 1.0:
            #     plot_t = tfix.flatten(order='F')
            #     plot_gx = gfix[..., 0].flatten(order='F')
            #     plot_gy = gfix[..., 1].flatten(order='F')
            #     plot_sx = sfix[..., 0].flatten(order='F')
            #     plot_sy = sfix[..., 1].flatten(order='F')

            #     is_nan = np.isnan(plot_gx) | np.isnan(plot_gy)
            #     plot_t = plot_t[~is_nan]
            #     plot_gx = plot_gx[~is_nan]
            #     plot_gy = plot_gy[~is_nan]
            #     plot_sx = plot_sx[~is_nan]
            #     plot_sy = plot_sy[~is_nan]
                
            #     plt.figure()
            #     plt.subplot(2, 1, 1)
            #     plt.scatter(plot_t, plot_sx, c="k")
            #     plt.scatter(plot_t, plot_gx, c="r")
            #     plt.ylabel("Horizontal positions")
            #     plt.title(f"S_{nb_round}{nb_subject:03d} | S2S RMS = {agg_s2s_rms:.2f}")
            #     plt.subplot(2, 1, 2)
            #     plt.scatter(plot_t, plot_sy, c="k")
            #     plt.scatter(plot_t, plot_gy, c="r")
            #     plt.ylabel("Vertical positions")
            #     plt.xlabel("Time (ms)")
            #     plt.show()
        
        # Write all measures to a CSV
        # ---------------------------
        df = pd.DataFrame(spatial_accuracies, index=[f"S{x:03d}" for x in np.arange(465) + 1], columns=["R1", "R2", "R3"])
        df.to_csv(accuracy_output_file)

        # Repeat for spatial precision measures
        # -------------------------------------
        df = pd.DataFrame(spatial_precisions, index=[f"S{x:03d}" for x in np.arange(465) + 1], columns=["R1", "R2", "R3"])
        df.to_csv(precision_output_file)
    
    # Create figure
    # -------------
    df = pd.read_csv(accuracy_output_file, index_col=0)
    spatial_accuracies = df.to_numpy()

    data = [[x for x in spatial_accuracies[:, i] if not np.isnan(x)] for i in range(3)]
    data = np.array(data, dtype=object)  # fix a VisibleDeprecationWarning in matplotlib about creating an ndarray from ragged nested sequences
    max_error = np.nanmax(spatial_accuracies)
    max_y = np.ceil(max_error)

    plt.figure(figsize=(3, 2.5), dpi=300)
    plt.boxplot(data, sym="r+")
    # plt.violinplot(data)
    plt.xticks(ticks=[1, 2, 3], labels=["R1", "R2", "R3"])
    plt.gca().yaxis.set_minor_locator(MultipleLocator(0.5))
    plt.ylabel("Spatial accuracy (dva)")
    plt.ylim([0, max_y])
    plt.xlabel("Recording round")
    plt.grid(which="major", axis="y")
    # plt.show()
    plt.tight_layout()
    plt.savefig("boxplot.png", bbox_inches="tight")

    # Repeat for spatial precision
    # ----------------------------
    df = pd.read_csv(precision_output_file, index_col=0)
    spatial_precisions = df.to_numpy()

    data = [[x for x in spatial_precisions[:, i] if not np.isnan(x)] for i in range(3)]
    data = np.array(data, dtype=object)  # fix a VisibleDeprecationWarning in matplotlib about creating an ndarray from ragged nested sequences
    max_error = np.nanmax(spatial_precisions)
    max_y = np.ceil(max_error)

    plt.figure(figsize=(3, 2.5), dpi=300)
    plt.boxplot(data, sym="r+")
    # plt.violinplot(data)
    plt.xticks(ticks=[1, 2, 3], labels=["R1", "R2", "R3"])
    plt.gca().yaxis.set_minor_locator(MultipleLocator(0.5))
    plt.ylabel("Spatial precision, log scale (dva)")
    plt.ylim([1e-2, max_y])
    plt.yscale("log")
    plt.xlabel("Recording round")
    plt.grid(which="major", axis="y")
    # plt.show()
    plt.tight_layout()
    plt.savefig("boxplot_precision.png", bbox_inches="tight")
