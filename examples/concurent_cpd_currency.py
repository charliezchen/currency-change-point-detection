import multiprocessing
import argparse
import os

from settings.default import (
    CURRENCY_TICKERS,
    CPD_CURRENCY_OUTPUT_FOLDER,
    CPD_DEFAULT_LBW,
)

N_WORKERS = len(CURRENCY_TICKERS)


def main(lookback_window_length: int, kernel: str, num_mixtures: int):
    if kernel == "SpectralMixture":
        result_path = CPD_CURRENCY_OUTPUT_FOLDER(lookback_window_length, kernel) + f'_Q{num_mixtures}'
    else:
        result_path = CPD_CURRENCY_OUTPUT_FOLDER(lookback_window_length, kernel)

    if not os.path.exists(result_path):
        os.mkdir(result_path)

    print(f"current kernel: {kernel}")
    print(f"lookback_window_length: {lookback_window_length}")

    all_processes = [
        f'python -m examples.cpd_quandl "{ticker}" "{os.path.join(result_path, ticker + ".csv")}" "1990-01-01" "2020-01-01" "{lookback_window_length}" --kernel {kernel} --num_mixtures {num_mixtures}'
        for ticker in CURRENCY_TICKERS
    ]
    process_pool = multiprocessing.Pool(processes=N_WORKERS)
    process_pool.map(os.system, all_processes)


if __name__ == "__main__":

    def get_args():
        """Returns settings from command line."""

        parser = argparse.ArgumentParser(
            description="Run changepoint detection module for all tickers"
        )
        parser.add_argument(
            "lookback_window_length",
            metavar="l",
            type=int,
            nargs="?",
            default=CPD_DEFAULT_LBW,
            help="CPD lookback window length",
        )
        parser.add_argument("--kernel",
            default="Matern32",
            help="Choose from Matern52, Matern32, Matern12, SpectralMixture, Matern12_32, Matern12_52"
        )
        parser.add_argument("--num_mixtures",
            default=5,
            type=int,
            help="Choose number of mixtures for the Spectral Mixture Kernel"
        )
        return [
            parser.parse_known_args()[0].lookback_window_length,
            parser.parse_known_args()[0].kernel,
            parser.parse_known_args()[0].num_mixtures,
        ]

    main(*get_args())
