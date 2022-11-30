import multiprocessing
import argparse
import os

from settings.default import (
    CURRENCY_TICKERS,
    CPD_CURRENCY_OUTPUT_FOLDER,
    CPD_DEFAULT_LBW,
)

N_WORKERS = len(CURRENCY_TICKERS)


def main(lookback_window_length: int, kernel: str):
    if not os.path.exists(CPD_CURRENCY_OUTPUT_FOLDER(lookback_window_length, kernel)):
        os.mkdir(CPD_CURRENCY_OUTPUT_FOLDER(lookback_window_length, kernel))

    print(f"current kernel: {kernel}")
    print(f"lookback_window_length: {lookback_window_length}")
    all_processes = [
        f'python -m examples.cpd_quandl "{ticker}" "{os.path.join(CPD_CURRENCY_OUTPUT_FOLDER(lookback_window_length, kernel), ticker + ".csv")}" "1990-01-01" "2020-01-01" "{lookback_window_length}" --kernel {kernel}'
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
            help="Choose from Matern52, Matern32, Matern12, SpectralMixture"
        )
        return [
            parser.parse_known_args()[0].lookback_window_length,
            parser.parse_known_args()[0].kernel,
        ]

    main(*get_args())
