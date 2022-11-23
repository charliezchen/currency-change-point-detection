import multiprocessing
import argparse
import os

from settings.default import (
    QUANDL_TICKERS,
    CPD_QUANDL_OUTPUT_FOLDER,
    CPD_DEFAULT_LBW,
)

N_WORKERS = len(QUANDL_TICKERS)


def main(lookback_window_length: int, kernel: str):
    if not os.path.exists(CPD_QUANDL_OUTPUT_FOLDER(lookback_window_length, kernel)):
        os.mkdir(CPD_QUANDL_OUTPUT_FOLDER(lookback_window_length, kernel))

    all_processes = [
        f'python -m examples.cpd_quandl "{ticker}" "{os.path.join(CPD_QUANDL_OUTPUT_FOLDER(lookback_window_length, kernel), ticker + ".csv")}" "1990-01-01" "2021-12-31" "{lookback_window_length}" --kernel {kernel}'
        for ticker in QUANDL_TICKERS
    ]
    print(all_processes)
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
            help="Choose from Matern52, Matern32, Matern12, "
        )
        return [
            parser.parse_known_args()[0].lookback_window_length,
            parser.parse_known_args()[0].kernel,
        ]

    main(*get_args())
