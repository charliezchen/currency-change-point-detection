import multiprocessing
import argparse
import os

from settings.default import (
    CURRENCY_TICKERS,
    CPD_CURRENCY_OUTPUT_FOLDER,
    CPD_DEFAULT_LBW,
)
from config import *

N_WORKERS = len(CURRENCY_TICKERS)


def main(lookback_window_length: int):
    if not os.path.exists(CPD_CURRENCY_OUTPUT_FOLDER(lookback_window_length, KERNEL_CHOICE)):
        os.mkdir(CPD_CURRENCY_OUTPUT_FOLDER(lookback_window_length, KERNEL_CHOICE))

    print("="*20)
    print("Kernel choice:", KERNEL_CHOICE)
    print("="*20)

    all_processes = [
        f'python -m examples.cpd_quandl "{ticker}" \
          "{os.path.join(CPD_CURRENCY_OUTPUT_FOLDER(lookback_window_length, KERNEL_CHOICE), ticker + ".csv")}" \
          "1990-01-01" "2020-01-01" "{lookback_window_length}"'
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
        return [
            parser.parse_known_args()[0].lookback_window_length,
        ]

    main(*get_args())
