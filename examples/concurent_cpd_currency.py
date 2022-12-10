import multiprocessing
import argparse
import os

from settings.default import (
    CURRENCY_TICKERS,
    CPD_CURRENCY_OUTPUT_FOLDER,
    CPD_DEFAULT_LBW,
    CPD_DEFAULT_KERNEL
)
from config import *

N_WORKERS = len(CURRENCY_TICKERS)
# N_WORKERS = 1


def main(args):
    if not os.path.exists(CPD_CURRENCY_OUTPUT_FOLDER(args.lookback_window_length, args.kernel_choice)):
        os.mkdir(CPD_CURRENCY_OUTPUT_FOLDER(args.lookback_window_length, args.kernel_choice))

    print("="*20)
    print("Kernel choice:", args.kernel_choice)
    print("="*20)

    all_processes = [
        f'python -m examples.cpd_quandl "{ticker}" \
          "{os.path.join(CPD_CURRENCY_OUTPUT_FOLDER(args.lookback_window_length, args.kernel_choice), ticker + ".csv")}" \
          "1990-01-01" "2020-01-01" "{args.lookback_window_length}" "{args.kernel_choice}"'
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
        parser.add_argument(
            "kernel_choice",
            metavar="k",
            type=str,
            nargs="?",
            default=CPD_DEFAULT_KERNEL,
            help="Kernel choice",
        )
        return parser.parse_args()

    main(get_args())
