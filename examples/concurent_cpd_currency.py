import multiprocessing
import argparse
import os

from settings.default import (
    CURRENCY_TICKERS,
    CPD_CURRENCY_OUTPUT_FOLDER,
    CPD_DEFAULT_LBW,
)

N_WORKERS = len(CURRENCY_TICKERS)


def main(lookback_window_length: int, kernel: str, num_mixtures: int, testing: bool, linear: bool):
    if kernel == "SpectralMixture":
        result_path = CPD_CURRENCY_OUTPUT_FOLDER(lookback_window_length, kernel) + f'_Q{num_mixtures}'
    else:
        result_path = CPD_CURRENCY_OUTPUT_FOLDER(lookback_window_length, kernel)

    # Product with Linear Kernel
    if linear:
        result_path += "_linear" 
    else:
        pass 
    
    # Testing code
    if testing:
        result_path += "_testing"
        start_d = "2020-01-02"
        end_d = "2022-11-14"
    else:
        start_d = "1990-01-01"
        end_d = "2020-01-01"

    if not os.path.exists(result_path):
        os.mkdir(result_path)

    print(f"current kernel: {kernel}")
    print(f"lookback_window_length: {lookback_window_length}")

    all_processes = [
        f'python -m examples.cpd_quandl "{ticker}" "{os.path.join(result_path, ticker + ".csv")}" {start_d} {end_d} "{lookback_window_length}" --kernel {kernel} --num_mixtures {num_mixtures}  --save_path {os.path.join(result_path, ticker+"_params.csv")} --testing {testing} --linear {linear}'
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
        parser.add_argument("--testing",
            action='store_true',
            help="Choose number of mixtures for the Spectral Mixture Kernel"
        )
        parser.add_argument("--linear",
            action='store_true',
            help="If specified, all Matern kernels are multiplied with linear kernels"
        )
        return [
            parser.parse_known_args()[0].lookback_window_length,
            parser.parse_known_args()[0].kernel,
            parser.parse_known_args()[0].num_mixtures,
            parser.parse_known_args()[0].testing,
            parser.parse_known_args()[0].linear,
        ]

    main(*get_args())
