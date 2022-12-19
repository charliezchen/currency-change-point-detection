import multiprocessing
import argparse
import os

from settings.default import (
    CURRENCY_TICKERS,
    CPD_CURRENCY_OUTPUT_FOLDER,
    CPD_DEFAULT_LBW,
    CPD_DEFAULT_KERNEL
)
import pickle


N_WORKERS = len(CURRENCY_TICKERS)


def main(args):
    path = CPD_CURRENCY_OUTPUT_FOLDER(args.start_date,
                                      args.end_date,
                                    args.lookback_window_length, 
                                    args.kernel_choice)
    if not os.path.exists(path):
        os.mkdir(path)

    print("="*20)
    print("Kernel choice:", args.kernel_choice)
    print("="*20)

    if args.all_cur:
        command = [f'python -m examples.cpd_quandl "all_cur" \
          "data/currency_all_cur/all_cur_{args.start_date}_{args.end_date}_lbw{args.lookback_window_length}_{args.kernel_choice}.csv" \
          "{args.start_date}" "{args.end_date}" \
          "{args.lookback_window_length}" "{args.kernel_choice}"']
        process_pool = multiprocessing.Pool(processes=1)
        process_pool.map(os.system, command)
        return

    all_processes = [
        f'python -m examples.cpd_quandl "{ticker}" \
          "{os.path.join(path, ticker + ".csv")}" \
          "{args.start_date}" "{args.end_date}" \
          "{args.lookback_window_length}" "{args.kernel_choice}"'
        for ticker in CURRENCY_TICKERS
    ]
    process_pool = multiprocessing.Pool(processes=N_WORKERS if not args.debug else 1)
    process_pool.map(os.system, all_processes)


if __name__ == "__main__":

    def get_args():
        """Returns settings from command line."""

        parser = argparse.ArgumentParser(
            description="Run changepoint detection module for all tickers"
        )
        parser.add_argument(
            "-l", "--lookback_window_length",
            type=int,
            nargs="?",
            default=CPD_DEFAULT_LBW,
            help="CPD lookback window length",
        )
        parser.add_argument(
            "-k", "--kernel_choice",
            type=str,
            nargs="?",
            default=CPD_DEFAULT_KERNEL,
            help="Kernel choice",
        )
        parser.add_argument('--start_date', 
                            type=str, 
                            default="1990-01-01",
                            help="start date for cpd")

        parser.add_argument('--end_date', 
                            type=str, 
                            default="2020-01-01",
                            help="end date for cpd")


        group = parser.add_mutually_exclusive_group()
        group.add_argument('--save_parameter', action='store_true', help='save parameter')
        group.add_argument('--load_parameter', action='store_true', help='load parameter')

        parser.add_argument('--debug', action='store_true', help='debug mode')
        parser.add_argument('--verbose', action='store_true', help='verbose mode')

        parser.add_argument('--all_cur', action='store_true', help='run all currency')


        parser.add_argument('--load_parameter_path', 
                            type=str, 
                            default="data/currency_all_cur/all_cur_%s_%s_lbw%d_%s_param.csv",
                            help="The path to load parameter")
        
        parser.add_argument('--experiment_name', 
                            type=str, 
                            default="default_experiment",
                            help="The name for the experiment")
        args = parser.parse_args()

        args.load_parameter_path = args.load_parameter_path  % \
                            (args.start_date, 
                            args.end_date, 
                            args.lookback_window_length, 
                            args.kernel_choice)

        with open("args.pkl", 'wb') as f:
            pickle.dump(args, f)

        return args

    main(get_args())
