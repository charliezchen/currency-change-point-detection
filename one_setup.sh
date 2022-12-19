
python -m examples.concurent_cpd_currency -l $1 \
       --start_date 2020-01-01 --end_date 2023-01-01 \
       -k $2 --save_parameter --all_cur 
python -m examples.concurent_cpd_currency -l $1 \
       --start_date 2020-01-01 --end_date 2023-01-01 \
       -k $2 --load_parameter

