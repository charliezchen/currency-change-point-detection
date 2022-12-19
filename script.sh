
sh one_setup.sh 21 Matern12 > log1.txt 2>&1 &
sh one_setup.sh 63 Matern12 > log2.txt 2>&1 &
sh one_setup.sh 21 Matern32 > log3.txt 2>&1 &
sh one_setup.sh 63 Matern32 > log4.txt 2>&1 &

wait
