usage: SA-tracert.py [-h] [--runs RUNS] [--max-ttl MAX_TTL] [--timeout TIMEOUT] [--sa-n SA_N] [--sa-t0 SA_T0] [--sa-alpha SA_ALPHA]
                     [--sa-tmin SA_TMIN] [--csv-dir CSV_DIR] [--no-plot] [--SA]
                     target

positional arguments:
  target               Destination hostname or IP

options:
  -h, --help           show this help message and exit
  --runs RUNS          Number of traceroute runs
  --max-ttl MAX_TTL    Max TTL
  --timeout TIMEOUT    Probe timeout (s)
  --sa-n SA_N          SA iterations per temp
  --sa-t0 SA_T0        SA initial temp
  --sa-alpha SA_ALPHA  SA cooling rate
  --sa-tmin SA_TMIN    SA min temp
  --csv-dir CSV_DIR    Directory to save CSV outputs
  --no-plot            Disable graph plotting
  --SA                 Enable Simulated Annealing optimization


example:
mặc định (dijkstra):
>python SA-tracert.py vitinhcatan.com --runs 30 --max-ttl=30 --timeout 1

giải thuật SA với các tham số:
--sa-t0 = 1000		nhiệt độ ban đầu = 1000
--sa-alpha = 0.05	tỉ lệ giảm nhiệt độ = 0.05 (5%)
--sa-n = 200		nhiệt độ kết thúc = 200
>python SA-tracert.py vitinhcatan.com --runs 30 --max-ttl=30 --timeout 1 -SA --sa-t0 1000 --sa-alpha 0.05 --sa-n 200