import subprocess
import argparse
import re
import math
from src.configs import parse_configs

def main():
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--allocated_hosts",
        type=str,
        required=True,
        help="Allocated host nodes (no default)",
    )
    parser.add_argument(
        "--current_host",
        type=str,
        required=True,
        help="Current host node (no default)",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Experiment configurations file (no default)",
    )
    args = parser.parse_args()
    user_configs = parse_configs(args.config_file)

    print(f"Allocated: {args.allocated_hosts[2:-1]}")
    print(f"Current: {args.current_host}")
    print(f"Configs: {args.config_file}")
    
    current_host = [f"{args.current_host[0]}{s}" for s in re.findall(r'\d+', args.current_host)]
    if '[' not in args.allocated_hosts:
        allocated_hosts = [f"{args.allocated_hosts[0]}{num}" for num in expand_range(args.allocated_hosts[1:])] 
    else:
        allocated_hosts = [f"{args.allocated_hosts[0]}{num}" for num in expand_range(args.allocated_hosts[2:-1])] 
    
    print(f"Processed Allocated: {allocated_hosts}")
    print(f"Processed Current: {current_host}")
    

    if len(current_host) > 1:
        print("Something Went Horribly Wrong!!!")
        return
    else:
        # Compute how many process to allocate per host
        total_num_client = user_configs["SERVER_CONFIGS"]["MIN_NUM_CLIENTS"]
        clients_per_node = int(math.ceil(total_num_client / len(allocated_hosts)))
        clients_on_node0 = total_num_client - (clients_per_node * (len(allocated_hosts)-1))

        if current_host[0] == allocated_hosts[0]:
            # need to run server on the first
            # allocated node of the cluster
            server_call = f'python src/run_fl_server.py --server_address="0.0.0.0:59999" --config_file="{args.config_file}"'
            server_proc = subprocess.Popen([server_call], shell=True)
            client_call = f'python src/run_fl_clients.py --server_address="127.0.0.1:59999" --total_clients={total_num_client} --num_clients={clients_on_node0} --start_cid=0 --config_file="{args.config_file}"'
            client_proc = subprocess.Popen([client_call], shell=True)
            # Wait for processes to terminate
            client_proc.wait()
            server_proc.wait()
        else:
            start_client_id = clients_on_node0 + (clients_per_node * (allocated_hosts.index(current_host[0]) - 1))
            # need to run client instance on
            # all other allocated nodes
            client_call = f'python src/run_fl_clients.py --server_address="{allocated_hosts[0]}:59999" --total_clients={total_num_client} --num_clients={clients_per_node} --start_cid={start_client_id} --config_file="{args.config_file}"'
            print(f"Client Call: {client_call}")
            subprocess.call([client_call], shell=True)

def expand_range(in_str):
    list_range = []
    for token in in_str.split(','):
        if '-' not in token:
            list_range.append(int(token))
        else:
            low, high = map(int, token.split('-'))
            list_range += range(low, high+1)
    return list_range

if __name__=="__main__":
    main()
