import subprocess
import argparse
import re

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
        "--configs_file",
        type=str,
        required=True,
        help="Experiment configurations file (no default)",
    )
    args = parser.parse_args()

    allocated_hosts = [f"{args.allocated_hosts[0]}{s}" for s in re.findall(r'\d+', args.allocated_hosts)]
    current_host = [f"{args.current_host[0]}{s}" for s in re.findall(r'\d+', args.current_host)]
    
    print(f"Allocated: {allocated_hosts}")
    print(f"Current: {current_host}")
    print(f"Configs: {args.configs_file}")

    if len(current_host) > 1:
        print("Something Went Horribly Wrong!!!")
        return
    else:

        if current_host[0] == allocated_hosts[0]:
            # need to run server on the first
            # allocated node of the cluster
            server_call = f'python src/run_fl_server.py --server_address="0.0.0.0:59999" --config_file="{args.configs_file}"'
            print(f"Client Call: {server_call}")
            subprocess.call([server_call], shell=True)
        else:
            # need to run client instance on
            # all other allocated nodes
            client_call = f'python src/run_fl_clients.py --server_address="{allocated_hosts[0]}:59999" --total_clients={len(allocated_hosts)} --num_clients=1 --start_cid={allocated_hosts.index(current_host[0])-1} --config_file="{args.configs_file}"'
            print(f"Client Call: {client_call}")
            subprocess.call([client_call], shell=True)

if __name__=="__main__":
    main()
