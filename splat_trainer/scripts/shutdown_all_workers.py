import socket
from splat_trainer.util.deploy import shutdown_all_workers, shutdown_all, get_connect_keys, read_config
from pathlib import Path



def main():
    hostname = socket.gethostname()
    shutdown_all_workers(f'redis://{hostname}:6379')
    
    config_path = Path(__file__).parent.parent / 'config'
    config = read_config(config_path / f"all.yaml")

    connect_kwargs = get_connect_keys(False)
    hosts = [ host for hosts in config['groups'].values() for host in hosts ]

    shutdown_all(hosts, connect_kwargs)
    print("All RQ Workers are shutdown.")


if __name__ == "__main__":
    main()