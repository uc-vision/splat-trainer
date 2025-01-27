import argparse
import fabric
import socket
import re
import redis
from paramiko.ssh_exception import SSHException, AuthenticationException
from concurrent.futures import ThreadPoolExecutor

from splat_trainer.multirun.deploy import shutdown_all_workers, get_connect_keys



def shutdown_all(hosts, connect_kwargs):
    def shutdown(host: str):
        try:
            with fabric.Connection(host, connect_kwargs=connect_kwargs) as c:
                command = "pgrep -f 'rq:worker'"
                result = c.run(command, hide=True, warn=True)
                num_rq_worker = len(result.stdout.strip().splitlines())
                print(f"Found {num_rq_worker} rq worker process running on {host}.")
                if num_rq_worker:
                    command = """
                                pkill -f rq:worker
                                pkill -f splat-trainer
                                """
                    c.run(command, hide=True, warn=True)
                    print(f"RQ worker terminated on {host}")

        except (SSHException, socket.error) as e:
            print(e)


    with ThreadPoolExecutor() as pool:
        return list(pool.map(shutdown, hosts))

    

def main():
    args = get_args()
    hostname = socket.gethostname()
    redis_url = f'redis://{hostname}:6379/0'
    shutdown_all_workers(redis_url)
    
    redis_conn = redis.from_url(redis_url, decode_responses=True)
    
    worker_keys = redis_conn.keys("rq:worker:*")
    if worker_keys:
        host_names = [re.search(r"rq:worker:([^_]+)__", key).group(1) for key in worker_keys if re.search(r"rq:worker:([^_]+)__", key)]
        connect_kwargs = get_connect_keys(args.getpass)
        shutdown_all(host_names, connect_kwargs)
        
        
def get_args():
    args = argparse.ArgumentParser(description='Shutdown rq workers')
    args.add_argument('--getpass', action="store_true", help='Prompt for password')  

    return args.parse_args()
    
    


if __name__ == "__main__":
    main()