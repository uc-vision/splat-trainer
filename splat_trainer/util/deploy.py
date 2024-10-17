import argparse
import datetime
import socket
from dataclasses import dataclass
from functools import partial
from getpass import getpass, getuser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cloudpickle
import fabric
import keyring
import yaml
from redis import Redis
from rq import Worker
from concurrent.futures import ThreadPoolExecutor, as_completed
from paramiko.ssh_exception import SSHException, AuthenticationException



@dataclass
class Machine:
  hostname:str
  err:Optional[str] = None
  msg:Optional[str] = None


def rq_worker():
  parser = argparse.ArgumentParser()
  parser.add_argument("--hostname", type=str, required=True, help="The hostname to connect to")
  parser.add_argument("--port", type=int, default=6379, help="Port number")
  args = parser.parse_args()

  redis_url = f'redis://{args.hostname}:{args.port}'
  w = Worker(['default'], connection=Redis.from_url(redis_url), serializer=cloudpickle) 
  w.work()


def deploy(worker: str):
  host = socket.gethostname()
  connect_kwargs = get_connect_keys()

  try:
    with fabric.Connection(worker, connect_kwargs=connect_kwargs) as c:
      command = """
        source ~/.bashrc
        conda activate splat-trainer
        cd ~/splat-trainer
        mkdir -p ./log
        rq-worker --host {host} > ./log/{worker}.log 2>&1
        """.format(host=host, worker=worker)
      c.run(command, hide=True, asynchronous=True)
      return Machine(worker, msg=f"RQ worker started on {worker}")

  except (SSHException, socket.error) as e:
    return Machine(worker, err=str(e))

  
def deploy_group(workers:List[str]):
  with ThreadPoolExecutor() as pool:
    return list(pool.map(deploy, workers))
  

def deploy_all(config):
  return {name:deploy_group(workers)  
          for name, workers in config['groups'].items()}


def deployer(args):

  config_path = Path(__file__).parent.parent / 'config'
  config = read_config(config_path / f"{args.config}.yaml")

  return partial(deploy_all, config)


def deploy_workers(args):
  deploy = deployer(args)
  return deploy()



def get_args():
  args = argparse.ArgumentParser(description='Monitor lab computers')
  args.add_argument('--config', default="test", help='List of host files to monitor')  
  args.add_argument('--port', default=8000, help='Specify the port on which Flask app runs')

  return args.parse_args()


def get_connect_keys():
  connect_kwargs = {}
  user = getuser()

  password = keyring.get_password('deploy_workers', user)

  if password is None:
    password = getpass()
    keyring.set_password('deploy_workers', user, password)

  connect_kwargs=dict(password=password)

  return connect_kwargs


def read_config(config_path):
  with open(config_path, 'r') as f:
      return yaml.safe_load(f)


def terminate_all(data: dict[str: List[Machine]]):
  machines = [machine for name, group in data.items() for machine in group]
  
  if any(machine.msg for machine in machines):
    print("\nTerminating rq worker processes...")
    result = {name:terminate_group(workers)  
            for name, workers in data.items()}


    for name, group in result.items():
      for machine in group:
        assert (machine.msg == f"RQ worker terminated on {machine.hostname}") \
          or (machine.msg is None and machine.err is not None), \
          f"Warning: RQ worker is not terminated on Machine {machine.hostname} properly."

    print("\nRQ workers terminated on worker machines successfully.\n")
    return result
  

def terminate_group(workers: List[Machine]):
  with ThreadPoolExecutor() as pool:
    return list(pool.map(terminate, workers))


def terminate(worker: Machine):
  connect_kwargs = get_connect_keys()
  if worker.msg:
    try:
      with fabric.Connection(worker.hostname, connect_kwargs=connect_kwargs) as c:
        command = """
          pkill -f rq:worker
          pkill -f splat-trainer
          """
        c.run(command, hide=True, asynchronous=True)
        return Machine(worker, msg=f"RQ worker terminated on {worker}")

    except (SSHException, socket.error) as e:
      return Machine(worker, err=str(e))

  else:
    return Machine(worker)



