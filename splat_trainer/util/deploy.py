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
from concurrent.futures import ThreadPoolExecutor, as_completed
from paramiko.ssh_exception import SSHException, AuthenticationException



@dataclass
class Machine:
  hostname:str
  error:Optional[str] = None
  result:Optional[str] = None


def read_config(config_path):
  with open(config_path, 'r') as f:
      return yaml.safe_load(f)

  
def deploy_group(workers:str, connect_kwargs):

  def deploy(worker: str):
      host = socket.gethostname()

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
          return Machine(worker, result=f"RQ worker started on {worker}")

      except (SSHException, socket.error) as e:
        return Machine(worker, error=str(e))


  with ThreadPoolExecutor() as pool:
    return list(pool.map(deploy, workers))
  

def deploy_all(config, connect_kwargs):
  return {name:deploy_group(workers, connect_kwargs)  
          for name, workers in config['groups'].items()}


def deploy_workers(args):
    config_path = Path(__file__).parent.parent / 'config'
    config = read_config(config_path / f"{args.config}.yaml")

    connect_kwargs = {}
    user = getuser()

    password = keyring.get_password('deploy_workers', user)

    if password is None:
      password = getpass()
      keyring.set_password('deploy_workers', user, password)

    connect_kwargs=dict(password=password)

    return partial(deploy_all, config, connect_kwargs)


def get_args():
    args = argparse.ArgumentParser(description='Monitor lab computers')
    args.add_argument('--config', default="test", help='List of host files to monitor')  

    return args.parse_args()