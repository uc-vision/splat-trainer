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
import redis
from rq import Worker
from rq.command import send_shutdown_command
from concurrent.futures import ThreadPoolExecutor, as_completed
from paramiko.ssh_exception import SSHException, AuthenticationException



@dataclass
class Machine:
  hostname:str
  err:Optional[str] = None
  msg:Optional[str] = None

  
def deploy_group(workers:List[str], connect_kwargs):

  def deploy(worker: str):
    host = socket.gethostname()
    try:
      with fabric.Connection(worker, connect_kwargs=connect_kwargs) as c:
        command = """
          export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/py310_cu121_worker
          source ~/.bashrc
          conda activate splat-trainer-py10
          cd ~/splat-trainer
          mkdir -p ./log
          rq worker -c splat_trainer.config.settings --serializer splat_trainer.util.deploy.cloudpickle > ./log/{worker}.log 2>&1
          """.format(host=host, worker=worker)
        # try:
        c.run(command, hide="stdout", asynchronous=True)
        return Machine(worker, msg=f"RQ worker started on {worker}")
        
        # except Exception as e:
        #   raise RuntimeError(f"{str(e)}")


    except (SSHException, socket.error) as e:
      return Machine(worker, err=str(e))


  with ThreadPoolExecutor() as pool:
    return list(pool.map(deploy, workers))
  

def deploy_all(config):
  connect_kwargs = get_connect_keys()

  result = {name:deploy_group(workers, connect_kwargs)  
          for name, workers in config['groups'].items()}

  count = 0
  machines = [machine for name, group in result.items() for machine in group]
  for machine in machines:
    if machine.msg:
      count += 1
    else:
      print(f"{machine.hostname}: {machine.err}")
  print(f"\nRQ workers started on {count}/{len(machines)} worker machines.\n")

  return result


def deployer(config_file: str):
  config_path = Path(__file__).parent.parent / 'config'
  config = read_config(config_path / f"{config_file}.yaml")

  return partial(deploy_all, config)


def deploy_workers(config_file: str):
  deploy = deployer(config_file)
  return deploy()  


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


def get_all_workers(redis_url):
  redis_conn = redis.from_url(redis_url)
  workers = Worker.all(connection=redis_conn)

  return workers, redis_conn


def shutdown_all_workers(redis_url):
  workers, redis_conn = get_all_workers(redis_url)
  
  for worker in workers:
    send_shutdown_command(redis_conn, worker.name)




