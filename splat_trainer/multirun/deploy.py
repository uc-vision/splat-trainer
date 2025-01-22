import socket
from dataclasses import dataclass
from functools import partial
from getpass import getpass, getuser
import logging
from pathlib import Path
from typing import Dict, List, Optional

import cloudpickle
import fabric
import keyring
import yaml
import redis
from redis import Redis
import uuid
from rq.worker import Worker, WorkerStatus
from rq.command import send_shutdown_command, send_kill_horse_command
from concurrent.futures import ThreadPoolExecutor
from paramiko.ssh_exception import SSHException


log = logging.getLogger(__name__)

@dataclass
class Machine:
  hostname:str
  err:Optional[str] = None
  msg:Optional[str] = None

  
def deploy_group(group_name: str, hosts:List[str], connect_kwargs, args):

  def deploy(host: str):

    try:
      with fabric.Connection(host, connect_kwargs=connect_kwargs) as c:
        worker_name = f'{host}__{group_name}__{str(uuid.uuid4())}'
        command = """
          export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/py311_cu121_{group_name}
          source ~/.bashrc
          conda activate splat-trainer
          cd ~/splat-trainer
          mkdir -p ./rq_worker_log
          rq worker default --url {redis_url} \\
                         --name {worker_name} \\
                         --serializer splat_trainer.multirun.deploy.cloudpickle \\
                         > ./rq_worker_log/{host}.log 2>&1
          """.format(group_name=group_name, 
                     redis_url=args.redis_url, 
                     worker_name=worker_name, 
                     host=host)

        c.run(command, hide="stdout", asynchronous=True)
        return Machine(host, msg=f"RQ worker started on {host}")
      

    except (SSHException, socket.error) as e:
      return Machine(host, err=str(e))


  with ThreadPoolExecutor() as pool:
    return list(pool.map(deploy, hosts))
  

def deploy_all(config, connect_kwargs, args):
 
  result = {group_name:deploy_group(group_name, hosts, connect_kwargs, args)  
          for group_name, hosts in config['groups'].items() if hosts}

  machines = [machine for group in result.values() for machine in group]
  
  count = 0
  for machine in machines:
    if machine.msg:
      count += 1
    else:
      log.info(f"{machine.hostname}: {machine.err}")

  log.info(f"\n{count}/{len(machines)} machines connected.")
  log.info(f"{count} RQ workers starts running...\n")

  return result


def deployer(args):
  config_path = Path(__file__).parents[1] / 'config/multirun/hosts'
  config = read_config(config_path / f"{args.config}.yaml")

  connect_kwargs = get_connect_keys(args.getpassword)

  return partial(deploy_all, config, connect_kwargs, args)


def deploy_workers(args):
  deploy = deployer(args)
  return deploy()  


def get_connect_keys(getpassword: bool):
  connect_kwargs = {}
  user = getuser()

  if getpassword:
    password = keyring.get_password('deploy_workers', user)

    if password is None:
      password = getpass()
      keyring.set_password('deploy_workers', user, password)

    connect_kwargs=dict(password=password)

  return connect_kwargs


def read_config(config_path):
  with open(config_path, 'r') as f:
      return yaml.safe_load(f)


def get_all_workers(redis: Redis):
  return Worker.all(redis)


def shutdown_all_workers(redis_url: str):
  redis_conn = redis.from_url(redis_url)
  all_workers = get_all_workers(redis_conn)
  
  log.info("Shutting down all workers and exiting...")
  
  for worker in all_workers:
    shutdown_worker(worker, redis_conn)
    

def shutdown_worker(worker, redis: Redis):
  try:
    if worker.state == WorkerStatus.BUSY:
      send_kill_horse_command(redis, worker.name)
      
    send_shutdown_command(redis, worker.name)
    log.info(f"Worker with pid {worker.pid} on {worker.hostname} has been shutdown.")
    
  except Exception as e:
    log.error(f"Error while shutting down worker on {worker.hostname} with PID {worker.pid}: {e}")
  
  

def shutdown_workers_on_host(redis_url: str, host: str):
  redis_conn = redis.from_url(redis_url)
  all_workers = get_all_workers(redis_conn)
  for worker in all_workers:
    if host in worker.name:
        shutdown_worker(worker, redis_conn)


def get_worker_num_on_host(host: str, redis_url: str):
  redis_conn = redis.from_url(redis_url)
  workers = get_all_workers(redis_conn)
  count = 0
  for worker in workers:
    if host in worker.name:
      count += 1
  return count


def flush_db(redis_url: str):
  redis_conn = redis.from_url(redis_url)
  redis_conn.flushdb()



