import datetime
import socket
from collections import Counter
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
import uuid
from rq.worker import Worker, WorkerStatus
from rq.command import send_shutdown_command, send_kill_horse_command
from concurrent.futures import ThreadPoolExecutor, as_completed
from paramiko.ssh_exception import SSHException, AuthenticationException



@dataclass
class Machine:
  hostname:str
  err:Optional[str] = None
  msg:Optional[str] = None

  
def deploy_group(group_name: str, hosts:List[str], connect_kwargs, args):

  def deploy(host: str):
    count = get_worker_num(host, args.redis_url)

    try:
      with fabric.Connection(host, connect_kwargs=connect_kwargs) as c:
        check_existing_worker = f"pgrep -f 'rq worker --url {args.redis_url}'"
        result = c.run(check_existing_worker, warn=True, hide="stdout")

        if result.ok and result.stdout.strip():
          worker_pids = result.stdout.strip().splitlines()

          if len(worker_pids) > count or count > args.max_num_worker:
            for worker_pid in worker_pids:
              kill_command = f"kill -9 {worker_pid}"
              c.run(kill_command)
              print(f"Killed existing RQ worker with PID {worker_pid.strip()} on {host}")

          if count == args.max_num_worker:
            return Machine(host, msg=f"RQ worker already exists on {host}")

        worker_name = f'{host}_{str(uuid.uuid4())}'
        command = """
          export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/py310_cu121_{group_name}
          source ~/.bashrc
          conda activate splat-trainer-py10
          cd ~/splat-trainer
          mkdir -p ./log
          rq worker --url {redis_url} \\
                    --name {worker_name} \\
                    --serializer splat_trainer.util.deploy.cloudpickle \\
                    > ./log/{host}.log 2>&1
          """.format(group_name=group_name, redis_url=args.redis_url, worker_name=worker_name, host=host)

        # TODO: add logic to catch asynchronous error
        # try:
        c.run(command, hide="stdout", asynchronous=True)
        return Machine(host, msg=f"RQ worker started on {host}")
    
        # except Exception as e:
        #   raise SystemExit(f"{str(e)}")
        #   shutdown_all_workers(args.redis_url)

    except (SSHException, socket.error) as e:
      return Machine(host, err=str(e))


  with ThreadPoolExecutor() as pool:
    return list(pool.map(deploy, hosts))
  

def deploy_all(config, connect_kwargs, args):
  flush_all(args.redis_url)
 
  result = {group_name:deploy_group(group_name, hosts, connect_kwargs, args)  
          for group_name, hosts in config['groups'].items() if hosts}

  machines = [machine for group_name, group in result.items() for machine in group]

  count = 0
  machine_names = []
  for machine in machines:
    if machine.msg:
      count += 1
      machine_names.append(machine.hostname)
    else:
      print(f"{machine.hostname}: {machine.err}")

  duplicate_counts = Counter(machine_names)
  max_duplicate = max(duplicate_counts.values())
  assert max_duplicate <= args.max_num_worker, f"Maximum {max_num_worker} rq workers allowed on each machine, got {max_duplicate}."

  print(f"\n{len(machine_names)}/{len(machines)} machines connected.")
  print(f"{count} RQ workers starts running...\n")

  return result


def deployer(args):
  config_path = Path(__file__).parent.parent / 'config'
  config = read_config(config_path / f"{args.config}.yaml")

  connect_kwargs = get_connect_keys(args.getpass)

  return partial(deploy_all, config, connect_kwargs, args)


def deploy_workers(args):
  deploy = deployer(args)
  return deploy()  


def get_connect_keys(getpass: bool):
  connect_kwargs = {}
  user = getuser()

  if getpass:
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

  return workers


def shutdown_all_workers(redis_url):
  redis_conn = redis.from_url(redis_url)
  workers = get_all_workers(redis_url)
  
  for worker in workers:
    if worker.state == WorkerStatus.BUSY:
      send_kill_horse_command(redis_conn, worker.name)

    send_shutdown_command(redis_conn, worker.name)


def get_worker_num(name: str, redis_url: str):
  workers = get_all_workers(redis_url)
  count = 0
  for worker in workers:
    if name in worker.name:
      count += 1
  return count


def flush_all(redis_url: str):
  redis_conn = redis.from_url(redis_url)
  redis_conn.flushall()




