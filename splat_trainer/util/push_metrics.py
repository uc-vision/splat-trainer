import keyring
from getpass import getpass, getuser
import socket
import time
from typing import Optional

from fabric import Connection
import redis


LOCAL_PORT = 2003
DOCKER_HOST = '172.18.0.2'
REMOTE_HOST = 'csse-maaratech1'
REMOTE_PORT = 2003

def push_metrics(averaged_results: Optional[dict]=None):
    
    if not averaged_results:
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        averaged_results = redis_client.hgetall("multirun_result:1")

    def push_stat(name, value, timestamp):
        assert isinstance(value, (float, int)), f"Invalid value: {value}"
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('127.0.0.1', LOCAL_PORT))
            sock.sendall(f"training_stats.splat_trainer.{name} {value} {timestamp}\n".encode())
        except Exception as e:
            print(f"Error sending metric {name}: {e}")
        finally:
            sock.close()

    with Connection(REMOTE_HOST, user='maara', connect_kwargs=get_password()).forward_local( LOCAL_PORT, REMOTE_PORT, DOCKER_HOST, "127.0.0.1" ):
        timestamp = int(time.time())
        for metric, value in averaged_results.items():
            push_stat(metric, float(value), timestamp)

        time.sleep(10)
        

def get_password():
    connect_kwargs = {}
    user = getuser()
    password = keyring.get_password(REMOTE_HOST, user)

    if password is None:
        password = getpass()
        keyring.set_password(REMOTE_HOST, user, password)

    connect_kwargs=dict(password=password)
    return connect_kwargs
      
        
        
if __name__ == "__main__":
    push_metrics()