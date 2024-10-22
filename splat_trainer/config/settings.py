import cloudpickle
import os
import socket
import uuid

HOSTNAME = socket.gethostname()

REDIS_URL = f'redis://cs24004kw:6379'
QUEUES = ['default']

NAME = f'{HOSTNAME}_{str(uuid.uuid4())}'

