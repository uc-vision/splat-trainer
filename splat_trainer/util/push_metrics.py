import time
from typing import Optional

from fabric import Connection
import redis

from grafana_analytics.plugins.util import GraphitePusher


LOCAL_PORT = 2003
DOCKER_HOST = '172.18.0.2'
REMOTE_HOST = 'csse-maaratech1'
REMOTE_PORT = 2003

def push_metrics(averaged_results: Optional[dict]=None):
    
    if not averaged_results:
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        averaged_results = redis_client.hgetall("multirun_result:1")

    graphite_pusher = GraphitePusher(hostname='127.0.0.1', prefix='training_stats.splat_trainer.')

    with Connection(REMOTE_HOST, user='maara').forward_local(LOCAL_PORT, REMOTE_PORT, DOCKER_HOST):
        timestamp = int(time.time())
        for metric, value in averaged_results.items():
            graphite_pusher.push_stat(metric, float(value), timestamp)

        time.sleep(10)
        
      
       
        
        
if __name__ == "__main__":
    push_metrics()