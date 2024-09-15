import argparse
import cloudpickle
from redis import Redis
from rq import Worker



def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--hostname", type=str, required=True, help="The hostname to connect to")
  parser.add_argument("--port", type=int, default=6379, help="Port number")
  args = parser.parse_args()

  redis_url = f'redis://{args.hostname}:{args.port}'
  w = Worker(['default'], connection=Redis.from_url(redis_url), serializer=cloudpickle) 
  w.work()
  

if __name__ == "__main__":
  main()