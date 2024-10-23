import argparse
import signal
import socket
import sys
import importlib.resources

import rq_dashboard
from flask import Flask, render_template

from splat_trainer.util.deploy import deploy_workers, shutdown_all_workers


REDIS_URL = None


with importlib.resources.path('splat_trainer', 'templates') as template_folder:
  app = Flask(__name__, template_folder=str(template_folder))


def set_rq_dashboard(app, redis_url):
  app.config.from_object(rq_dashboard.default_settings)
  app.config['RQ_DASHBOARD_REDIS_URL'] = redis_url
  rq_dashboard.web.setup_rq_connection(app)
  app.register_blueprint(rq_dashboard.blueprint, url_prefix="/rq")


data = {}


@app.route('/')
def index():
  global data
  return render_template('index.html', data=data)


def signal_handler(signum, frame):
  global REDIS_URL

  try:
      print("Shutting down all workers and exiting.")
      shutdown_all_workers(REDIS_URL)

  except Exception as e:
      print(f"Error while shutting down workers: {e}")

  finally:
      sys.exit(0)


def get_args():
  args = argparse.ArgumentParser(description='Start rq workers')
  args.add_argument('--config', default="test", help='List of host files to monitor')  
  args.add_argument('--getpass', action="store_true", help='Prompt for password')  
  args.add_argument('--flask_port', default=8000, help='Port number on which Flask app runs')
  args.add_argument('--redis_port', default=6379, help='Port number on which Redis server runs')

  parsed_args = args.parse_args()  
  hostname = socket.gethostname()
  parsed_args.redis_url = f"redis://{hostname}:{parsed_args.redis_port}"

  return  parsed_args



def main():
  global data
  global REDIS_URL

  args = get_args()
  REDIS_URL = args.redis_url

  set_rq_dashboard(app, REDIS_URL)

  result = deploy_workers(args)
  data.update(result)

  signal.signal(signal.SIGINT, signal_handler)
  signal.signal(signal.SIGTSTP, signal_handler)

  app.run(host="0.0.0.0", debug=True, port=args.flask_port, use_reloader=False)


if __name__ == '__main__':
    main()