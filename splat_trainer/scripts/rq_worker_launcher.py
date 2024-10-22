import argparse
import signal
import importlib.resources

import rq_dashboard
from flask import Flask, render_template

from splat_trainer.util.deploy import deploy_workers, shutdown_all_workers


REDIS_URL = "redis://localhost:6379/0"


with importlib.resources.path('splat_trainer', 'templates') as template_folder:
  app = Flask(__name__, template_folder=str(template_folder))

app.config.from_object(rq_dashboard.default_settings)
app.config['RQ_DASHBOARD_REDIS_URL'] = REDIS_URL
rq_dashboard.web.setup_rq_connection(app)
app.register_blueprint(rq_dashboard.blueprint, url_prefix="/rq")

data = {}


@app.route('/')
def index():
  global data
  return render_template('index.html', data=data)


def signal_handler(signum, frame):
  shutdown_all_workers(REDIS_URL)

  exit(0)


def get_args():
  args = argparse.ArgumentParser(description='Start rq workers')
  args.add_argument('--config', default="test", help='List of host files to monitor')  
  args.add_argument('--port', default=8000, help='Specify the port on which Flask app runs')

  return args.parse_args()   



def main():
  global data

  args = get_args()
  result = deploy_workers(args.config)
  data.update(result)

  signal.signal(signal.SIGINT, signal_handler)
  signal.signal(signal.SIGTSTP, signal_handler)

  app.run(host="0.0.0.0", debug=True, port=args.port, use_reloader=False)


if __name__ == '__main__':
    main()