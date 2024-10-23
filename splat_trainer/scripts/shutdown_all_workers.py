import socket
from splat_trainer.util.deploy import shutdown_all_workers



def main():
    hostname = socket.gethostname()
    shutdown_all_workers(f'redis://{hostname}:6379')


if __name__ == "__main__":
    main()