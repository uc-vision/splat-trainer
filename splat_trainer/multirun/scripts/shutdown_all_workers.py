import socket
from splat_trainer.multirun.deploy import shutdown_all_workers



def main():
    hostname = socket.gethostname()
    redis_url = f'redis://{hostname}:6379/0'
    shutdown_all_workers(redis_url)

    print(f"All RQ Workers connecting to {redis_url} are shutdown.")


if __name__ == "__main__":
    main()