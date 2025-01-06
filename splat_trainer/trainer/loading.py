from queue import Queue
from threading import Thread
import traceback


class ThreadedLoader:
  def __init__(self, iter, queue_size=4):
    self.queue = Queue()
    self.control_queue = Queue()
    self.iter = iter
    self.thread = Thread(target=self._loader_thread)
    self.thread.start()

    # Prime the control queue to fill the buffer
    for _ in range(queue_size):
      self.control_queue.put(True)

  def _loader_thread(self):
    try:
      while self.control_queue.get():  # Continue while True, exit on False
        data = next(self.iter)
        self.queue.put(data)
    except Exception as e:
      print(f"Error in loader thread: {e}")
      traceback.print_exc()

  def next(self):
    item = self.queue.get()
    self.control_queue.put(True)  # Keep going
    return item

  def stop(self):
    self.control_queue.put(False)  # Signal to stop
    self.thread.join()
