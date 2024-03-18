from multiprocessing import cpu_count, get_logger
from multiprocessing.pool import ThreadPool
from pathlib import Path
from beartype.typing import List


from beartype import beartype
import cv2
from tqdm import tqdm
import traceback

# Shortcut to multiprocessing's logger
def error(msg, *args):
    return get_logger().error(msg, *args)

class LogExceptions(object):
    def __init__(self, callable):
        self.__callable = callable

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)

        except Exception as e:
            # Here we add some debugging help. If multiprocessing's
            # debugging is on, it will arrange to log the traceback
            error(traceback.format_exc())
            # Re-raise the original exception so the Pool worker can
            # clean up
            raise

        # It was fine, give a normal answer
        return result


def parmap_list(f, xs, j=cpu_count() // 4, chunksize=1, pool=ThreadPool, progress=tqdm):

  with pool(processes=j) as pool:
    iter = pool.imap(LogExceptions(f), xs, chunksize=chunksize)
    
    if progress is not None:
      iter = progress(iter, total=len(xs))

    return list(iter)


@beartype
def load_image(filename:Path, image_scale:float=1.0):
  assert filename.is_file(), f"load_image: file {filename} does not exist"

  image = cv2.imread(str(filename), cv2.IMREAD_COLOR)
  if image_scale != 1.0:
    image = cv2.resize(image, fx=image_scale, fy=image_scale, interpolation=cv2.INTER_AREA)

  assert image is not None, f"load_image: could not read {filename}"
  return image

@beartype
def load_images(filenames:List[str], base_path:Path, **map_options):
  return parmap_list(load_image, [base_path / f for f in filenames], **map_options)  
