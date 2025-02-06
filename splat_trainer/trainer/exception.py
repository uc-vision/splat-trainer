class TrainingException(Exception):
  """
  Base class for all training exceptions where the training should be aborted for a given reason
  """
  pass


class NaNParameterException(TrainingException):
  """
  Exception raised when non-finite parameters are detected in the training process
  """
  pass


class NoProgressException(TrainingException):
  """
  Exception raised when the training aborts early due to no progress
  """
  pass


class TrainingTimeoutException(TrainingException):
  """
  Exception raised when the training should be aborted due to a timeout (epoch not completed within a given time)
  """
  pass


