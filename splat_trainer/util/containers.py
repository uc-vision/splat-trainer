

def transpose_rows(rows):
    return {key: [row[key] for row in rows] for key in rows[0]}


def replace_dict(d, **kwargs):
  return {**d, **kwargs}