

def transpose_rows(rows):
    return {key: [row[key] for row in rows] for key in rows[0]}