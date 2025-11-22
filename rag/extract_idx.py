import re

def find(text : str):
    ids = re.findall(r"msg_\d+", text)
    return ids