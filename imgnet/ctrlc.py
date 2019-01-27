#-- Ayan Chakrabarti <ayan@wustl.edu>

import signal

stop = False
_orig = None

def handler(a,b):
    global stop
    stop = True
    signal.signal(signal.SIGINT,_orig)

_orig = signal.signal(signal.SIGINT,handler)
