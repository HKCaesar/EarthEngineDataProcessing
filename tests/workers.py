import multiprocessing 
import re

def mp_worker(filename):
    with open(filename) as f:
        text = f.read()
    m = re.findall("x+", text)
    count = len(max(m, key=len))
    return filename, count

def mp_handler():
    p = multiprocessing.Pool(32)
    with open('infilenamess.txt') as f:
        filenames = [line for line in (l.strip() for l in f) if line]
    with open('results.txt', 'w') as f:
        for result in p.imap(mp_worker, filenames):
            # (filename, count) tuples from worker
            f.write('%s: %d\n' % result)

if __name__=='__main__':
    mp_handler()