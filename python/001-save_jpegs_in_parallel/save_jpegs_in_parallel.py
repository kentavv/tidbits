#!/usr/bin/env python3

# This tests methods of saving 12-bit cv2 images files, which are represented as numpy arrays,
# in parallel to 8-bit jpeg files. Pickling and shared memory methods are used. Various 
# numbers of and sizes of images are tested. Using random images is not reflective of 
# compression of normal images but is not expected to invalidate the comparison. For the 
# number of and sizes of images, the best speedup was ~4x on an eight-core Ryzen 7 1800X 
# machine. This tidbit is also an example of shared memory.

import ctypes
import multiprocessing as mp
import hashlib
import sys
import time
import timeit
from datetime import datetime
from typing import Tuple, List, Union, NewType, Iterable

import cv2
import numpy as np

Image = Union[np.ndarray, Iterable, int, float]
Filename = NewType('Filename', str)


arrays = []
method = 0


def file_checksum(fn: Filename) -> str:
    return hashlib.sha256(open(fn, 'rb').read()).hexdigest()


def init_shared_memory(share):
    global arrays
    arrays = share


def write_jpeg_parallel(ind: int) -> Tuple[bool, str, Filename]:
    """
    Given an image, convert the image to 8 bit and save as a jpeg.
    (All errors generate warnings but the final return value is always true.)
    """

    rv = True
    msg = ''

    args = arrays[ind]
    fn, m, s = args

    m_np = None
    if method == 0:
        print('For method 0, use map(write_jpeg_serial)')
        sys.exit(1)
    elif method == 1:
        # method 1. pass the image as-is and convert to eight-bit in the child process
        m_np = np.frombuffer(m, dtype='I').reshape(s)
        m_np = np.divide(m_np, 2 ** 4).astype(np.uint8)
    elif method == 2:
        # method 2. like method 1, but use specific datatypes of minimal length
        m_np = np.frombuffer(m, dtype=np.uint16).reshape(s)
        m_np = np.divide(m_np, 2 ** 4).astype(np.uint8)
    elif method == 3:
        # method 3. convert the image to eight-bit in the parent process
        m_np = np.frombuffer(m, dtype=np.uint8).reshape(s)

    options = [cv2.IMWRITE_JPEG_QUALITY, 95, cv2.IMWRITE_JPEG_OPTIMIZE, 1]

    if not cv2.imwrite(fn, m_np, options):
        rv = False
        msg = f'Unable to write jpeg file (ignoring): {fn}'

    return rv, msg, fn


def write_jpegs_parallel(imgs: List[Tuple[Filename, Image]]) -> Tuple[bool, str]:
    """
    Given a list of images, which are tuples of filename string and image data, convert the image to 8 bit and save as
    a jpeg. (All errors generate warnings but the final return value is always true.)
    """

    rv = True
    msg = ''

    if method == 0:
        # method 0. use simple multiprocessing module maps which require pickling data for IPC transport
        with mp.Pool() as pool:
            for res in pool.imap(write_jpeg_serial, imgs):
                rv, msg, fn = res
                if not rv:
                    print(msg)
                    #break
    else:
        arrays = []
        for img in imgs:
            fn, img = img

            m_np = None
            if method == 1:
                # method 1. pass the image as-is and convert to eight-bit in the child process
                m = mp.RawArray('I', int(np.prod(img.shape))) # I = C's unsigned int
                m_np = np.frombuffer(m, dtype='I').reshape(img.shape)
            if method == 2:
                # method 2. like method 1, but use specific datatypes of minimal length
                # When specifying the datatypes, use ctypes and np types so you're sure of what you are getting.
                # Typenames like 'I' and 'B' have a minimum size but might be larger and waste space and transport time
                # I = C's unsigned int, with a minimum of two bytes, but likely four, or overkill for 16-bit images
                # more type codes at https://docs.python.org/3.7/library/array.html#module-array
                m = mp.RawArray(ctypes.c_uint16, int(np.prod(img.shape)))
                m_np = np.frombuffer(m, dtype=np.uint16).reshape(img.shape)
            elif method == 3:
                # method 3. convert the image to eight-bit in the parent process
                # No synchronization of shared memory is required, so RawArray (or Array with lock=False) should be fine
                img = np.divide(img, 2 ** 4).astype(np.uint8)
                m = mp.RawArray(ctypes.c_uint8, int(np.prod(img.shape)))
                m_np = np.frombuffer(m, dtype=np.uint8).reshape(img.shape)

            # Copy the numpy array into the numpy array that overlays the shared memory
            np.copyto(m_np, img)
            arrays += [(fn, m, img.shape)]

        # Can't pass the arrays through an mp.imap. An error will appear similar to "objects must be passed through
        # inheritance, which must be IPC inheritance not OO inheritance. So, use an init function to set a global
        # variable that the children will see.
        with mp.Pool(initializer=init_shared_memory, initargs=(arrays,)) as pool:
            for res in pool.imap(write_jpeg_parallel, list(range(len(arrays)))):
                rv, msg, fn = res
                if not rv:
                    print(msg)
                    #break

    # TODO Currently, ignoring all errors
    rv = True
    msg = ''

    return rv, msg


def write_jpeg_serial(args: Tuple[Filename, Image]) -> Tuple[bool, str, Filename]:
    """
    Given an image, convert the image to 8 bit and save as a jpeg.
    (All errors generate warnings but the final return value is always true.)
    """

    rv = True
    msg = ''

    fn, img = args

    img = np.divide(img, 2 ** 4).astype(np.uint8)
    options = [cv2.IMWRITE_JPEG_QUALITY, 95, cv2.IMWRITE_JPEG_OPTIMIZE, 1]

    if not cv2.imwrite(fn, img, options):
        rv = False
        msg = f'Unable to write jpeg file (ignoring): {fn}'

    return rv, msg, fn


def write_jpegs_serial(imgs: List[Tuple[Filename, Image]]) -> Tuple[bool, str]:
    """
    Given a list of images, which are tuples of filename string and image data, convert the image to 8 bit and save as
    a jpeg. (All errors generate warnings but the final return value is always true.)
    """

    rv = True
    msg = ''

    for img in imgs:
        rv, msg, fn = write_jpeg_serial(img)
        if not rv:
            print(msg)
            #break

    # TODO Currently, ignoring all errors
    rv = True
    msg = ''

    return rv, msg


def test(n_images, shape):
    global method

    col_width = 60
    timeit_tries = 10

    t0 = time.time()
    print(f'Creating {n_images} random images of size {shape} ...'.ljust(col_width), end='', flush=True)
    imgs = [np.random.randint(0, 2**12, size=shape, dtype=np.uint16) for _ in range(n_images)]
    print('{:.02f}s'.format(time.time() - t0))

    print('Writing images in serial ... '.ljust(col_width), end='', flush=True)
    fns = [f'test_{i:02d}_s.jpg' for i in range(n_images)]
    closure = lambda: write_jpegs_serial(zip(fns, imgs))
    st = timeit.timeit(closure, number=timeit_tries)
    print('{:.02f}s {:.02f}x'.format(st, 1.))
    cs_s = [file_checksum(fn) for fn in fns]

    for method in [0, 1, 2, 3]:
        print(f'Writing images in parallel (method {method}) ...'.ljust(col_width), end='', flush=True)
        fns = [f'test_{i:02d}_p{method}.jpg' for i in range(n_images)]
        closure = lambda: write_jpegs_parallel(zip(fns, imgs))
        pt = timeit.timeit(closure, number=timeit_tries)
        print('{:.02f}s {:.02f}x'.format(pt, st / pt))
        cs_p = [file_checksum(fn) for fn in fns]
        if cs_s != cs_p:
            print('\tWARNING: checksums don\'t match')
            for (a, b) in zip(cs_s, cs_p):
                print(f'\t\t{a} {b}')


def print_system_information():
    print('Current time:', datetime.now())
    try:
        print('Processor:', [x.split(':', 1)[1].strip() for x in open('/proc/cpuinfo').readlines() if x.startswith('model name')][0])
    except:
        print('Processor: Unable to determine')
    try:
        print('MemTotal:', [x.split(':', 1)[1].strip() for x in open('/proc/meminfo').readlines() if x.startswith('MemTotal')][0])
        print('MemAvailable:', [x.split(':', 1)[1].strip() for x in open('/proc/meminfo').readlines() if x.startswith('MemAvailable')][0])
    except:
        print('Processor: Unable to determine')
    print('Python version:', sys.version.replace('\n', '\t'))
    print('OpenCV version:', cv2.__version__)
    print('Numpy version:', np.__version__)
    print()


def main():
    print_system_information()

    for n_images in [1, 2, 4, 8, 16]:
        for shape in [(1000, 1000, 3), (2000, 2000, 3), (4000, 4000, 3), (8000, 8000, 3)]:
            test(n_images, shape)
            print()


if __name__ == "__main__":
    main()
