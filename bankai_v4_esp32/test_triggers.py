import serial
import time
import sys

try:
    s = serial.Serial('COM9', 115200, timeout=0.1)
    s.dtr = False
    s.rts = True
    time.sleep(0.1)
    s.rts = False
    s.dtr = True
    time.sleep(1.0)

    print("=== READING BOOT OUTPUT & BENCHMARK ===", flush=True)
    start = time.time()
    while time.time() - start < 3.0:
        data = s.read(1024)
        if data:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()

    print("\n\n>>> INJECTING KEYBOARD TRIGGER: 'b' (Simulating Spoken BANKAI)...", flush=True)
    s.write(b'b')
    start = time.time()
    while time.time() - start < 1.5:
        data = s.read(1024)
        if data:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()

    print("\n\n>>> INJECTING KEYBOARD TRIGGER: 'u' (Simulating Spoken UNKNOWN Speech)...", flush=True)
    s.write(b'u')
    start = time.time()
    while time.time() - start < 1.5:
        data = s.read(1024)
        if data:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()

    s.close()
    print("\n\n=== VERIFICATION SUITE COMPLETE ===", flush=True)
except Exception as e:
    print("Error:", e)
