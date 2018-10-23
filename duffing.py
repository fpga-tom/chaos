import numpy as np
from scipy.io.wavfile import write
import scipy.integrate as integrate
import binascii

delta = 0.5
h = 0.000001
gamma = 0.825

freq = 31277
w0 = freq*2*np.pi
print(w0)

message = b"Hello, world!"
message_hex = binascii.hexlify(message).decode()
message_bin = list(format(int(message_hex,16), '0>128b'))

#while w0 > 2*np.pi:
#    w0 -= 2*np.pi

g_sig = 0

def dxvdt(t, xv):
    x1 = xv[0]
    x2 = xv[1]
    dx1_dt = w0 * x2
    dx2_dt = w0 * (-delta * x2 + x1 - x1**3 + gamma * np.cos(w0 * t) + g_sig)
    return [dx1_dt, dx2_dt]


time = range(0, 4000)

x1 = np.random.normal(1)
x2 = np.random.normal(1)

xv =  [x1, x2]

xv = [0,-1]

print(xv)


dt = h
bit_duration = 0.01
t_max = bit_duration * len(message_bin)
t = 0
amp = 0.04
sigma = 0.26
data = []
auto = []
data_bit = message_bin[0]
data_bit_idx=0
data_bits = []
t_next = bit_duration
with open('d.txt', 'w') as f:
    while t < t_max:
        if t >= t_next:
            t_next += bit_duration
            data_bits.append(auto)
            auto = []
            data_bit_idx += 1
            data_bit = message_bin[data_bit_idx]
            xv = [0,-1]
        r = np.random.normal(0,sigma,1)[0]
        g_sig = amp*np.cos(w0*t) if data_bit == '1' else 0
        g_sig += r
        # 4th order Runge-Kutta step
        k1 = [ dt * d for d in dxvdt(t, xv) ]
        xv_step = [ xv[i] + k1[i] / 2 for i in range(2) ]
        k2 = [ dt *  d for d in dxvdt(t + 0.5 * dt, xv_step) ]
        xv_step = [ xv[i] + k2[i] / 2 for i in range(2) ]
        k3 = [ dt * d for d in dxvdt(t + 0.5 * dt, xv_step) ]
        xv_step = [ xv[i] + k3[i] for i in range(2) ]
        k4 = [ dt * d for d in dxvdt(t + dt, xv_step) ]
        for i in range(2):
            xv[i] += (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6.0
        t += dt
        f.write("%.9f %.9f\n" % (xv[0],xv[1]))
        data.append(g_sig)
        auto.append(xv[0] + xv[1])

scaled = np.int16(data/np.max(np.abs(data)) * 32767)
write('test.wav', 32000, scaled)

bit_rx = []
mags = []
for auto in data_bits:
    q0 = 0
    q1 = 0
    q2 = 0
    N=len(auto)
    k = int(0.5 + N*freq*h)
    w = (2*np.pi/N)*k
    cosine = np.cos(w)
    sine = np.sin(w)
    coeff = 2 * cosine
    for s in auto:
        q0 = coeff*q1 - q2 + s
        q2 = q1
        q1 = q0

    real = q1 - q2 * cosine
    imag = q2*sine
    magnitude = np.sqrt(real**2 + imag**2)
    mags.append(magnitude)

max_val = max(mags)
min_val = min(mags)

mean_val = (max_val + min_val)/2.
print(mean_val)

for magnitude in mags:
    bit_rx.append(1 if magnitude > mean_val else 0)

rx_string = "".join(chr(int("".join(map(str,bit_rx[i:i+8])),2)) for i in range(0,len(bit_rx),8))
print(rx_string)

def hd(a,b):
    return sum([1 for _a,_b in zip(a,b) if _a != _b])

print(hd(bit_rx, [0 if x == '0' else 1 for x in  message_bin]))
#with open('m.txt','w') as f:
#    while t < t_max:
#        m = m_f(1.7, data, t, t_max)
#        f.write("%f %f\n" % (t, m))
#        t += dt

