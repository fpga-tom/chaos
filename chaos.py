import numpy as np
from scipy.io.wavfile import write
from scipy import signal
from scipy.optimize import minimize
import binascii


def hd(a,b):
    return sum([1 for _a,_b in zip(a,b) if _a != _b])

message = b"Hello, world!"
message_hex = binascii.hexlify(message).decode()
message_bin = list(format(int(message_hex,16), '0>128b'))

def fitness(guess):
    np.random.seed(1)
    LEs = guess[0:len(guess)/2]
    LEs1 = guess[len(guess)/2:len(guess)]
    print(LEs)
    print(LEs1)
    def gen_a(LEs, f):
        len_LEs=len(LEs)
        q = f*np.eye(len_LEs) + np.ones([len_LEs, len_LEs])
        qi = np.linalg.inv(q)
        la = np.eye(len_LEs) * np.exp(LEs)
        A=np.matmul(np.matmul(q,la),qi)
        ev = np.linalg.eigvals(A)
        return A, (ev)

    A, aev = gen_a(LEs, 8)
    B, bev = gen_a(LEs1, 8)
    x0 = np.random.rand(len(LEs))
    x1 = np.random.rand(len(LEs))
    x3 = np.random.rand(len(LEs))


    fs = 32000*6
    seconds = 4
    time = range(0,fs*seconds)
    da = []
    bit_tx = []
    bit_rx = []
    oversample=fs/32
    for t in range(0, 32*seconds):
        r = 0 if message_bin[t] == '0' else 1
        bit_tx.append(r)
        for i in range(0,oversample):
            da.append(r)

    data_dem = []
    data_x = []

    data = []
    x_prev = 0
    x_prev1 = 0
    filt = [1./256 for i in range(0, 256)]
    filt_data = []
    N=1024
    with open('x.txt', 'w') as f:
        with open('x3.txt', 'w') as f3:
            with open('d.txt', 'w') as f4:
                for i in time:
                    if da[i] == 1:
                        x1 = np.matmul(A, x1) % 1
                        x_out = x1
                    else:
                        x0 = np.matmul(B, x0) % 1
                        x_out = x0
                    idx = 0
                    x_in = x_out[idx]
                    x_in += np.random.normal(0,1,1)
                    data.append(x_in)
                    x3[idx] = x_in
                    x3 = np.matmul(A, x3) % 1
                    f.write("%d %f\n" % (i, x_out[idx]))
                    f3.write("%d %f\n" % (i+1, x3[idx]))
                    filt_data.append(abs(-x_prev + x_in))
                    x_prev = x3[idx]
                fd = signal.lfilter(filt, [1.], filt_data,0)
                m = sum(fd)/len(fd)
                print('mean: %f' % m)
                for b in range(0,len(fd),oversample):
                    v = sum(fd[b:b+oversample])/oversample
                    bit = 1 if v < m else 0
                    bit_rx.append(bit)


                rx_string = "".join(chr(int("".join(map(str,bit_rx[i:i+8])),2)) for i in range(0,len(bit_rx),8))
                print(rx_string)
                for i,fd_filt in enumerate(fd):
                    f4.write("%d %f\n" % (i, fd_filt))

    scaled = np.int16(data/np.max(np.abs(data)) * 32767)
    write('test.wav', 32000, scaled)
    
    result = hd(bit_rx, bit_tx)
    print(result)
    return result


sp = np.random.rand(10)-.5
sp = [0.44219481, -0.36352587, -0.1619554 ,  0.34725717, -0.14898118, -0.86185026, -0.40825187, -0.60434407, -0.38570763, -0.73558969 ]
print(fitness(sp))
#result = minimize(fitness, sp, method='Powell',options={'maxiter': 20000})
#print(result.x)
#print(result.message)
