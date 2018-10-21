import numpy as np

delta = 0.5
h = 0.004
gamma = 0.815

w0 = (2*np.pi*20)

g_sig = 0

def dxvdt(t, xv):
    x1 = xv[0]
    x2 = xv[1]
    dx1_dt = w0 * x2
    dx2_dt = w0 * (-delta * x2 + x1 - x1**3 + gamma * np.cos(w0 * t) + g_sig)
    return [dx1_dt, dx2_dt]


time = range(0, 4000)

x1 = np.random.rand(1)
x2 = np.random.rand(1)

xv =  [x1, x2]


dt = h
t_max = 5.
t = 0
amp = 0.09
sigma = 0.26
with open('d.txt', 'w') as f:
    while t < t_max:
        r = np.random.normal(0,sigma,1)
        g_sig = amp*np.cos(2*np.pi*20*t)
        g_sig += r
        # 4th order Runge-Kutta step
        k1 = [ dt * d for d in dxvdt(t, xv) ]
        xv_step = [ xv[i] + k1[i] / 2 for i in range(2) ]
        g_sig = amp*np.cos(2*np.pi*20*(t + .5 * dt))
        g_sig += r
        k2 = [ dt *  d for d in dxvdt(t + 0.5 * dt, xv_step) ]
        xv_step = [ xv[i] + k2[i] / 2 for i in range(2) ]
        k3 = [ dt * d for d in dxvdt(t + 0.5 * dt, xv_step) ]
        xv_step = [ xv[i] + k3[i] for i in range(2) ]
        g_sig = amp*np.cos(2*np.pi*20*(t + dt))
        g_sig += r
        k4 = [ dt * d for d in dxvdt(t + dt, xv_step) ]
        for i in range(2):
            xv[i] += (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6.0
        t += dt
        f.write("%f %f\n" % (xv[0], xv[1]))
