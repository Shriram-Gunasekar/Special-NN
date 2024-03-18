import numpy as np
import matplotlib.pyplot as plt

def izhikevich_neuron(a, b, c, d, I, v_init, u_init, num_steps, dt):
    v = np.zeros(num_steps)
    u = np.zeros(num_steps)
    spikes = []

    v[0] = v_init
    u[0] = u_init

    for t in range(1, num_steps):
        dv_dt = 0.04 * v[t - 1]**2 + 5 * v[t - 1] + 140 - u[t - 1] + I
        du_dt = a * (b * v[t - 1] - u[t - 1])

        v[t] = v[t - 1] + dt * dv_dt
        u[t] = u[t - 1] + dt * du_dt

        if v[t] >= 30:  # Spike condition
            v[t] = c
            u[t] += d
            spikes.append(t * dt)

    return v, spikes

# Parameters
a = 0.02
b = 0.2
c = -65
d = 8
I = 10  # Input current
v_init = -70  # Initial membrane potential
u_init = b * v_init  # Initial recovery variable u
num_steps = 10000
dt = 0.1  # Time step (ms)

# Simulate neuron
v, spikes = izhikevich_neuron(a, b, c, d, I, v_init, u_init, num_steps, dt)

# Plotting
time = np.arange(0, num_steps * dt, dt)
plt.figure(figsize=(10, 5))
plt.plot(time, v, label='Membrane Potential (v)')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.title('Izhikevich Neuron Model')
plt.legend()
plt.grid(True)
plt.show()

# Display spikes
print('Spikes:', spikes)
