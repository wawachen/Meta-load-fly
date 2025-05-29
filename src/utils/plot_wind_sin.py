import numpy as np
import matplotlib.pyplot as plt

kx = 2 * np.pi / 50

x = np.linspace(0, 300, 300)

plt.figure(figsize=(10, 8))

plt.subplot(3,1,1)
y = 0.2*np.sin(kx * x)
plt.plot(x, y, color='blue')
plt.xlabel('Timestep t',fontsize=13)
plt.ylabel('Wind force [N]', fontsize=13)
plt.title('Sinusoidal Wind Field: amplitude = 0.2', fontsize=18)
plt.ylim([-1, 1])
plt.grid()


plt.subplot(3,1,2)
y = 0.5*np.sin(kx * x)
plt.plot(x, y, color='blue')
plt.title('Sinusoidal Wind Field: amplitude = 0.5', fontsize=18)
plt.xlabel('Timestep t',fontsize=13)
plt.ylabel('Wind force [N]',fontsize=13)
plt.ylim([-1, 1])
plt.grid()


plt.subplot(3,1,3)
y = 0.8*np.sin(kx * x)
plt.plot(x, y, color='blue')
plt.title('Sinusoidal Wind Field: amplitude = 0.8', fontsize=18)

plt.xlabel('Timestep t',fontsize=13)
plt.ylabel('Wind force [N]',fontsize=13)
plt.xlim([0, 300])
plt.ylim([-1, 1])
plt.tight_layout()
plt.grid()
plt.show()