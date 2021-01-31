import numpy as np
from KS import KS
import matplotlib.pyplot as plt
import matplotlib.animation as animation

L   = 16           # domain is 0 to 2.*np.pi*L
N   = 128          # number of collocation points
dt  = 0.5          # time step
diffusion = 1.0
ks = KS(L=L,diffusion=diffusion,N=N,dt=dt) # instantiate model

# define initial condition
#u = np.cos(x/L)*(1.0+np.sin(x/L)) # smooth IC
u = 0.01*np.random.normal(size=N) # noisy IC
# remove zonal mean
u = u - u.mean()
# spectral space variable.
ks.xspec[0] = np.fft.rfft(u)

# time stepping loop.
nmin = 1000; nmax = 20000#10001
uu = []; tt = []
vspec = np.zeros(ks.xspec.shape[1], np.float)
x = np.arange(N)
fig, ax = plt.subplots()
line, = ax.plot(x, ks.x.squeeze())
ax.set_xlim(0,N-1)
ax.set_ylim(-3,3)
#Init only required for blitting to give a clean slate.
def init():
    global line
    line.set_ydata(np.ma.array(x, mask=True))
    return line,

# spinup
for n in range(nmin):
    ks.advance()

def updatefig(n):
    global tt,uu,vspec
    ks.advance()
    vspec += np.abs(ks.xspec.squeeze())**2
    u = ks.x.squeeze()
    line.set_ydata(u)
    print(n,u.min(),u.max())
    uu.append(u); tt.append(n*dt)
    return line,

ani = animation.FuncAnimation(fig, updatefig, np.arange(1,nmax+1), init_func=init,
                              interval=25, blit=True, repeat=False)
plt.show()

plt.figure()
# make contour plot of solution, plot spectrum.
ncount = len(uu)
vspec = vspec/ncount
uu = np.array(uu); tt = np.array(tt)
print(tt.min(), tt.max())
nplt = 200
plt.contourf(x,tt[:nplt],uu[:nplt],31,extend='both')
plt.xlabel('x')
plt.ylabel('t')
plt.colorbar()
#plt.title('chaotic solution of the K-S equation')
plt.savefig("K-S_exact.png")
plt.savefig("K-S_exact.eps")

"""plt.figure()
plt.loglog(ks.wavenums,vspec)
plt.title('variance spectrum')
plt.ylim(0.001,10000)
plt.xlim(0,100)
"""
plt.show()


import warnings

warnings.filterwarnings('ignore')

import csv
import numpy as np

numICs = 1#0000

x1range = [-2, 2]
x2range = [-2, 2]
x3range = [-2, 2]
#tSpan = np.arange(0, 2.5 + 0.1, 0.25)# np.arange(0, 125, 0.25)  # 0:0.02:1
tSpan = np.arange(0, 160 + 0.0001, 0.001)



def make_csv(filename, X):
    with open(filename, 'w') as csv_file:
        fieldnames = []
        for i in range(128):
            fieldnames.append('prediction_x' + "{stp:02}".format(stp=i))
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        dic = {}
        for i in range(len(X)):
            for j in range(128):
                dic[fieldnames[j]] = X[i][j]
            writer.writerow(dic)




z = uu#[:nplt]
z = np.array(z)

X_train = z
###############Discrete_Linear#############################
filenamePrefix = 'K_S_FFT'
"""
#tSpan = np.arange(0, 12.5, 0.25)  # 0, 12.5, 0.25
tSpan = np.arange(0, 2100 - 0.0001, 0.001)  # 160
seed = 10
#X_train = Kuramoto_Sivashinsky_ODE(x1range, x2range, x3range, numICs, tSpan, seed, "")
#tSpan = np.arange(0, 2.5 - 0.1, 0.25)
X_train = z[:-1]
filename_train = filenamePrefix + '_train_x.csv'
make_csv(filename_train, X_train)

#seed = 10
tSpan = np.arange(0, 2100 + 0.0001, 0.001)
#tSpan = np.arange(0, 2.5 + 0.1, 0.25) # 0, 2.5 + 0.1, 0.25
filename_train = filenamePrefix + '_train_y.csv'
X_train = z[1:]
make_csv(filename_train, X_train)

#seed = 1#1
#tSpan = np.arange(0, 12.5, 0.25)
tSpan = np.arange(0, 2100 + 0.0001, 0.001)
filename_train = filenamePrefix + "_E_recon_50" + '.csv'
X_train = z[:-1]
make_csv(filename_train, X_train)
"""
#seed = 3
#tSpan = np.arange(0, 0.25 + 0.1, 0.25)  # 0, 12.5, 0.25
tSpan = np.arange(0, 100 + 0.001, 0.001)
filename_train = filenamePrefix + "_E_eigfunc" + '.csv'
X_train = z#[:-1]
make_csv(filename_train, X_train)