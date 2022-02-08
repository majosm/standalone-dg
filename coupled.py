import numpy as np
import matplotlib.pyplot as plt

# Some different configurations to try

# If True, both sides use air properties, otherwise the lower half uses graphite
both_sides_fluid = False

# If True, both sides use energy as the state variable; otherwise the lower half
# uses temperature. Only works if both_sides_fluid is True
both_sides_solve_for_energy = False

# If true, sets initial temperature to 300 everywhere; otherwise varies linearly
# from 250 to 350 left to right
constant_initial_temp = True

# If initial temp is constant, gradient might end up being exactly 0;
# this adds a little noise to simulate the extra floating point ops in mirgecom
add_noise = True

assert not both_sides_solve_for_energy or both_sides_fluid, (
    "Can only set both_sides_solve_for_energy to True if both_sides_fluid "
    "is also True")

# Set up mesh

npts = 21

assert npts % 2 == 1, "Must have an odd number of points"

nnodes = 2*npts-2
nels = npts - 1
xpts = np.linspace(-0.001, 0.001, npts)
node_to_point = (np.indices((2*npts,))[0] // 2)[1:-1]
node_to_elem = np.indices((2*npts-2,))[0] // 2
print(f"{node_to_point=}")
print(f"{node_to_elem=}")
x = xpts[node_to_point]
dx = (xpts[-1] - xpts[0])/nels
print(f"{x=}")
print(f"{dx=}")

# Set up operators

Mblock = dx * np.array([
    [1/3, 1/6],
    [1/6, 1/3]])
Mblock_inv = np.linalg.inv(Mblock)
Minv = np.kron(np.eye(nels), Mblock_inv)
print(f"{Minv=}")

Kblock = np.array([
    [ 1/2,  1/2],
    [-1/2, -1/2]])
K = np.kron(np.eye(nels), Kblock)
print(f"{K=}")

Fblock = np.array([
    [-1/2, -1/2],
    [ 1/2,  1/2]])
F = np.kron(np.eye(npts), Fblock)[1:-1, 1:-1]
print(f"{F=}")

Dvol = Minv @ K
Dface = Minv @ -F

print(f"{Dvol=}")
print(f"{Dface=}")

D = np.empty((nnodes, nnodes+2))
D[:, 1:-1] = Dvol + Dface
D[:, 0] = Dface[:, 0]
D[:, -1] = Dface[:, -1]

print(f"{D=}")

def with_ext(f, *, ext=None):
    f_with_ext = np.empty(nnodes+2)
    f_with_ext[1:-1] = f
    if ext is not None:
        f_with_ext[0] = ext[0]
        f_with_ext[-1] = ext[1]
    else:
        f_with_ext[0] = f[0]
        f_with_ext[-1] = f[-1]
    return f_with_ext

def gradient_operator(f, *, bdry, add_noise=False):
    f_with_ext = with_ext(f, ext=[2*bdry[0]-f[0], 2*bdry[1]-f[-1]])
    grad_f = np.dot(D, f_with_ext)
    if add_noise:
        grad_f = grad_f + 1e-8*np.random.rand(nnodes)
    return grad_f

def diffusion_operator(kappa, f, *, bdry, add_noise=False):
    grad_f = gradient_operator(f, bdry=bdry, add_noise=add_noise)
    kappa_with_ext = with_ext(kappa)
    grad_f_with_ext = with_ext(grad_f)
    diff_f = np.dot(D, kappa_with_ext * grad_f_with_ext)
    if add_noise:
        diff_f = diff_f + 1e-8*np.random.rand(nnodes)
    return diff_f

# Define volumes

elem_centers = 0.5*(xpts[:-1] + xpts[1:])
lower_half_mask = elem_centers[node_to_elem] < 0
upper_half_mask = elem_centers[node_to_elem] > 0

# Define materials

rho_air = 1.1766
c_air = 1004.69
kappa_air = 0.02565

rho_graphite = 1625
c_graphite = 770
kappa_graphite = 247.5

rho_upper = rho_air
c_upper = c_air
kappa_upper = kappa_air

if both_sides_fluid:
    rho_lower = rho_air
    c_lower = c_air
    kappa_lower = kappa_air
else:
    rho_lower = rho_graphite
    c_lower = c_graphite
    kappa_lower = kappa_graphite

rho = rho_lower * lower_half_mask + rho_upper * upper_half_mask
c = c_lower * lower_half_mask + c_upper * upper_half_mask
kappa = kappa_lower * lower_half_mask + kappa_upper * upper_half_mask

if both_sides_solve_for_energy:
    beta = 0*x + 1
else:
    beta = 1/(rho_lower * c_lower) * lower_half_mask + upper_half_mask

# Inspect RHS operator

# Ignoring outer BCs for now (not sure how to include them properly)
A = beta * D[:, 1:-1] @ (kappa * D[:, 1:-1])
print(f"{A=}")

fig, ax = plt.subplots()
ax.matshow(np.abs(A), cmap=plt.cm.Blues)
plt.show()

eigenvalues = np.linalg.eig(A)[0]
print(f"{eigenvalues=}")

# Initialize state

gamma = 1.4
gas_const = 8.314462/0.0289647

if both_sides_solve_for_energy:
    # Assume fluid on both sides
    state_to_temperature_factor = (gamma-1)/(rho_upper*gas_const) * (0*x+1)
else:
    state_to_temperature_factor = (
        lower_half_mask + (gamma-1)/(rho_upper*gas_const) * upper_half_mask)

def state_to_temperature(u):
    return state_to_temperature_factor * u

def temperature_to_state(T):
    return T / state_to_temperature_factor

if constant_initial_temp:
    T0 = 300*(0*x+1)
else:
    T0 = 250*(x[-1]-x)/(x[-1]-x[0]) + 350*(x-x[0])/(x[-1]-x[0])

u0 = temperature_to_state(T0)

# Run simulation

def rk4_step(state, t, dt, rhs):
    """Take one step using the fourth-order Classical Runge-Kutta method."""
    k1 = rhs(t, state)
    k2 = rhs(t+dt/2, state + dt/2*k1)
    k3 = rhs(t+dt/2, state + dt/2*k2)
    k4 = rhs(t+dt, state + dt*k3)
    return state + dt/6*(k1 + 2*k2 + 2*k3 + k4)

plt.ion()
plt.show()

alpha_air = kappa_air/(rho_air * c_air)
alpha_graphite = kappa_graphite/(rho_graphite * c_graphite)

print(f"{alpha_air=}, {alpha_graphite=}")

dt = 0.1 * dx**2 / max(alpha_air, alpha_graphite)

print(f"{dt=}")

fig, (ax1, ax2) = plt.subplots(1, 2)

u = u0.copy()
T = T0.copy()
t = 0
for i in range(100000):
    def get_rhs(t, u):
        T = state_to_temperature(u)
        return beta * diffusion_operator(
            kappa, T, bdry=[T0[0], T0[-1]], add_noise=add_noise)

    def plot_results(u):
        T = state_to_temperature(u)
        rhs = get_rhs(t, u)
        ax1.cla()
        ax1.plot(x, T)
        ax1.set_title('T')
        ax2.cla()
        ax2.plot(x, rhs)
        ax2.set_title('rhs')
        plt.draw()
        plt.pause(0.001)

    if i % 100 == 0:
        print(f"Step {i}")
        plot_results(u)

    u = rk4_step(u, t, dt, get_rhs)
    t = t + dt

    if not all(np.isfinite(u)):
        print(f"Solution failed health check; exiting.")
        plot_results(u)
        break

plt.pause(1000)
