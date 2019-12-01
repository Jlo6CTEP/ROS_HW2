from matplotlib import pyplot as plt
from numpy import linspace, vstack, sqrt, arctan2, array, cos as cos_n, \
    pi as pi_n
from sympy import symbols, sin, simplify, cos, lambdify, pi, \
    Matrix
from sympy.physics.mechanics import dynamicsymbols, vlatex

g = 9.81

m1, m2, j1, j2, s, l = symbols('m1, m2, j1, j2, s, l')
nums = [2, 2, 1, 2, 0.2, 0.2]
m1_n, m2_n, j1_n, j2_n, s_n, l_n = nums

fi, q = dynamicsymbols(r'\phi, q')
fi_d, q_d = dynamicsymbols(r'\phi, q', 1)
fi_dd, q_dd = dynamicsymbols(r'\phi, q', 2)

def fk(phi, q):
    return (s_n + q + l_n) * cos(phi), (s_n + q + l_n) * sin(phi)


def rk(x, y):
    return arctan2(y, x), sqrt(x ** 2 + y ** 2) - s_n - l_n


time = linspace(0, 1, 1000)

phi, phi_0, phi_tay, tay, t = symbols(r'\phi \phi(0) \phi(\tau) \tau t')
init = [0, pi / 3, 1]

phi_t = -(phi_tay - phi_0) * (10 - 15 * t / tay + 6 * (t / tay) ** 2) * (
            t ** 3 / tay ** 3)
phi_t_d = phi_t.diff(t, 1)
phi_t_dd = phi_t.diff(t, 2)

phi_t_f = lambdify([t, phi_tay, phi_0, tay], phi_t, "numpy")
phi_t_d_f = lambdify([t, phi_tay, phi_0, tay], phi_t_d, "numpy")
phi_t_dd_f = lambdify([t, phi_tay, phi_0, tay], phi_t_dd, "numpy")

q_t = 3 * l / cos(fi) - (s + l)
q_t_d = q_t.diff(t, 1)
q_t_dd = q_t.diff(t, 2)

q_t_f = lambdify([fi, l, s], q_t)
q_t_d_f = lambdify([fi_d, fi, l, s], q_t_d)
q_t_dd_f = lambdify([fi_dd, fi_d, fi, l, s], q_t_dd)

dyn_vars = []

for t_n in time:
    f, f_d, f_dd = float(phi_t_f(t_n, *init)), \
                   float(phi_t_d_f(t_n, *init)), \
                   float(phi_t_dd_f(t_n, *init))
    o, o_d, o_dd = float(q_t_f(f, l_n, s_n)), \
                   float(q_t_d_f(f_d, f, l_n, s_n)), \
                   float(q_t_dd_f(f_dd, f_d, f, l_n, s_n))

    dyn_vars.append(array([f_dd, f_d, f, o_dd, o_d, o]))

dyn_vars = vstack(dyn_vars)
dyn_vars_t = dyn_vars.T

plt.plot(time, dyn_vars_t[3], label='accel of q')
plt.plot(time, dyn_vars_t[0], label='accel of phi')
plt.title('Acceleration vs time')
plt.xlabel('Time, s')
plt.ylabel('Acceleration')
plt.legend()
plt.show()

plt.plot(time, dyn_vars_t[5], label='coord of q')
plt.plot(time, dyn_vars_t[2], label='coord of phi')
plt.title('Coordinate vs time')
plt.ylabel('Coordinate')
plt.xlabel('Time, s')
plt.legend()
plt.show()

plt.plot(time, dyn_vars_t[4], label='speed of q')
plt.plot(time, dyn_vars_t[1], label='speed of phi')
plt.title('Speed vs time')
plt.ylabel('Speed')
plt.xlabel('Time, s')
plt.legend()
plt.show()


Ek_1l = 0
Ek_1a = j1 * fi_d ** 2 / 2

Ek_2l = m2 * q_d ** 2 / 2
Ek_2a = (j2 + m2 * (s + q) ** 2) * fi_d ** 2 / 2

Ep_1 = m1 * g * s * sin(fi) / 2
Ep_2 = m2 * g * (s + q) * sin(fi)

Ek = Ek_1l + Ek_1a + Ek_2l + Ek_2a
Ep = Ep_1 + Ep_2

L = Ek - Ep

l_diff = Matrix([[(L.diff(fi_d)).diff(t) - L.diff(fi)],
                 [(L.diff(q_d)).diff(t) - L.diff(q)]])

l_f = lambdify([fi_dd, fi_d, fi, q_dd, q_d, q, m1, m2, j1, j2, s, l],
               l_diff, "numpy")

torques = []
for x in dyn_vars:
    torques.append(l_f(*x, *nums).flatten())

torques = vstack(torques)

plt.plot(time, torques[:, 0], label='phi')
plt.plot(time, torques[:, 1], label='q')
plt.title('Generalized forces on joints')
plt.xlabel('Time, s')
plt.legend()
plt.show()

# Fi Q

dx = 1 / 1000

phi_cur = 0
phi_cur_d = 0
phi_cur_dd = 0

q_cur = 0.2
q_cur_d = 0
q_cur_dd = 0

phi_s = []
phi_s_d = []
phi_s_dd = []

q_s = []
q_s_d = []
q_s_dd = []

coord = []

count = 0

for x in torques:
    phi_cur_dd = (x[0] - m1_n * g * s_n * cos_n(phi_cur) / 2
                  - m2_n * g * (s_n + q_cur) * cos_n(phi_cur)
                  - 2*m2_n*q_cur_d*phi_cur_d*(s_n+q_cur))\
                 /(j1_n + m2_n * (s_n + q_cur) ** 2 + j2_n)
    q_cur_dd = (x[1] - m2_n * g * cos_n(pi_n/2-phi_cur) +
                m2_n * (phi_cur_d * (s_n+q_cur))**2/(s_n+q_cur))/m2_n

    q_cur += (q_cur_d * dx + q_cur_dd * dx ** 2 / 2)
    phi_cur += (phi_cur_d * dx + phi_cur_dd * dx ** 2 / 2)

    q_cur_d += q_cur_dd * dx
    phi_cur_d += phi_cur_dd * dx

    phi_s_dd.append(phi_cur_dd)
    q_s_dd.append(q_cur_dd)
    phi_s_d.append(phi_cur_d)
    q_s_d.append(q_cur_d)
    phi_s.append(phi_cur)
    q_s.append(q_cur)

    coord.append(fk(phi_cur, q_cur))
    count+=1

coord = vstack(coord).T

plt.plot(*coord, label='trajectory')
plt.title('Robot trajectory')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.show()

plt.plot(time, q_s_dd, label='accel of q')
plt.plot(time, phi_s_dd, label='accel of phi')
plt.title('Acceleration vs time')
plt.xlabel('Time, s')
plt.ylabel('Acceleration')
plt.legend()
plt.show()

plt.plot(time, q_s, label='coord of q')
plt.plot(time, phi_s, label='coord of phi')
plt.title('Coordinate vs time')
plt.ylabel('Coordinate')
plt.xlabel('Time, s')
plt.legend()
plt.show()

plt.plot(time, q_s_d, label='speed of q')
plt.plot(time, phi_s_d, label='speed of phi')
plt.title('Speed vs time')
plt.ylabel('Speed')
plt.xlabel('Time, s')
plt.legend()
plt.show()
