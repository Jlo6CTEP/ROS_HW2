import pickle

from numpy import array, loadtxt, pi, arctan, \
    arccos, eye, arctan2, sqrt
from numpy.linalg import inv
from sympy import Matrix, sin, cos, eye, symbols, pprint, simplify, \
    BlockMatrix, diff

l1_len = 346
l2_len = 324
l3_len = 312
l4_len = 1075
l5_len = 225
l6_len = 1280
l7_len = 215


def j_col(m):
    print('jcol')
    return Matrix([[*m[0:3, 3], m[2, 1], m[0, 2], m[1, 0]]]).T


def r_z(angle):
    return Matrix([[cos(angle), -sin(angle), 0, 0],
                   [sin(angle), cos(angle), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])


def r_y(angle):
    return Matrix([[cos(angle), 0, sin(angle), 0],
                   [0, 1, 0, 0],
                   [-sin(angle), 0, cos(angle), 0],
                   [0, 0, 0, 1]])


def r_x(angle):
    return Matrix([[1, 0, 0, 0],
                   [0, cos(angle), -sin(angle), 0],
                   [0, sin(angle), cos(angle), 0],
                   [0, 0, 0, 1]])


def shift_by_vector(shift):
    matrix = eye(4)
    matrix[0:3, 3] = shift
    return matrix


def t_x(shift):
    return shift_by_vector([shift, 0, 0])


def t_y(shift):
    return shift_by_vector([0, shift, 0])


def t_z(shift):
    return shift_by_vector([0, 0, shift])


q1, q2, q3, q4, q5, q6, l1, l2, l3, l4, l5, l6, l7 = \
    symbols('q1 q2 q3 q4 q5 q6 l1 l2 l3, l4, l5, l6 l7')

# j1 = Matrix([*t0[:3, 2].cross(t2[:3, 3] - t0[:3, 3]).T, *t0[:3, 2].T])
# j2 = Matrix([*t1[:3, 2].cross(t2[:3, 3] - t1[:3, 3]).T, *t1[:3, 2].T])
# jabobian = simplify(Matrix([[j1, j2]]))
# pprint(jabobian)

T = t_z(l1) * r_z(q1) * t_z(l2) * t_x(l3) * r_y(q2) * t_z(l4) * \
    r_y(q3) * t_z(l5) * t_x(l6) * r_x(q4) * r_y(q5) * r_x(q6) * t_x(l7)

T = simplify(T)

T_inv = eye(4)
T_inv[:3, :3] = T[:3, :3].T

print('inverse')

t0 = eye(4)
t1 = t0 * t_z(l1) * r_z(q1) * t_z(l2) * t_x(l3)
t2 = t1 * r_y(q2) * t_z(l4)
t3 = t2 * r_y(q3) * t_z(l5) * t_x(l6)
t4 = t3 * r_x(q4)
t5 = t4 * r_y(q5)
t6 = t5 * r_x(q6) * t_x(l7)

j_columns = \
    [Matrix([*t_i[:3, axis].cross(t6[:3, 3] - t_i[:3, 3]).T, *t_i[:3, axis].T])
     for t_i, axis in zip([t0, t1, t2, t3, t4, t5], [2, 1, 1, 0, 1, 0])]

jabobian_skew = simplify(Matrix([[*j_columns]]))

pickle.dump(jabobian_skew, open('pickled_jabobian', 'wb'))

# a = jabobian_skew.evalf(subs={
#    q1: 0.5, q2: 0.8, q3: 0.7, q4: 1.5, q5: 0.2, q6: 0.9,
#    l1: l1_len, l2: l2_len, l3: l3_len, l4:
#    l4_len, l5: l5_len, l6: l6_len, l7: l7_len})

print('t1')
T1 = \
    t_z(l1) * diff(r_z(q1), q1) * t_z(l2) * t_x(l3) * r_y(q2) * t_z(l4) * \
    r_y(q3) * t_z(l5) * t_x(l6) * r_x(q4) * r_y(q5) * r_x(q6) * t_x(l7) * T_inv


print('t2')
T2 = \
    t_z(l1) * r_z(q1) * t_z(l2) * t_x(l3) * diff(r_y(q2), q2) * t_z(l4) * \
    r_y(q3) * t_z(l5) * t_x(l6) * r_x(q4) * r_y(q5) * r_x(q6) * t_x(l7) * T_inv


print('t3')
T3 = \
    t_z(l1) * r_z(q1) * t_z(l2) * t_x(l3) * r_y(q2) * t_z(l4) * \
    diff(r_y(q3), q3) * t_z(l5) * t_x(l6) * r_x(q4) * r_y(q5) * r_x(q6) * \
    t_x(l7) * T_inv


print('t4')
T4 = \
    t_z(l1) * r_z(q1) * t_z(l2) * t_x(l3) * r_y(q2) * t_z(l4) * r_y(q3) * \
    t_z(l5) * t_x(l6) * diff(r_x(q4), q4) * r_y(q5) * r_x(q6) * t_x(l7) * T_inv


print('t5')
T5 = \
    t_z(l1) * r_z(q1) * t_z(l2) * t_x(l3) * r_y(q2) * t_z(l4) * r_y(q3) * \
    t_z(l5) * t_x(l6) * r_x(q4) * diff(r_y(q5), q5) * r_x(q6) * t_x(l7) * T_inv


print('t6')
T6 = \
    t_z(l1) * r_z(q1) * t_z(l2) * t_x(l3) * r_y(q2) * t_z(l4) * r_y(q3) * \
    t_z(l5) * t_x(l6) * r_x(q4) * r_y(q5) * diff(r_x(q6), q6) * t_x(l7) * T_inv


j_columns = [simplify(j_col(t)) for t in [T1, T2, T3, T4, T5, T6]]
jabobian_num = simplify(Matrix([[*j_columns]]))
print('jabobian_num')

pprint(jabobian_num, wrap_line=False)
pprint(simplify(jabobian_num - jabobian_skew), wrap_line=False)
