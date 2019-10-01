from numpy import array, loadtxt, pi, arctan, \
    arccos, sin, eye, arctan2, sqrt, cos
from numpy.linalg import inv

BASE = 670
BASE_ROT = 324
BASE_SHIFT = 312
ELBOW = 1075
SHIFT = 225
FOREARM = 1280
TOOl = 215


def rotate_z(angle):
    return array([[cos(angle), -sin(angle), 0, 0],
                  [sin(angle), cos(angle), 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])


def rotate_y(angle):
    return array([[cos(angle), 0, sin(angle), 0],
                  [0, 1, 0, 0],
                  [-sin(angle), 0, cos(angle), 0],
                  [0, 0, 0, 1]])


def rotate_x(angle):
    return array([[1, 0, 0, 0],
                  [0, cos(angle), -sin(angle), 0],
                  [0, sin(angle), cos(angle), 0],
                  [0, 0, 0, 1]])


def shift_by_vector(shift):
    matrix = array(eye(4))
    matrix[0:3, 3] = shift
    return matrix


def shift_x(shift):
    return shift_by_vector([shift, 0, 0])


def shift_y(shift):
    return shift_by_vector([0, shift, 0])


def shift_z(shift):
    return shift_by_vector([0, 0, shift])


# def rot2eul(R):
#   # q2 is less then 0 and its sin is less then 0
#   q1 = np.arctan2(R[1][0], -1*R[2][0])
#   sq2 = -1*np.sqrt(R[1][0]**2 + R[2][0]**2)
#   q2 = np.arctan2(sq2, R[0][0])
#   q3 = np.arctan2(-1*R[0][1], -1*R[0][2])
#   return np.array([q1, q2, q3])


def calculate_euler(r):
    if r[0][0] == 1:
        return array([arctan2(r[2][1], r[1][1]), 0, 0])
    elif r[0][0] == -1:
        return array([arctan2(-r[2][1], r[1][1]), 0, 0])
    elif r[0][0] >= 0:
        return array((arctan2(-r[1][0], r[2][0]),
                      arctan2(sqrt(r[1][0] ** 2 + r[2][0] ** 2), r[0][0]),
                      arctan2(-r[0][1], -r[0][2])))
    else:
        return array((arctan2(r[1][0], +r[2][0]),
                      arctan2(sqrt(r[1][0] ** 2 + r[2][0] ** 2), r[0][0]),
                      arctan2(r[0][1], r[0][2])))


def fk(q):
    return (shift_z(BASE).dot((rotate_z(q[0]))).dot(shift_x(BASE_SHIFT)).
            dot(rotate_y(q[1])).dot(shift_z(ELBOW)).dot(rotate_y(q[2])).
            dot(shift_z(SHIFT)).dot(shift_x(FOREARM)).dot(rotate_x(q[3])).
            dot(rotate_y(q[4])).dot(rotate_x(q[5])).dot(shift_x(TOOl)))


def rk(t):
    pos = t[0:3, 3] - t[0:3, 0] * TOOl

    for_simple = sqrt(FOREARM ** 2 + SHIFT ** 2)

    q1 = arctan2(pos[1], pos[0])
    y, x = sqrt(pos[0] ** 2 + pos[1] ** 2) - BASE_SHIFT, \
           pos[2] - BASE
    q2 = (pi / 2 - arctan(x / y) -
          arccos((-for_simple ** 2 + ELBOW ** 2 + x ** 2 + y ** 2) /
                 (2 * ELBOW * sqrt(x ** 2 + y ** 2))))

    q3 = (pi / 2 - arccos((for_simple ** 2 + ELBOW ** 2 - x ** 2 - y ** 2) /
                          (2 * for_simple * ELBOW)) + arctan(SHIFT / FOREARM))

    frame = (shift_z(BASE).dot((rotate_z(q1))).dot(shift_x(BASE_SHIFT)).
             dot(rotate_y(q2)).dot(shift_z(ELBOW))
             .dot(rotate_y(q3)).dot(shift_z(SHIFT)).dot(shift_x(FOREARM)))

    q4, q5, q6 = calculate_euler(inv(frame[0:3, 0:3]).dot(t[0:3, 0:3])) + \
                 array([pi, 0, pi])

    return q1, q2, q3, q4, q5, q6


a = fk([0.5, -0.5, 0.5, 1, 1, 1])
print(a)
print(rk(a))
print(calculate_euler(loadtxt('matrix')))
print('')

l1 = 1075
l2 = 1280
