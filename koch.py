# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 22:41:53 2017

@author: Owner
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class direct_vec:

    def __init__(self, theta):
        self.vx, self.vy = np.cos(theta), np.sin(theta)
        self.theta = theta

    def rot(self, theta):
        return direct_vec(theta + self.theta)

    def slope(self):
        return self.vy / self.vx

    def get_cos(self):
        return self.vx

    def get_sin(self):
        return self.vy

class point:

    def __init__(self, x0, y0):
        self.x = x0
        self.y = y0

    def get_point(self):
        return (self.x, self.y)

    def set_point(self, x=None, y=None):
        if x!=None:
            self.x = x
        if y!=None:
            self.y = y

    def copy(self):
        return point(self.x, self.y)

def plot_linear(point, vec, length):
    x0, y0= point.get_point()
    x_range = np.linspace(start=x0, stop=x0 + vec.vx * length, num=20)
    plt.plot(x_range, vec.slope() * (x_range - x0) + y0)


def koch(p0, vec, L, n):
    p = p0.copy()
    if n < 0:
        print("error")
        return None
    if n == 0:
        # 直線のみ
        plot_linear(p, vec, 3 * L)
        return
    # 再帰数nが正の数の場合
    # 裾野部分
    koch(p, vec, L / 3, n-1)
    p.set_point(p0.x + vec.get_cos() * 2 * L, p0.y + vec.get_sin() * 2 * L)
    koch(p, vec, L / 3, n-1)
    # 山の左を作る
    p.set_point(p0.x + vec.get_cos() * L, p0.y + vec.get_sin() * L)
    koch(p, vec.rot(np.pi / 3), L / 3, n - 1)
    # 山の右を作る
    p.set_point(p0.x + vec.get_cos() * L + vec.rot(np.pi / 3).vx * L, p0.y + vec.get_sin() * L + vec.rot(np.pi / 3).vy * L)
    koch(p, vec.rot(- np.pi / 3), L / 3, n-1)


vec = direct_vec(0)
L = 1
N = 3
p0 = point(0, 0)
fig = plt.figure(figsize=(10,10))

def koch_outer(N, p0, vec, L):
    plt.cla()

    p = p0.copy()
    rot_vec = vec.rot(np.pi / 3)
    koch(p, rot_vec, L, N)

    p.set_point(p0.x + rot_vec.get_cos() * 3 * L, p0.y + rot_vec.get_sin() * 3 * L)
    rot_vec = vec.rot(-np.pi / 3)
    koch(p, rot_vec, L, N)

    p.set_point(p0.x + vec.get_cos() * 3 * L, p0.y + vec.get_sin() * 3 * L)
    rot_vec = vec.rot(-np.pi / 3)
    koch(p, vec.rot(-np.pi), L, N)

    plt.title("Koch Curve n=" + str(N))
    # 描画方法を中心指定のｘ軸に（水平な）三角形にして描画判を正方形に指定していきたい
    # center_point = point(p0.x + (vec.get_cos() + vec.rot(np.pi / 6).get_cos()) * L,
    #                     p0.y + (vec.get_sin() + vec.rot(np.pi / 6).get_sin()) * L)
    plt.xlim(-L / 2, 7 * L / 2)
    plt.ylim(-L, 3 * L)


"""
def koch_inter(N, vec, L):
    plt.cla()
    koch(0, 0, vec, L, N)
    koch(vec.get_cos() * 3 * L, vec.get_sin() * 3 * L, direct_vec(-np.pi / 3), L, N)
    koch(3 * L, 0, direct_vec(-np.pi), L, N)
    plt.title("Koch Curve n=" + str(N))
    # 描画方法を中心指定のｘ軸に（水平な）三角形にして描画判を正方形に指定していきたい
    plt.xlim(-L / 2, L * 7 / 2)
    plt.ylim(-L, 3 * L)
"""

ani = animation.FuncAnimation(fig, koch_outer, fargs=(p0, vec, L), interval=1000, frames=N)
# plt.savefig("Koch_curve.png")
ani.save("koch.gif", writer='imagemagick')
plt.show()
