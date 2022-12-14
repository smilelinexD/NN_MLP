import math
import re
import numpy as np
import random


def to_rad(theta):
    return theta * 2 * np.pi / 360


def to_degree(theta):
    return theta * 360 / (2 * np.pi)


class Car:
    def __init__(self, x, y, angle, r=3.0):
        self.r = r
        self.x = x
        self.y = y
        self.angle = angle
        self.log = []
        self.log_6D = []

    def reset(self, map, start=None, x=None, y=None):
        if not (x == None and y == None):
            self.x = x
            self.y = y
        else:
            rnum = random.random()
            self.x = start.d1.x + rnum * (start.d2.x - start.d1.x)
            self.y = start.d1.y + rnum * (start.d2.y - start.d1.y)
            if self.reach(map):
                self.reset(map, start=start)
        self.angle = 90.0
        self.log = []

    def run(self, theta, map):
        self.logging(theta, map)
        self.x = self.x + np.cos(to_rad(self.angle + theta)) + \
            np.sin(to_rad(theta)) * np.sin(to_rad(self.angle))
        self.y = self.y + np.sin(to_rad(self.angle + theta)) - \
            np.sin(to_rad(theta)) * np.cos(to_rad(self.angle))
        self.angle = self.angle - \
            to_degree(np.arcsin((2*np.sin(to_rad(theta))) / (self.r * 2)))
        if self.angle > 270:
            self.angle -= 360
        elif self.angle < -90:
            self.angle += 360

    def distance(self, dir, map):
        minimum = 10000
        for line in map:
            if(self.will_touch_line(line, dir)):
                dis = self.distance_to_line(line, dir)
                if dis < minimum and dis > 0:
                    minimum = dis
        return minimum

    def will_touch_line(self, line, dir):
        def dim(dot):
            if dot.x >= self.x and dot.y >= self.y:
                return 1
            if dot.x <= self.x and dot.y >= self.y:
                return 2
            if dot.x <= self.x and dot.y <= self.y:
                return 3
            if dot.x >= self.x and dot.y <= self.y:
                return 4
        ang1 = to_degree(
            np.arctan((line.d1.y - self.y) / (line.d1.x - self.x)))
        ang2 = to_degree(
            np.arctan((line.d2.y - self.y) / (line.d2.x - self.x)))

        dim1 = dim(line.d1)
        dim2 = dim(line.d2)

        if dim1 == 2 or dim1 == 3:
            ang1 += 180
        if dim1 == 4:
            ang1 += 360
        if dim2 == 2 or dim2 == 3:
            ang2 += 180
        if dim2 == 4:
            ang2 += 360

        if ang1 > ang2:
            ang1, ang2 = ang2, ang1

        if dir == 'front':
            shift = 0
        elif dir == 'left':
            shift = 45
        else:
            shift = -45

        realfacing = self.angle + shift
        if realfacing < 0:
            realfacing += 360

        if ang2 - ang1 < 180:
            return realfacing >= ang1 and realfacing <= ang2
        else:
            return realfacing <= ang1 or realfacing >= ang2

    def distance_to_line(self, line, dir):
        if dir == 'front':
            shift = 0
        elif dir == 'left':
            shift = 45
        else:
            shift = -45

        xdiff = (line.d1.x - line.d2.x, self.x -
                 (self.x + self.r * np.cos(to_rad(self.angle + shift))))
        ydiff = (line.d1.y - line.d2.y, self.y -
                 (self.y + self.r * np.sin(to_rad(self.angle + shift))))

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            raise Exception('lines do not intersect')

        d = (det([line.d1.x, line.d1.y], [line.d2.x, line.d2.y]), det([self.x, self.y], [
            self.x + self.r * np.cos(to_rad(self.angle + shift)), self.y + self.r * np.sin(to_rad(self.angle + shift))]))
        onx = det(d, xdiff) / div
        ony = det(d, ydiff) / div

        return math.sqrt((self.x - onx) ** 2 + (self.y - ony) ** 2)

    def reach(self, lines):
        for line in lines:
            if self.point_distance_line(line) < 3:
                return True
        return False

    def point_distance_line(self, line):
        xlow = self.x <= line.d1.x and self.x <= line.d2.x
        xhigh = self.x >= line.d1.x and self.x >= line.d2.x
        ylow = self.y <= line.d1.y and self.y <= line.d2.y
        yhigh = self.y >= line.d1.y and self.y >= line.d2.y
        if (xlow and ylow) or (xlow and yhigh) or (xhigh and ylow) or (xhigh and yhigh):
            return min(math.sqrt((self.x - line.d1.x) ** 2 + (self.y - line.d1.y) ** 2), math.sqrt((self.x - line.d2.x) ** 2 + (self.y - line.d2.y) ** 2))
        point = np.array([self.x, self.y])
        line_point1 = np.array([line.d1.x, line.d1.y])
        line_point2 = np.array([line.d2.x, line.d2.y])
        vec1 = line_point1 - point
        vec2 = line_point2 - point
        distance = np.abs(np.cross(vec1, vec2)) / \
            np.linalg.norm(line_point1-line_point2)
        return distance

    def logging(self, theta, map):
        tmp = []
        # tmp.append(self.x)
        # tmp.append(self.y)
        tmp.append(self.distance('front', map))
        tmp.append(self.distance('right', map))
        tmp.append(self.distance('left', map))
        tmp.append(theta)
        self.log.append(tmp)
        tmp.insert(0, self.y)
        tmp.insert(0, self.x)
        self.log_6D.append(tmp)

    def dumplog(self):
        with open('./output/train4D.txt', 'a') as f:
            for line in self.log:
                for i in line:
                    f.write('{:.7f} '.format(i))
                f.write('\n')
            f.write('\n')
        with open('./output/train6D.txt', 'a') as f:
            for line in self.log_6D:
                for i in line:
                    f.write('{:.7f} '.format(i))
                f.write('\n')
            f.write('\n')


class Dot:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Line:
    def __init__(self, d1, d2):
        self.d1 = d1
        self.d2 = d2


def load_map(directory='./???????????????.txt'):
    with open(directory, 'r') as f:
        line = f.readline()
        match = re.findall(r'-?\d+\.*\d*', line)
        car = Car(float(match[0]), float(match[1]), float(match[2]))

        line = f.readline()
        match = re.findall(r'-?\d+\.*\d*', line)
        x1 = float(match[0])
        y2 = float(match[1])
        line = f.readline()
        match = re.findall(r'-?\d+\.*\d*', line)
        x2 = float(match[0])
        y1 = float(match[1])

        end = []
        end.append(Line(Dot(x1, y1), Dot(x2, y1)))
        end.append(Line(Dot(x2, y1), Dot(x2, y2)))
        end.append(Line(Dot(x1, y1), Dot(x1, y2)))
        end.append(Line(Dot(x1, y2), Dot(x2, y2)))

        line = f.readline()
        match = re.findall(r'-?\d+\.*\d*', line)
        olddot = Dot(float(match[0]), float(match[1]))
        map = []
        outx = []
        for line in f:
            match = re.findall(r'-?\d+\.*\d*', line)
            newdot = Dot(float(match[0]), float(match[1]))
            if newdot.y < 0:
                outx.append(newdot.x)
            map.append(Line(olddot, newdot))
            olddot = newdot

        if outx[0] > outx[1]:
            outx[0], outx[1] = outx[1], outx[0]

        start = Line(Dot(outx[0], 0.0), Dot(outx[1], 0.0))
        # print(map)
        return car, end, map, start
