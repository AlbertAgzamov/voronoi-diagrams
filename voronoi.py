import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from math import sqrt


# Первым пяти точкам из массива points соответствуют оранжевые точки с номерами 0,1,2,3,4
# Оставшимся - синие точки с номерами 5,6,7,8,9
points = np.array([[31,79], [56, 3], [94,88], [38,60], [41,8], \
                  [59,92], [80,88], [81,60], [38,40], [89,37]])

points = np.append(points, [[999,999], [-999,999], [999,-999], [-999,-999]], axis = 0)

test_points = np.array([[50,40],[90,60],[20,30]])

vor = Voronoi(points)
fig = voronoi_plot_2d(vor, show_vertices = False)
plt.plot(points[:5][:,0], points[:5][:,1], 'o')
plt.plot(points[5:10][:,0], points[5:10][:,1], 'ob')
plt.plot(test_points[:,0], test_points[:,1], 'og')

for j,p in enumerate(points):
    plt.text(p[0]-0.24, p[1]+0.24, j, ha='right')

for j,p in enumerate(test_points):
    plt.text(p[0]-0.26, p[1]+0.26, p, ha='right')
    
#plt.xlim(-550, 150); plt.ylim(-50, 200)
plt.xlim(-50, 150); plt.ylim(-50, 150)


for i, p in enumerate(points):
    r = vor.regions[vor.point_region[i]]
    if not -1 in r:
        polygon = [vor.vertices[i] for i in r]
        if i < 5:
            plt.fill(*zip(*polygon), '#F5BC38')
        else:
            plt.fill(*zip(*polygon), '#07D179')
plt.grid()
plt.show()



points = np.array([[31,79], [56, 3], [94,88], [38,60], [41,8], \
                  [59,92], [80,88], [81,60], [38,40], [89,37]])

def classify_point(point):

    def distance(p1,p2):
        return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        
        
    nearest_p = [9999,9999]
    index_of_nearest_point = 15
    for i,p in enumerate(points):
        if (distance(point, p) < distance(point, nearest_p)):
            nearest_p = p
            index_of_nearest_point = i
    res = 0
    if index_of_nearest_point < 5:
        res = 1
    else: 
        res = -1
    return res
  
# Возможные классы точек {-1,1}  
print(classify_point(np.array([50,40])))
# Result: -1   (точка [50,40]^T приналежит 1-му классу)
print(classify_point(np.array([90,60])))
# Result: -1   (точка [90,60]^T приналежит 1-му классу)
print(classify_point(np.array([20,30])))
# Result: -1   (точка [20,30]^T приналежит 1-му классу)
   