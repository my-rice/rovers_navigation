import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import matplotlib.pyplot as plt

def my_cilinder(x,u):
    r = 0.4
    k = 1000
    height = 100
    
    temp = height*(-1/(1+np.exp(-k*((x[0]-u[0])**2+(x[1]-u[1])**2-(r**2)))) + 1)
    return temp


bound1 = -1.5
bound2 = 1.5
bound3 = -1.0
bound4 = 1.0

resolution = 0.1

def V(x,y):
    return x**2+y**2


def V2(x,y):
    a = 0.5**2
    b = 0.5**2
    return sympy.sqrt((x**2/(a)) + (y**2/(b)))


def cost(x,y):
    value = 0.05
    obs_points = np.array(np.mat('0 0 0 0 0;0.2 0.4 0.6 0.8 -0.8;0 0 0 0 0'))
    goal_points = np.array(np.mat('-1.4; -0.8; 0'))
    state = np.array([x,y])
    gauss_sum = 0

    for i in range(np.size(obs_points,axis=1)):        
        gauss_sum += my_cilinder(state[:2],obs_points[:2,i])
    
    goal_point_cost = (state[0]-goal_points[0])**2 + (state[1]-goal_points[1])**2 # + abs((state[0]-goal_points[0])) + abs((state[1]-goal_points[1]))
    
    cost_value = 30*goal_point_cost + gauss_sum + 10*(np.exp(-0.5*((state[0]-(-1.5))/value)**2)/(value*np.exp(2*np.pi))
                + np.exp(-0.5*((state[0]-1.5)/value)**2)/(value*np.exp(2*np.pi)) + np.exp(-0.5*((state[1]-1.0)/value)**2)/(value*np.exp(2*np.pi))
                + np.exp(-0.5*((state[1]-(-1.0))/value)**2)/(value*np.exp(2*np.pi)))
    return cost_value

# x,y=sympy.symbols('x y')
# fun=cost(x,y)
# gradfun=[sympy.diff(fun,var) for var in (x,y)]
# numgradfun=sympy.lambdify([x,y],gradfun)

x = np.linspace(bound1,bound2,500)
y = np.linspace(bound3,bound4,500)
X,Y=np.meshgrid(x,y)

#graddat=numgradfun(X,Y)

Z = cost(X,Y)
grad_x, grad_y = np.gradient(Z, x, y)

fig = plt.figure(figsize=(13,10))
ax = fig.add_subplot()

Z = np.sqrt(grad_x**2 + grad_y**2)

pcolormesh = ax.pcolormesh(X, Y, Z, cmap='coolwarm', zorder=1, vmax=300)

# Plot the level curves of the cost function
contour= ax.contour(X, Y, Z, cmap='coolwarm', linestyles='dashed', zorder=2, alpha=1, vmax=300) #vmin and vmax are used to make the contour lines more visible

# Add a colorbar
fig.colorbar(pcolormesh,ax=ax) 


# plt.pcolor(X, Y, Z,vmax=400)
# plt.colorbar()
plt.savefig("prova1.png")
plt.close()
