from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
from numpy import sin, cos, pi, sqrt
from itertools import product, combinations
import re


# Initial setup
# =============

def getInput(prompt, regex, default = False, isArray = False):
    while True:
        text = raw_input(prompt)
        if text == "" and default is not False:
            out = np.array(default) if isArray else default
            print "Used default: " + str(out)
            break
        if re.match(regex, text):
            out = np.array(text.split()).astype(float) if isArray else float(text)
            break
    return out
            
dim = getInput("Enter 3 for 3D plot, or 2 for the 2D projection:\n",\
                    "^[23]$")

if dim == 2:
    fovDist = getInput("Enter distance from viewing position to viewing\n"\
                            "   plane/window...\n"\
                            "Leave blank to use default.\n"\
                            "Must be greater than 0.0 and less than 15.0\n"\
                            "The smaller the value is the nearer one is to the plane,\n"\
                            "   and thus the wider the viewing angle.\n",\
                        "^0?\.\d*[1-9]\d*|[1-9](\.\d*)?|1[0-4](\.\d*)?$",\
                        5.0)
                        
    lightVec = getInput("Enter light vector...\n"\
                           "Leave blank to use defaults.\n"\
                           "Note that camera looks down positive x-axis from zero.\n"\
                           "Form: (\"x y z\" | -1.0 " + u"\u2264" + " x,y,z " + u"\u2264" + " 1.0)\n"\
                           "Example: -1 -1 -1 " + u"\u2192" + " light direction: left, toward\n"\
                           "   camera, down\n",\
                        "^(-?(1(\.0*)?|(0(\.\d*)?|\.\d+))\s+){2}-?(1(\.0*)?|(0(\.\d*)?|\.\d+))\s*$",\
                        [1.0,0.0,-1.0], True)
    
    lineOp = getInput("Enter opacity for frame of shaded 3D shapes...\n"\
                        "0.0 (transparent) to 1.0 (solid)\n",\
                    "^\.\d*|0(\.\d*)?|1(\.0*)?$",\
                    0.2)
                        
cubeSize = getInput("Enter size for cube (length of each side)...\n"\
                         "Leave blank to use default.\n"\
                         "Size must be greater than 0.0 and smaller than 8.0,\n"\
                         "   room dimension is 10*10*10.\n",\
                    "^0?\.\d*[1-9]\d*|[1-7](\.\d*)?$",\
                    2.5)

cubePos = getInput("Enter position for cube...\n" \
                        "Leave blank to use defaults.\n" \
                        "Note that camera looks down positive y-axis from zero,\n" \
                        "   x:left/right y:depth z:height\n"\
                        "Form: (\"x y z\" | -5.0 " + u"\u2264" + " x,z " + u"\u2264" \
                        " 5.0, 0.0 " + u"\u2264" + " y " + u"\u2264" +" 10.0)\n",\
                    "^-?(\.\d+|[0-5](\.\d*)?)\s+(10(\.0*)?|\d(\.\d*)?|\.\d+)\s+-?(\.\d+|[0-5](\.\d*)?)\s*$",\
                    [0.0,5.0,0.0], True)

cubeRot = getInput("Enter rotation for cube...\n" \
                        "Leave blank to use defaults.\n" \
                        "Rotations are in radians.\n" \
                        "Form: \"rX rY rZ\" where the cube is rotated around its\n" \
                        "   center in the yz-plane rX*" + u"\u03C0" + " radians, and similarly\n"\
                        "   for the xz and xy-planes. rX, rY, rZ " + u"\u2208" + " [0.0, 2.0).\n",\
                    "^((\.\d+|[01](\.\d*)?)\s+){2}(\.\d+|[01](\.\d*)?)\s*$",\
                    [0.28,0.25,1.55], True)

ddhSize = getInput("Enter size for dodecahedron (length of edges)...\n"\
                         "Leave blank to use default.\n"\
                         "Size must be greater than 0.0 and smaller than 4.0.\n",\
                    "^0?\.\d*[1-9]\d*|[1-3](\.\d*)?$",\
                    0.5)

ddhPos = getInput("Enter position for dodecahedron...\n" \
                        "Leave blank to use defaults.\n" \
                        "Note that camera looks down positive y-axis from zero,\n" \
                        "   x:left/right y:depth z:height\n"\
                        "Form: (\"x y z\" | -5.0 " + u"\u2264" + " x,z " + u"\u2264" \
                        " 5.0, 0.0 " + u"\u2264" + " y " + u"\u2264" +" 10.0)\n",\
                    "^-?[0-5](\.\d*)?\s+(10(\.0*)?|\d(\.\d*)?)\s+-?[0-5](\.\d*)?\s*$",\
                    [-4.0,7.0,-2.0], True)

sphSize = getInput("Enter size for sphere (radius)...\n"\
                         "Leave blank to use default.\n"\
                         "Size must be greater than 0.0 and smaller than 5.0.\n",\
                    "^0?\.\d*[1-9]\d*|[1-4](\.\d*)?$",\
                    1.0)

sphPos = getInput("Enter position for sphere...\n" \
                        "Leave blank to use defaults.\n" \
                        "Note that camera looks down positive y-axis from zero,\n" \
                        "   x:left/right y:depth z:height\n"\
                        "Form: (\"x y z\" | -5.0 " + u"\u2264" + " x,z " + u"\u2264" \
                        " 5.0, 0.0 " + u"\u2264" + " y " + u"\u2264" +" 10.0)\n",\
                    "^-?[0-5](\.\d*)?\s+(10(\.0*)?|\d(\.\d*)?)\s+-?[0-5](\.\d*)?\s*$",\
                    [2.0,3.0,2.0], True)


# Functions to generate cube, dodecahedron, sphere, and room
# ==========================================================

def genCubeEdges(size, pos, rot):
    # Generate array of 3D coordinates representing cube
    #    vertices - the cube is unrotated, centered at (0, 0, 0)
    d = [-size/2, size/2]
    verts = np.array(list(product(d,d,d)))
    
    # Find all pairs of vertices that form an edge using
    #    the distance between them... Stack in array
    edges = np.empty([0,3])
    for i in range (0,8):
        for j in range(i+1,8):
            if np.sum(np.abs(verts[i] - verts[j])) == cubeSize:
                edges = np.vstack([edges, verts[i], verts[j]])
    
    # Rotation around x, y, and z-axes
    rot = rot * pi
    for i in range(0,24):
        ytmp = edges[i][1]
        ztmp = edges[i][2]
        edges[i][1] = ytmp*cos(rot[0]) - ztmp*sin(rot[0])
        edges[i][2] = ytmp*sin(rot[0]) + ztmp*cos(rot[0])
    for i in range(0,24):
        xtmp = edges[i][0]
        ztmp = edges[i][2]
        edges[i][0] = xtmp*cos(rot[1]) + ztmp*sin(rot[1])
        edges[i][2] = -xtmp*sin(rot[1]) + ztmp*cos(rot[1])
    for i in range(0,24):
        xtmp = edges[i][0]
        ytmp = edges[i][1]
        edges[i][0] = xtmp*cos(rot[2]) - ytmp*sin(rot[2])
        edges[i][1] = xtmp*sin(rot[2]) + ytmp*cos(rot[2])
    
    # Move cube to correct position
    for i in range (0,24):
        edges[i] += pos
    
    # Return array of vertex pairs. Format is such that
    #    indices 0 and 1 make an edge, as do indices 2
    #    and 3, 4 and 5, and so on...
    return edges

def genDodecaEdges(size, pos):
    # This is the Golden Ratio
    gr = (1 + 5**0.5) / 2
    
    # Generate array of 3D coordinates representing dodecahedron
    #    vertices - centered at (0, 0, 0). Thanks to Wikipedia
    #    for the Cartesian coordinate definition!
    d0 = [0.0]
    d1 = [-ddhSize*(gr/2.0), ddhSize*(gr/2.0)]
    d2 = [-ddhSize/2.0, ddhSize/2.0]
    d3 = [-ddhSize*gr*gr/2.0, ddhSize*gr*gr/2.0]
    verts = np.concatenate((np.array(list(product(d1,d1,d1))),\
                            np.array(list(product(d0,d2,d3))),\
                            np.array(list(product(d2,d3,d0))),\
                            np.array(list(product(d3,d0,d2)))), axis=0)

    # Find all pairs of vertices that form an edge using
    #    the distance between them... Stack in array
    edges = np.empty([0,3])
    for i in range (0, 20):
        for j in range (i+1, 20):
            if size - size/10 <=\
            ((verts[i][0] - verts[j][0])**2 +\
             (verts[i][1] - verts[j][1])**2 +\
             (verts[i][2] - verts[j][2])**2) ** 0.5\
             <= size + size/10:
                edges = np.vstack([edges, verts[i], verts[j]])
    
    # Move dodecahedron to correct position
    for i in range (0,60):
        edges[i] += pos
    
    # Return array of vertex pairs. Same as for cube
    return edges

def genSphereFrame(r, pos):
    # Set up horizontal and vertical angular divisions
    phi = np.arange(0,2*pi,pi/10.0)
    theta = np.arange(0,pi,pi/10.0)
    
    # Create wireframe that consists of an array of horizontal
    #    lines and an array of vertical lines
    frameA = np.empty([0,3])
    for j in range(0,20):
        for i in range(0,10):
            frameA = np.vstack([frameA, [r*sin(theta[i])*cos(phi[j]),\
                                         r*sin(theta[i])*sin(phi[j]),\
                                         r*cos(theta[i])]])
        frameA = np.vstack([frameA, [r*sin(pi)*cos(phi[j]),\
                                     r*sin(pi)*sin(phi[j]),\
                                     r*cos(pi)]])
    frameB = np.empty([0,3])
    for i in range(0,10):
        for j in range(0,20):
            frameB = np.vstack([frameB, [r*sin(theta[i])*cos(phi[j]),\
                                         r*sin(theta[i])*sin(phi[j]),\
                                         r*cos(theta[i])]])
        frameB = np.vstack([frameB, [r*sin(theta[i])*cos(phi[0]),\
                                     r*sin(theta[i])*sin(phi[0]),\
                                     r*cos(theta[i])]])
    
    # Move sphere to correct position
    for i in range(0,220):
        frameA[i] += pos
    for i in range(0,210):
        frameB[i] += pos
    
    # Return both arrays
    return [frameA, frameB]

def genRoom():
    # ...Generate room edges
    d = [-4.9, 4.9]
    verts = np.array(list(product(d,d,d)))
    pos = np.array([0.0, 5.0, 0.0])
    for i in range (0,8):
        verts[i] += pos
    edges = np.empty([0,3])
    for i in range (0,8):                      
        for j in range(i+1,8):
            if np.sum(np.abs(verts[i] - verts[j])) == 9.8:
                edges = np.vstack([edges, verts[i], verts[j]])
    return edges
           

# Functions for drawing perspective projection and shading
# ========================================================

def translation(frame):
    frame2D = np.empty([0,2])
    for i in range(0,frame.size/3):
        frame2D = np.vstack([frame2D, [frame[i][0]*fovDist/frame[i][1],\
                                    frame[i][2]*fovDist/frame[i][1]]])
    return frame2D

def draw2D(frame2D, c, a=1):
    for i in range(0,frame2D.size/2,2):
        plt.plot([frame2D[i][0], frame2D[i+1][0]],\
                 [frame2D[i][1], frame2D[i+1][1]],\
                 '-', alpha=a, color=c, linewidth=1)
                 
def mag(vect):
    return sqrt(np.dot(vect, vect))

def getSurfNormal(vertA, vertB, vertC):
    return np.cross(vertA - vertB, vertC - vertB)

def getAngleBetween(vect1, vect2):
    return np.degrees(np.arccos(np.dot(vect1, vect2) / \
                               (mag(vect1) * mag(vect2))))

# Culling and Shading for Cube
def cubeShade(vertA, vertB, vertC, vertA2D, vertB2D, vertC2D, vertD2D, lightVect):
    normal = getSurfNormal(vertA, vertB, vertC)
    cameraPos = np.array([0,0,0])
    lineOfSight = vertA - cameraPos
    
    visibilityAngle = getAngleBetween(normal, lineOfSight)
    reflectedAngle = getAngleBetween(normal, lightVect)
        
    if visibilityAngle < 90:
        fill([vertA2D[0],vertB2D[0],vertC2D[0],vertD2D[0]],\
             [vertA2D[1],vertB2D[1],vertC2D[1],vertD2D[1]],\
             'k', alpha = reflectedAngle/180)
          

# Generation of shapes and drawing in 2D/3D
# =========================================

cubeEdges = genCubeEdges(cubeSize, cubePos, cubeRot)
ddhEdges = genDodecaEdges(ddhSize, ddhPos)
sphFrameA = genSphereFrame(sphSize, sphPos)[0]
sphFrameB = genSphereFrame(sphSize, sphPos)[1]
roomEdges = genRoom()

if (dim == 2):
    # Setup figure
    plt.figure(1, facecolor='white');
    plt.clf();
    plt.axis([-5,5,-5,5])

    # Cube translation
    cubeEdges2D = translation(cubeEdges)
    # Draw cube projection
    draw2D(cubeEdges2D, 'lime', lineOp)
    # Cube shading.The correct vertices were recorded and
    #    entered manually to shade the faces of the cube.
    #    *Change this to an algorithm.*         3-------7
    #                                         / |     / |
    #                      (shd[] indices    1-------5  |
    #                       for unrotated    |  2----|--6
    #                       cube)            | /     | /
    #                                        0-------4
    #
    shd = np.array([cubeEdges[0],cubeEdges[1],cubeEdges[3],cubeEdges[7],\
                      cubeEdges[5],cubeEdges[9],cubeEdges[13],cubeEdges[15]])
    shd2D = np.array([cubeEdges2D[0],cubeEdges2D[1],cubeEdges2D[3],cubeEdges2D[7],\
                        cubeEdges2D[5],cubeEdges2D[9],cubeEdges2D[13],cubeEdges2D[15]])
    cubeShade(shd[0],shd[4],shd[5],shd2D[0],shd2D[4],shd2D[5],shd2D[1],lightVec)
    cubeShade(shd[2],shd[0],shd[1],shd2D[2],shd2D[0],shd2D[1],shd2D[3],lightVec)
    cubeShade(shd[4],shd[6],shd[7],shd2D[4],shd2D[6],shd2D[7],shd2D[5],lightVec)
    cubeShade(shd[2],shd[6],shd[4],shd2D[2],shd2D[6],shd2D[4],shd2D[0],lightVec)
    cubeShade(shd[1],shd[5],shd[7],shd2D[1],shd2D[5],shd2D[7],shd2D[3],lightVec)
    cubeShade(shd[6],shd[2],shd[3],shd2D[6],shd2D[2],shd2D[3],shd2D[7],lightVec)

    # Dodecahedron translation
    ddhEdges2D = translation(ddhEdges)
    # Draw dodecahedron projection
    draw2D(ddhEdges2D, 'crimson')
    
    # Sphere translation
    sphEdges2D = translation(sphFrameA)
    sphEdges2Db = translation(sphFrameB)
    # Draw sphere projection. This is messy, it needs to be fixed and
    #    then integrated with the draw2D function somehow
    for i in range(0,219):
        if (i % 11 != 10):
            plt.plot([sphEdges2D[i][0], sphEdges2D[i+1][0]],\
                     [sphEdges2D[i][1], sphEdges2D[i+1][1]],\
                     '-', linewidth=1.0, color='dodgerblue')
    for i in range(0,209):
        if (i % 21 != 20):
            plt.plot([sphEdges2Db[i][0], sphEdges2Db[i+1][0]],\
                     [sphEdges2Db[i][1], sphEdges2Db[i+1][1]],\
                     '-', linewidth=1.0, color='dodgerblue')
    
    # Room translation
    roomEdges2D  = translation(roomEdges)
    # Draw room projection
    draw2D(roomEdges2D, 'k')
        
    plt.show()

else:
    
    # Setup figure
    fig = plt.figure()
    plt.clf()
    ax = fig.gca(projection='3d')
    ax.set_aspect("equal")
    
    # Draw cube
    for i in range(0,24,2):
        ax.plot3D(*zip(cubeEdges[i],cubeEdges[i+1]), color='lime')
    
    # Draw dodecahedron
    for i in range(0,60,2):
        ax.plot3D(*zip(ddhEdges[i],ddhEdges[i+1]), color='crimson')

    # Draw sphere
    for i in range(0,219):
        if (i % 11 != 10):
            ax.plot3D(*zip(sphFrameA[i],sphFrameA[i+1]), color='dodgerblue')
    for i in range(0,209):
        if (i % 21 != 20):
            ax.plot3D(*zip(sphFrameB[i],sphFrameB[i+1]), color='dodgerblue')    

    # Draw room
    for i in range(0,24,2):
        ax.plot3D(*zip(roomEdges[i],roomEdges[i+1]), color="k")

    # More setup
    plt.axis([-5,5,0,10])
    ax.set_zbound(-5,5)
    
    plt.show()
