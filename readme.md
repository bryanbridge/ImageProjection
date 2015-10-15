**Generating 3D shapes, perspective projection, and shading**

A cube, sphere, and dodecahedron are generated as cartesian coordinates, and their edges either plotted in 3D or drawn in 2D with perspective projection. The size and position of the objects can be changed, the cube can be rotated. In 2D the FOV is adjustable, and a light vector must be set... The cube will is shaded with a basic gradient method using its surface normals. There is a quick and easy to use but in-depth setup menu run in the console - not only was this great linear algebra review, but I got some quality regex time logged.

All graphics were done with numpy and matplotlib, plot and fill functions, etc... Looking back I'm certain there was a better way to accomplish what I wanted.

This was another group presentation for my Numerical Analysis II course, however I wrote the program, because hey, learning how 3D graphics are coded is fun! I've also included the write-up I did.

Since finishing the class I've cleaned up my work and am looking forward to adding some things when I get the chance: shading for the dodecahdron, the ability to change the camera position, and finally interpolative shading.