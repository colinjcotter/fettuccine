from firedrake import *

m = UnitIntervalMesh(50)
mesh = ExtrudedMesh(m, 1)

deg = 1
NewX = VectorFunctionSpace(mesh, "CG", 1, dim=3)
mesh = Mesh(Function(NewX))

V = VectorFunctionSpace(mesh, "CG", deg)
Q = VectorFunctionSpace(mesh, "DG", deg-1)
A = FunctionSpace(mesh, "CG", deg-1)

W = V*
