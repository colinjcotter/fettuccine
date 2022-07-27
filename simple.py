from firedrake import *

m = UnitIntervalMesh(100)
mesh = m#ExtrudedMesh(m, 1)

x = SpatialCoordinate(mesh)

NewX = Function(VectorFunctionSpace(mesh, "CG", 1, dim=3))
NewX.interpolate(as_vector([x[0],0,0]))

mesh = Mesh(Function(NewX))
x = SpatialCoordinate(mesh)

deg = 2
V = VectorFunctionSpace(mesh, "CG", deg)
Q = FunctionSpace(mesh, "DG", deg-1)

# q, ll
W = V * Q

w = Function(W)
dw = TestFunction(W)

#splitting for equations
q, ll = split(w)
dq, dll = split(dw)

n = FacetNormal(mesh)

def D2(q):
    return q.dx(0).dx(0)

def D2n(q):
    return dot(n, dot(n, q.dx(0).dx(0)))

def D1(q):
    return q.dx(0)

def both(q):
    return q('+') + q('-')

eta = Constant(10.)
e = avg(CellVolume(mesh))

eqn = inner(D2(q),D2(dq))*dx
eqn += inner(avg(D2(q)), jump(grad(dq),n))*dS
eqn += inner(avg(D2(dq)),jump(grad(q ),n))*dS
eqn += eta/e*inner(jump(grad(q),n),jump(grad(dq),n))*dS
eqn += ll*inner(D1(q),D1(dq))*dx
eqn += dll*(inner(D1(q),D1(q))-1)*dx

eqn2 = eqn + dll*ll*dx
Jp = derivative(eqn2, w)

q, ll = w.split()
x = SpatialCoordinate(mesh)
q.interpolate(x + as_vector([0,0,1.0e-4]))

qstar0 = as_vector([0,0,0])
qstar1 = as_vector([0.999,0,0])

bcs = [DirichletBC(W.sub(0), qstar0, 1),
       DirichletBC(W.sub(0), qstar1, 2)]


solver_parameters = {'ksp_type':'gmres',
                     #'ksp_monitor': None,
                     'ksp_atol': 1.0e-10,
                     'mat_type':'aij',
                     'pc_type':'lu',
                     "pc_factor_mat_ordering_type": "rcm",
                     "pc_factor_shift_type": "nonzero",
                     'snes_monitor':None}

v_basis = VectorSpaceBasis(constant=True)
nullspace = MixedVectorSpaceBasis(W, [W.sub(0), v_basis])

solve(eqn == 0, w, bcs = bcs, Jp=Jp, nullspace=nullspace,
      solver_parameters = solver_parameters)
