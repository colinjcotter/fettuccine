from firedrake import *

m = UnitIntervalMesh(100)
mesh = m#ExtrudedMesh(m, 1)

x = SpatialCoordinate(mesh)

deg = 1
NewX = Function(VectorFunctionSpace(mesh, "CG", 1, dim=3))
NewX.interpolate(as_vector([x[0],0,0]))

mesh = Mesh(Function(NewX))
x = SpatialCoordinate(mesh)

V = VectorFunctionSpace(mesh, "CG", deg)
Q = VectorFunctionSpace(mesh, "DG", deg-1)
A = FunctionSpace(mesh, "DG", deg-1)

# q, t, b, p, lambda, mu, alpha, beta
W = V * V * V * Q * Q * Q * A * A

w = Function(W)
dw = TestFunction(W)

#boundary conditions
qstar0 = Constant(as_vector([0,0,0]))
qstar1 = Constant(as_vector([1.0,0,0]))
tstar0 = Constant(as_vector([1.0,0.,0]))
tstar1 = Constant(as_vector([1.0,0.,0]))
bstar0 = Constant(as_vector([0,1.0,0]))
bstar1 = Constant(as_vector([0,1.0,0]))

#initial guesses
# q, t, b, p, lambda, mu, alpha, beta
q, t, b, p, ll, mu, alpha, beta = w.split()

#splitting for equations
q, t, b, p, ll, mu, alpha, beta = split(w)

kappasq = alpha**2
tausq = alpha**2 + beta**2
F = ((kappasq + tausq)**2/kappasq)*dx

inner(p, q.dx(0))

S = F + (
    inner(ll, t.dx(0) - alpha*cross(b, t))
    + inner(mu, b.dx(0) - beta*cross(t, b))
    + inner(p, q.dx(0) - t)
)*dx

eqn = derivative(S, w, dw)

bcs = [DirichletBC(W.sub(0), qstar0, 1),
       DirichletBC(W.sub(0), qstar1, 2),
       DirichletBC(W.sub(1), tstar0, 1),
       DirichletBC(W.sub(1), tstar1, 2),
       DirichletBC(W.sub(2), bstar0, 1),
       DirichletBC(W.sub(2), bstar1, 2)]
       

solver_parameters = {'ksp_type':'preonly',
                     'mat_type':'aij',
                     'pc_type':'lu',
                     'snes_monitor':None}

q, ll, mu, p, b, t, alpha, gamma = TrialFunctions(W)
dq, dll, dmu, dp, db, dt, dalpha, dgamma = TestFunctions(W)

solve(eqn == 0, w, bcs = bcs,
      solver_parameters = solver_parameters)
