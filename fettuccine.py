from firedrake import *

m = UnitIntervalMesh(50)
mesh = ExtrudedMesh(m, 1)

deg = 1
NewX = VectorFunctionSpace(mesh, "CG", 1, dim=3)
mesh = Mesh(Function(NewX))

V = VectorFunctionSpace(mesh, "CG", deg)
Q = VectorFunctionSpace(mesh, "DG", deg-1)
A = FunctionSpace(mesh, "DG", deg-1)

W = V * V * V * Q * Q * Q * A * A

w = Function(W)
dw = TestFunction(W)

q, ll, mu, p, b, t, alpha, gamma = split(w)

qstar0 = Constant(as_vector([0,0,0]))
qstar1 = Constant(as_vector([1.0,0,0]))
tstar0 = Constant(as_vector([1.0,0,0]))
tstar1 = Constant(as_vector([1.0,0,0]))
bstar0 = Constant(as_vector([0,1.0,0]))
bstar1 = Constant(as_vector([0,1.0,0]))

F = alpha**2*(2 + gamma**2)*dx

inner(p, q.dx(0))

S = F + (
    - inner(ll.dx(0), t)
    - inner(alpha*ll, cross(b, t))
    - inner(mu.dx(0), b)
    - inner(alpha*gamma*mu, cross(t, b))
    + inner(p, q.dx(0) - t)
)*dx + (
    inner(ll, tstar1)*ds_v(2)
    -inner(ll, tstar0)*ds_v(1)
    +inner(mu, bstar1)*ds_v(2)
    -inner(mu, bstar0)*ds_v(1)
)

eqn = derivative(S, w, dw)

bcs = [DirichletBC(W.sub(0), qstar0, 1),
       DirichletBC(W.sub(0), qstar1, 2)]

solver_parameters = {'ksp_type':'preonly',
                     'pc_type':'lu',
                     'snes_monitor':None,
                     'pc_factor_mat_solver_type':'mumps'}



nvp = NonlinearVariationalProblem(eqn, w, bcs = bcs)
nvs = NonlinearVariationalSolver(nvp, solver_parameters = solver_parameters)

nvs.solve()


