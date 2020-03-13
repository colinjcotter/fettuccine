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
q, ll, mu, p, b, t, alpha, gamma = w.split()

#splitting for equations
q, ll, mu, p, b, t, alpha, gamma = split(w)

F = (alpha**2 + gamma**2)*dx

inner(p, q.dx(0))

S = F + (
    - inner(ll.dx(0), t)
    - inner(alpha*ll, cross(b, t))
    - inner(mu.dx(0), b)
    - inner(gamma*mu, cross(t, b))
    + inner(p, q.dx(0) - t)
)*dx + (
    inner(ll, tstar1)*ds(2)
    -inner(ll, tstar0)*ds(1)
    +inner(mu, bstar1)*ds(2)
    -inner(mu, bstar0)*ds(1)
)

eqn = derivative(S, w, dw)

bcs = [DirichletBC(W.sub(0), qstar0, 1),
       DirichletBC(W.sub(0), qstar1, 2)]

solver_parameters = {'ksp_type':'gmres',
                     'ksp_max_it':80,
                     'mat_type':'aij',
                     'ksp_monitor':None,
                     'ksp_converged_reason':None,
                     'pc_type':'fieldsplit',
                     'pc_fieldsplit_type':'additive',
                     'fieldsplit_0_ksp_type':'preonly',
                     'fieldsplit_0_pc_type':'lu',
                     'fieldsplit_1_ksp_type':'preonly',
                     'fieldsplit_1_pc_type':'lu',
                     'fieldsplit_2_ksp_type':'preonly',
                     'fieldsplit_2_pc_type':'lu',
                     'fieldsplit_3_ksp_type':'preonly',
                     'fieldsplit_3_pc_type':'lu',
                     'fieldsplit_4_ksp_type':'preonly',
                     'fieldsplit_4_pc_type':'lu',
                     'fieldsplit_5_ksp_type':'preonly',
                     'fieldsplit_5_pc_type':'lu',
                     'fieldsplit_6_ksp_type':'preonly',
                     'fieldsplit_6_pc_type':'lu',
                     'fieldsplit_7_ksp_type':'preonly',
                     'fieldsplit_7_pc_type':'lu',                     
                     'snes_monitor':None}

q, ll, mu, p, b, t, alpha, gamma = TrialFunctions(W)
dq, dll, dmu, dp, db, dt, dalpha, dgamma = TestFunctions(W)

Jp = (inner(q, dq) + inner(q.dx(0), dq.dx(0))
      + inner(ll, dll) + inner(ll.dx(0), dll.dx(0))
      + inner(mu, dmu) + inner(mu.dx(0), dmu.dx(0))
      + inner(p, dp) + inner(b, db) + inner(t, dt)
      + alpha*dalpha + gamma*dgamma)*dx

assemble(Jp)

solve(eqn == 0, w, bcs = bcs, Jp = Jp,
      solver_parameters = solver_parameters)

qstar1.assign(as_vector([0.9999,0.0,0]))
solve(eqn == 0, w, bcs = bcs, Jp = Jp,
      solver_parameters = solver_parameters)
