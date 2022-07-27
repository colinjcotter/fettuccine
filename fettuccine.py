from firedrake import *

m = UnitIntervalMesh(100)
mesh = m#ExtrudedMesh(m, 1)

x = SpatialCoordinate(mesh)

deg = 1
NewX = Function(VectorFunctionSpace(mesh, "CG", 1, dim=3))
NewX.interpolate(as_vector([pi*x[0],0,0]))

mesh = Mesh(Function(NewX))
x = SpatialCoordinate(mesh)

V = VectorFunctionSpace(mesh, "CG", deg)
Q = VectorFunctionSpace(mesh, "DG", deg-1)
A = FunctionSpace(mesh, "DG", deg-1)

# q, t, b, p, lambda, mu, alpha, gamma
W = V * V * V * Q * Q * Q * A * A

w = Function(W)
dw = TestFunction(W)


#initial guesses
# q, t, b, p, lambda, mu, alpha, gamma
q, t, b, p, ll, mu, alpha, beta = w.split()

s, _, _ = SpatialCoordinate(mesh)

def q_para(s):
    return as_vector([-sin(s), cos(s), 0])

def t_para(s):
    return as_vector([-cos(s), -sin(s), 0])


#boundary conditions
qstar0 = q_para(Constant(0))
qstar1 = q_para(Constant(1))
tstar0 = t_para(Constant(0))
tstar1 = t_para(Constant(1))
bstar0 = Constant(as_vector([0,0.0,1]))
bstar1 = Constant(as_vector([0,0.0,1]))

q.interpolate(q_para(s))
t.interpolate(t_para(s))
b.interpolate(as_vector([0, 0, 1]))
alpha.assign(1.0)
beta.assign(0.)

#splitting for equations
q, t, b, p, ll, mu, alpha, beta = split(w)

#F = alpha**2*(2+ gamma**2)
F = (2*alpha**2 + beta**2) # /alpha**2

S = (F +
     inner(ll, t.dx(0) - alpha*cross(b, t))
     + inner(mu, b.dx(0) - beta*cross(t, b))
     + inner(p, q.dx(0) - t)
)*dx

eqn = derivative(S, w, dw)

S1 = S + inner(p, p)*dx + inner(mu, mu)*dx + inner(ll, ll)*dx
S1 += inner(q, q)*dx + inner(t, t)*dx + inner(b, b)*dx
eqn1 = derivative(S1, w)
Jp = derivative(eqn1, w)


bcs = [DirichletBC(W.sub(0), qstar0, 1),
       DirichletBC(W.sub(0), qstar1, 2),
       DirichletBC(W.sub(1), tstar0, 1),
       DirichletBC(W.sub(1), tstar1, 2),
       DirichletBC(W.sub(2), bstar0, 1),
       DirichletBC(W.sub(2), bstar1, 2)]

solver_parameters = {'ksp_type':'gmres',
                     'ksp_monitor': None,
                     'mat_type':'aij',
                     'pc_type':'lu',
                     "pc_factor_mat_ordering_type": "rcm",
                     "pc_factor_shift_type": "nonzero",
                     'snes_monitor':None}

solve(eqn == 0, w, bcs = bcs, Jp=Jp,
      solver_parameters = solver_parameters)
