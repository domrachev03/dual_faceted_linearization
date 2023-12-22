import numpy as np
import casadi as cs

# TODO:
# add checking for dimensions of P and Dynamics
# add options for solver

class EllipsoidalInvariantSet:
    def __init__(self, 
                 dynamics = None, 
                 P = None,
                 num_points = 100, 
                 verbose = True):
        self.num_points = num_points
        self.dim = None
        self._matrix_dim = None
        self._state_dim = None
        self._verbose = verbose
        self.dynamics = dynamics
        self.set_matrix(P)
        
        self.opti_problem = cs.Opti()

        # self.sphere_samples = self.sampled_sphere(self.dim, self.num_points)

    def set_dynamics(self, dynamics):
        self.dynamics = dynamics

    def set_matrix(self, P):
        self.P = P
        if self.P is None and self._verbose:
            print("P matrix is not set, please use set_matrix(P) with PD matrix as argument")
            self._ready_to_assemble = False

        else:
            evalues, evectors = np.linalg.eig(self.P)
            if not np.all(evalues > 0):
                self._ready_to_assemble = False
                self.inv_sqrt_P = None 
                raise ValueError("P matrix is not positive definite")
            else:
                self.P_inv_sqrt = evectors @ np.diag(1 / np.sqrt(evalues)) @ evectors.T
                self._ready_to_assemble = True

                # parameters
                self.dim = self.P.shape[0]
                self.inv_sqrt = self.opti_problem.parameter(*self.P.shape)
                self.P_param = self.opti_problem.parameter(*self.P.shape)
                self.opti_problem.set_value(self.P_param, self.P)
                self.opti_problem.set_value(self.inv_sqrt, self.P_inv_sqrt)

        # return self.P, self.inv_sqrt_P


    def set_sampling_dim(self, num_points):
        self.num_points = num_points

    def sampled_sphere(self, dim, num_points = 100):
        sphere_samples = []
        for _ in range(num_points):
            v = np.random.randn(dim)
            sphere_samples.append(v / np.linalg.norm(v))
        return sphere_samples

    def sampled_boundary(self, rho, inv_sqrt, sphere_samples):
        x_rho = []
        for sphere_sample in sphere_samples:
            x_rho.append(inv_sqrt @ sphere_sample * rho)
        return x_rho

    def assemble_problem(self):
        if not self._ready_to_assemble:
            raise ValueError("Problem is not ready to be assemble, set PD P matrix and dynamics first")
        
        # self.dim = self.P.shape[0]
        
        self.state = cs.MX.sym('state', self.dim)
        self.rho = self.opti_problem.variable(1)
        # parameters
        # self.inv_sqrt = self.opti_problem.parameter(*self.P.shape)
        # self.P_param = self.opti_problem.parameter(*self.P.shape)

        self.f = cs.vertcat(*self.dynamics(self.state))
        self._V = self.state.T @ self.P_param @ self.state
        self.lyapunov_candidate = cs.Function('lyapunov_candidate',
                                               [self.state, self.P_param],
                                               [self._V],
                                               ["state", "pd_matrix"], ["V"])
        
        self._V_jacobian = 2 * self.state.T @ self.P_param
        self.lyapunov_derivative = cs.Function('lyapunov_derivative',
                                               [self.state, self.P_param],
                                               [self._V_jacobian @ self.f],
                                               ["state", "pd_matrix"], ["dV"])
        
        self.sphere_samples = self.sampled_sphere(self.dim, self.num_points)
        boundary = self.sampled_boundary(self.rho, self.inv_sqrt, self.sphere_samples)
        
        for boundary_point in boundary:
            self.opti_problem.subject_to(self.lyapunov_derivative(boundary_point, self.P_param) <= 0)
        self.opti_problem.subject_to(self.rho >= 10e-7)
        self.opti_problem.minimize(-self.rho)
        self.opti_problem.solver('ipopt')

        return self.opti_problem

    def find(self):
        self.assemble_problem()
        self.solve()
        return self.rho_opt
        
    def solve(self):
        self.opti_problem.solve()
        rho_sqr_opt = self.opti_problem.value(self.rho)
        self.rho_opt = rho_sqr_opt ** 2
        return self.rho_opt
    
