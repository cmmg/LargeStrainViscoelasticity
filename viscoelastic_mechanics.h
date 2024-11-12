template <int dim>
class Material {

    public:

        Material();

        void load_material_parameters(ParameterHandler &);
        void perform_constitutive_update();
        void compute_spatial_tangent_modulus();
        void compute_initial_tangent_modulus();

        Tensor<2, dim>          deformation_gradient; 
        SymmetricTensor<2, dim> kirchhoff_stress;
        SymmetricTensor<4, dim> spatial_tangent_modulus;

        double delta_t;

    private:

        // History variables
        Tensor<2, dim> F_B;
        Tensor<2, dim> F_D;

        // Material parameters
        double K; // Bulk modulus
        double f_1; // Volume fraction of moisture in brain tissue
        double mu_0; // Shear modulus
        double lambda_L; // Maximum stretch
        double sigma_0; // Strength parameter for viscous stretch rate
        double n; // Exponent for viscous flow rule
        double G_0; // Elastic modulus for element C
        double G_infinity; // Elastic modulus for element E
        double eta; // Viscosity for element D
        double gamma_dot_0; // Dimensionless scaling constant
        double alpha; // For removing singularity in flow rule

        // Material parameters
        /*double K = 800.0; // Bulk modulus*/
        /*double f_1 = 0.8; // Volume fraction of moisture in brain tissue*/
        /*double mu_0 = 20.0; // Shear modulus*/
        /*double lambda_L = 1.09; // Maximum stretch*/
        /*double sigma_0 = 40; // Strength parameter for viscous stretch rate*/
        /*double n = 2; // Exponent for viscous flow rule*/
        /*double G_0 = 4500.0; // Elastic modulus for element C*/
        /*double G_infinity = 600.0; // Elastic modulus for element E*/
        /*double eta = 60000.0; // Viscosity for element D*/
        /*double gamma_dot_0 = 1e-4; // Dimensionless scaling constant*/
        /*double alpha = 0.005; // For removing singularity in flow rule*/

        double inverse_Langevin(double y);
        double d_inverse_Langevin_dy(double y);

        SymmetricTensor<2, dim> compute_deviatoric_stress(SymmetricTensor<2, dim> B_A_bar);
        SymmetricTensor<2, dim> hencky_strain(Tensor<2, dim> F);
        SymmetricTensor<2, dim> square_root(SymmetricTensor<2, dim> B);
        SymmetricTensor<2, dim> multiply_symmetric_tensors(SymmetricTensor<2, dim> A, SymmetricTensor<2, dim> B);
};

template <int dim>
Material<dim>::Material() {

    SymmetricTensor<2, dim> I = Physics::Elasticity::StandardTensors<dim>::I;

    deformation_gradient = I;
    kirchhoff_stress = 0;

    // Initialize history variables;
    F_B = I;
    F_D = I;

}

template <int dim>
void Material<dim>::compute_initial_tangent_modulus() {

    // Computes the small strain elastic tangent modulus for use in the first
    // iteration of the first time step.

    // Initialize the spatial tangent modulus
    SymmetricTensor<4, dim> S   = Physics::Elasticity::StandardTensors<dim>::S;
    SymmetricTensor<4, dim> IxI = Physics::Elasticity::StandardTensors<dim>::IxI;

    double K_s  = K / (1 - f_1);
    double mu_s = mu_0 * lambda_L * inverse_Langevin(1/lambda_L);

    spatial_tangent_modulus = (K_s - 2.0 * mu_s / 3.0) * IxI + 2 * mu_s * S;

}

template <int dim>
SymmetricTensor<2, dim> Material<dim>::hencky_strain(Tensor<2, dim> F) {

    SymmetricTensor<2, dim> B = symmetrize(F * transpose(F));

    auto eigen_solution = eigenvectors(B, SymmetricTensorEigenvectorMethod::jacobi);

    double eigenvalue_1 = eigen_solution[0].first;
    double eigenvalue_2 = eigen_solution[1].first;
    double eigenvalue_3 = eigen_solution[2].first;

    Tensor<1, dim> eigenvector_1 = eigen_solution[0].second;
    Tensor<1, dim> eigenvector_2 = eigen_solution[1].second;
    Tensor<1, dim> eigenvector_3 = eigen_solution[2].second;

    // Automatically initialized to zero when created
    SymmetricTensor<2, dim> E; 

    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            E[i][j] = 
                log(sqrt(eigenvalue_1)) * eigenvector_1[i] * eigenvector_1[j]
              + log(sqrt(eigenvalue_2)) * eigenvector_2[i] * eigenvector_2[j]
              + log(sqrt(eigenvalue_3)) * eigenvector_3[i] * eigenvector_3[j];
        }
    }

    return E;

}

template <int dim>
SymmetricTensor<2, dim> Material<dim>::square_root(SymmetricTensor<2, dim> B) {

    auto eigen_solution = eigenvectors(B, SymmetricTensorEigenvectorMethod::jacobi);

    double eigenvalue_1 = eigen_solution[0].first;
    double eigenvalue_2 = eigen_solution[1].first;
    double eigenvalue_3 = eigen_solution[2].first;

    Tensor<1, dim> eigenvector_1 = eigen_solution[0].second;
    Tensor<1, dim> eigenvector_2 = eigen_solution[1].second;
    Tensor<1, dim> eigenvector_3 = eigen_solution[2].second;

    // Automatically initialized to zero when created
    SymmetricTensor<2, dim> V; 

    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            V[i][j] = 
                sqrt(eigenvalue_1) * eigenvector_1[i] * eigenvector_1[j]
              + sqrt(eigenvalue_2) * eigenvector_2[i] * eigenvector_2[j]
              + sqrt(eigenvalue_3) * eigenvector_3[i] * eigenvector_3[j];
        }
    }

    return V;
}

template <int dim>
SymmetricTensor<2, dim> Material<dim>::multiply_symmetric_tensors(SymmetricTensor<2, dim> A, SymmetricTensor<2, dim> B) {

    SymmetricTensor<2, dim> tmp;

    for(unsigned int i = 0; i < dim; ++i) {
        for(unsigned int j = 0; j < dim; ++j) {
            for(unsigned int k = 0; k < dim; ++k) {
                tmp[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return tmp;

}

template <int dim>
double Material<dim>::inverse_Langevin(double y) {

    double a = 36.0/35.0;
    double b = 33.0/35.0;

    return y * (3 - a * pow(y, 2)) / (1 - b * pow(y, 2));

}

template <int dim>
double Material<dim>::d_inverse_Langevin_dy(double y) {
    // Derivative of the approximation of the inverse Langevin function with
    // respect to its argument

    double a = 36.0/35.0;
    double b = 33.0/35.0;

    double numerator   = a * b * pow(y, 4.0) + 3.0 * (b - a) * pow(y, 2.0) + 3.0;
    double denominator = pow(b * y * y - 1.0, 2.0);

    return numerator / denominator;
}

template <int dim>
SymmetricTensor<2, dim> Material<dim>::compute_deviatoric_stress(SymmetricTensor<2, dim> B_A_bar) {

    double J = determinant(deformation_gradient);

    double lambda = sqrt(trace(B_A_bar) / 3.0);

    SymmetricTensor<2, dim> T_d = (mu_0 / J)
                                * (lambda_L / lambda)
                                * inverse_Langevin(lambda / lambda_L)
                                * deviator(B_A_bar); 

    return T_d;

}

template <int dim>
void Material<dim>::load_material_parameters(ParameterHandler &parameter_handler) {

    K           = parameter_handler.get_double("K");
    f_1         = parameter_handler.get_double("f1");
    mu_0        = parameter_handler.get_double("mu0");
    lambda_L    = parameter_handler.get_double("lambdaL");
    sigma_0     = parameter_handler.get_double("sigma0");
    n           = parameter_handler.get_double("n");
    G_0         = parameter_handler.get_double("G0");
    G_infinity  = parameter_handler.get_double("Ginfinity");
    eta         = parameter_handler.get_double("eta");
    gamma_dot_0 = parameter_handler.get_double("gammadot0");
    alpha       = parameter_handler.get_double("alpha");
        
}

template <int dim>
void Material<dim>::perform_constitutive_update() {
    
    // This function performs the constitutive update for this material class.
    // It assumes that whatever piece of code has called this function, has
    // already set the deformation graident to its latest values before calling
    // this function.

    SymmetricTensor<2, dim> I = Physics::Elasticity::StandardTensors<dim>::I;

    // Total deformation gradient of the unresolved time step. 
    Tensor<2, dim> F = deformation_gradient;

    // At this stage, F is the deformation gradient of the new time step. F_B
    // and F_D are the viscous deformation gradients for the previous time
    // step, waiting to be updated.

    // Volumetric part of the response is completely elastic
    double J = determinant(F);
    SymmetricTensor<2, dim> T_h = K * log((J - f_1)/(1 - f_1)) * I;

    SymmetricTensor<2, dim> T_d; // Deviatoric part of Cauchy stress
    SymmetricTensor<2, dim> T_A; // Cauchy stress

    // Before the loop starts, we set up certain variables useful for
    // constitutive update

    Tensor<2, dim> F_B_trial = F_B;
    Tensor<2, dim> F_B_old = F_B_trial;
    Tensor<2, dim> F_B_new;

    Tensor<2, dim> F_A = F * invert(F_B_trial);

    SymmetricTensor<2, dim> B_A_bar_trial = pow(J, -2.0/3.0) * symmetrize(F_A * transpose(F_A));
    SymmetricTensor<2, dim> B_A_bar = B_A_bar_trial;

    Tensor<2, dim> F_D_trial = F_D;
    Tensor<2, dim> F_D_old = F_D_trial;
    Tensor<2, dim> F_D_new;
    Tensor<2, dim> F_D_dot;

    Tensor<2, dim> F_C; // Strain in spring element C
    Tensor<2, dim> F_E; // Strain in spring element E

    SymmetricTensor<2, dim> E_C; // Hencky Strain in spring element C
    SymmetricTensor<2, dim> E_E; // Hencky Strain in spring element E

    SymmetricTensor<2, dim> S_C; // Stress in spring element C
    SymmetricTensor<2, dim> S_D; // Stress in viscous element D
    SymmetricTensor<2, dim> S_E; // Stress in spring element E

    SymmetricTensor<2, dim> T_B; // Stress in viscous element B
    SymmetricTensor<2, dim> T_B_dev; // Deviatoric part of stress in viscous element B
    SymmetricTensor<2, dim> n_B; // Unit vector in the direction of T_B_dev
    SymmetricTensor<2, dim> C_B_inv; // Viscous strain in element B

    SymmetricTensor<2, dim> D_tilde_D; // Stretch rate in viscous element D

    double f_R, gamma_dot_B;

    while (true) {

        // First, stresses in the model are calculated using the old estimates
        // of F_B and F_D

        F_A = F * invert(F_B_old);

        F_C = F_B_old * invert(F_D_old);
        F_E = F_D_old;

        E_C = hencky_strain(F_C);
        E_E = hencky_strain(F_E);

        S_C = G_0 * E_C;
        S_E = G_infinity * E_E;

        S_D = S_C - symmetrize(F_C * S_E * transpose(F_C));

        T_d = compute_deviatoric_stress(B_A_bar);
        T_A = T_h + T_d;
        T_B = T_A - (1/J) * symmetrize(F_A * S_C * transpose(F_A));

        // This completes the calculation of stresses in the model using the
        // previous estimates of the history variables F_B and F_D. This is
        // followed by the use of these stresses in the flow rules of the
        // elements F_B and F_D to get new estimates of F_B and F_D.

        C_B_inv = invert(symmetrize(transpose(F_B_old) * F_B_old));

        f_R = pow(alpha, 2.0)
            / pow(alpha + sqrt(3/trace(C_B_inv)) - 1.0, 2.0);

        T_B_dev = deviator(T_B);

        if (T_B_dev.norm() == 0) {

            F_B = F_B_trial;
            F_D = F_D_trial;

            break;

        }

        gamma_dot_B = gamma_dot_0
                    * f_R
                    * pow(T_B_dev.norm() / sigma_0, n);

        n_B = T_B_dev / T_B_dev.norm();

        B_A_bar = B_A_bar_trial - gamma_dot_B * n_B * delta_t;
        B_A_bar = B_A_bar * pow(determinant(B_A_bar), -1.0/3.0);

        F_A = square_root(B_A_bar) * pow(J, 1.0/3.0); 

        F_B_new = invert(F_A) * F;
        // To make sure that the viscous deformation is isochoric
        F_B_new = pow(determinant(F_B_new), -1.0/3.0) * F_B_new;

        D_tilde_D = S_D / eta;

        F_D_dot = invert(F_C) * D_tilde_D * F_B_new;

        F_D_new = F_D_trial + F_D_dot * delta_t;
        // To make sure that the viscous deformation is isochoric
        F_D_new = pow(determinant(F_D_new), -1.0/3.0) * F_D_new;

        if (
            (F_B_new - F_B_old).norm() < 1e-12
            and
            (F_D_new - F_D_old).norm() < 1e-12
        ) {

            F_B = F_B_new;
            F_D = F_D_new;

            break;

        } else {

            F_B_old = F_B_new;
            F_D_old = F_D_new;

        }

    } // End of constitutive update loop

    kirchhoff_stress = J * T_A;

}

template <int dim>
void Material<dim>::compute_spatial_tangent_modulus() {

    SymmetricTensor<2, dim> I = Physics::Elasticity::StandardTensors<dim>::I;

    // Fourth order tensors involved in the construction of the tangent modulus
    SymmetricTensor<4, dim> S = Physics::Elasticity::StandardTensors<dim>::S;

    Tensor<2, dim> F = deformation_gradient;

    SymmetricTensor<2, dim> dJ_dC = 
    Physics::Elasticity::StandardTensors<dim>::ddet_F_dC(F);

    SymmetricTensor<4, dim> dC_inv_dC = 
    Physics::Elasticity::StandardTensors<dim>::dC_inv_dC(F);

    double J = determinant(F);
    Tensor<2, dim> F_inv = invert(F);
    Tensor<2, dim> F_A = F * invert(F_B);
    SymmetricTensor<2, dim> B_A_bar = pow(J, -2.0/3.0) 
                                    * symmetrize(F_A * transpose(F_A));

    SymmetricTensor<2, dim> C       = symmetrize(transpose(F) * F);
    SymmetricTensor<2, dim> C_inv   = invert(C);
    SymmetricTensor<2, dim> C_B     = symmetrize(transpose(F_B) * F_B);
    SymmetricTensor<2, dim> C_B_inv = invert(C_B);

    double lambda = sqrt(trace(B_A_bar) / 3.0);
    double y      = lambda / lambda_L;

    double H    = mu_0 * inverse_Langevin(y) / y;
    double dHdl = (mu_0 / lambda_L)
                * (d_inverse_Langevin_dy(y)/y - inverse_Langevin(y)/(y*y));

    SymmetricTensor<4, dim> A_1, A_2, A_3, A_4, A_5;

    A_1 = (1/(6 * lambda))
        * outer_product(
          (2.0/3.0) * (dHdl * lambda * lambda + 2 * H * lambda) * (C_B_inv * C) * pow(J, -5.0/3.0) * C_inv
        - (2.0/3.0) * dHdl * pow(J, -7.0/3.0) * (C_B_inv * C) * C_B_inv
          ,
          dJ_dC);

    A_2 = (1/(6 * lambda))
        * outer_product(
          dHdl * pow(J, -4.0/3.0) * C_B_inv - pow(J, -2.0/3.0) * (dHdl * lambda * lambda + 2 * H * lambda) * C_inv
          ,
          C);

    A_3 = (1/(6 * lambda))
        * outer_product(
          dHdl * pow(J, -4.0/3.0) * C_B_inv - pow(J, -2.0/3.0) * (dHdl * lambda * lambda + 2 * H * lambda) * C_inv
          ,
          C_B_inv);

    A_4 = A_1 + A_3
        - (2.0/3.0) * H * pow(J, -5.0/3.0) * outer_product(C_B_inv, dJ_dC) 
        - H * lambda * lambda * dC_inv_dC;

    A_5 = A_2 + H * pow(J, -2.0/3.0) * S;

    Tensor<2, dim> F_C = F_B * invert(F_D);

    SymmetricTensor<2, dim> E_C = hencky_strain(F_C);

    SymmetricTensor<2, dim> S_C = G_0 * E_C;

    SymmetricTensor<2, dim> T_h = K * log((J - f_1)/(1 - f_1)) * I;
    SymmetricTensor<2, dim> T_d = compute_deviatoric_stress(B_A_bar);
    SymmetricTensor<2, dim> T_A = T_h + T_d;
    SymmetricTensor<2, dim> T_B = T_A - (1/J) * symmetrize(F_A * S_C * transpose(F_A));

    SymmetricTensor<2, dim> T_B_dev = deviator(T_B);
    SymmetricTensor<2, dim> S_B = symmetrize(F_inv * T_B_dev * transpose(F_inv));

    SymmetricTensor<2, dim> C_SB    = multiply_symmetric_tensors(C, S_B);
    SymmetricTensor<2, dim> SB_C_SB = multiply_symmetric_tensors(S_B, C_SB);

    SymmetricTensor<2, dim> SB_C   = multiply_symmetric_tensors(S_B, C);
    SymmetricTensor<2, dim> C_SB_C = multiply_symmetric_tensors(C, SB_C);

    double T_B_mod = T_B_dev.norm();

    double g_1 = - gamma_dot_0 * pow(alpha, 2.0) / pow(sigma_0, n);
    double g_2 = alpha + sqrt(3/trace(C_B_inv)) - 1.0;

    SymmetricTensor<4, dim> B_1, B_2, B_3, B_4, B_5;

    B_1 = g_1 
        * sqrt(3) * pow(trace(C_B_inv), -1.5) * pow(T_B_mod, n-1) * pow(g_2, -3.0)
        * outer_product(S_B, I);

    B_2 = g_1
        * pow(J, 2.0) * (n - 1) * pow(T_B_mod, n-3) * pow(g_2, -2.0)
        * outer_product(S_B, SB_C_SB);

    B_3 = g_1
        * pow(J, 2.0) * (n - 1) * pow(T_B_mod, n-3) * pow(g_2, -2.0)
        * outer_product(S_B, C_SB_C);

    B_4 = g_1
        * pow(T_B_mod, n-1) * pow(g_2, -2.0)
        * S;

    B_5 = g_1
        * J * (C_SB * SB_C) * (n-1) * pow(T_B_mod, n-3) * pow(g_2, -2.0)
        * outer_product(S_B, dJ_dC);

    SymmetricTensor<4, dim> C_1 = invert(S - B_1 * delta_t) * (B_2 + B_5) * delta_t;
    SymmetricTensor<4, dim> C_2 = invert(S - B_1 * delta_t) * (B_3 + B_4) * delta_t;

    SymmetricTensor<4, dim> dS_d_dC;

    dS_d_dC = invert(S - A_5 * C_2) * (A_4 + A_5 * C_1);

    SymmetricTensor<4, dim> dS_h_dC;

    double fJ = K * J * log((J - f_1) / (1.0 - f_1));
    double dfJ_dJ = K * (log((J - f_1) / (1.0 - f_1)) + J/(J - f_1));

    dS_h_dC = fJ * dC_inv_dC + dfJ_dJ * outer_product(C_inv, dJ_dC);

    SymmetricTensor<4, dim> dS_dC = dS_d_dC + dS_h_dC;

    spatial_tangent_modulus = J * Physics::Transformations::Contravariant::push_forward(2 * dS_dC, F);

}
