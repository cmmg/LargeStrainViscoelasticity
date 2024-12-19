template <int dim>
class Material {

    public:

        Material();

        void load_material_parameters(ParameterHandler &);
        void compute_initial_tangent_modulus();

        void perform_constitutive_update();
        void compute_spatial_tangent_modulus();

        Tensor<2, dim>          deformation_gradient; 
        SymmetricTensor<2, dim> cauchy_stress;
        SymmetricTensor<4, dim> spatial_tangent_modulus;

        double delta_t;
        unsigned int integration_point_index;
        unsigned int cell_index;

        std::ofstream *text_output_file;

    private:

        // Material parameters
        double K; // Bulk modulus
        double mu; // Shear modulus
        double gamma_dot_0; // Reference viscous stretching rate
        double sigma_0; // Resistance to viscous stretching
        double m; // Viscous flow rule exponent

        Tensor<2, dim> F_A; // Total elastic strain
        Tensor<2, dim> F_B; // Accumulated viscous strain
        Tensor<2, dim> F_B_old; // Accumulated viscous strain previous time step
        SymmetricTensor<2, dim> b_A_bar_trial;
        SymmetricTensor<2, dim> n_trial; // Unit tensor in direction of trial stress
        double x; // Variable necessary for integration point Newton Raphson iterations

        // Internal variable
        SymmetricTensor<4, dim> C_el; // Elastic tangent modulus
        SymmetricTensor<2, dim> tau_d; // deviator(cauchy_stress)
        double tau_d_norm; // norm(deviator(cauchy_stress))
        double tau_d_norm_trial; // trial norm(deviator(cauchy_stress))

        // Required functions
        double NR_function(double x, double tau_d_norm_trial);
        double NR_function_derivative(double x, double tau_d_norm_trial);
        SymmetricTensor<2, dim> tensor_square_root(SymmetricTensor<2, dim> b);
        Tensor<2, dim> multiply_symmetric_tensors(SymmetricTensor<2, dim> A, SymmetricTensor<2, dim> B);

};

template <int dim>
Material<dim>::Material() {

    SymmetricTensor<2, dim> I = Physics::Elasticity::StandardTensors<dim>::I;

    deformation_gradient = I;
    cauchy_stress = 0;

    F_A = I;
    F_B = I;

}

template <int dim>
void Material<dim>::load_material_parameters(ParameterHandler &parameter_handler) {

    K    = parameter_handler.get_double("Bulk Modulus");
    mu = parameter_handler.get_double("Shear Modulus");

    gamma_dot_0 = parameter_handler.get_double("Dimensional Scaling Constant");
    sigma_0     = parameter_handler.get_double("Viscous Resistance");
    m           = parameter_handler.get_double("Strain Rate Exponent");

}

template <int dim>
void Material<dim>::compute_initial_tangent_modulus() {

    // Computes the small strain elastic tangent modulus for use in the first
    // iteration of the first time step.

    // Initialize the spatial tangent modulus
    SymmetricTensor<4, dim> S   = Physics::Elasticity::StandardTensors<dim>::S;
    SymmetricTensor<4, dim> IxI = Physics::Elasticity::StandardTensors<dim>::IxI;

    spatial_tangent_modulus = (K - 2.0 * mu / 3.0) * IxI + 2.0 * mu * S;

    C_el = spatial_tangent_modulus;

}

template <int dim>
double Material<dim>::NR_function(double x, double tau_d_norm_trial) {

    double tmp;

    tmp = x
        + (mu * gamma_dot_0 * delta_t)
        * pow(tau_d_norm_trial + x, m)
        * pow(sigma_0, -m);

    return tmp;

}

template <int dim>
double Material<dim>::NR_function_derivative(double x, double tau_d_norm_trial) {

    double tmp;

    tmp = 1.0
        + (mu * m * gamma_dot_0 * delta_t)
        * pow(tau_d_norm_trial + x, m - 1.0)
        * pow(sigma_0, -m);

    return tmp;
}

template <int dim>
SymmetricTensor<2, dim> Material<dim>::tensor_square_root(SymmetricTensor<2, dim> B) {

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
Tensor<2, dim> Material<dim>::multiply_symmetric_tensors(
    SymmetricTensor<2, dim> A,
    SymmetricTensor<2, dim> B) {

    Tensor<2, dim> tmp;

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
void Material<dim>::perform_constitutive_update() {

    SymmetricTensor<2, dim> I = Physics::Elasticity::StandardTensors<dim>::I;

    Tensor<2, dim> F = deformation_gradient;

    double J = determinant(F);

    /*SymmetricTensor<2, dim> b_bar = pow(J, -2.0/3.0) * symmetrize(F * transpose(F));*/
    /*cauchy_stress = K * log(J) * I + (mu/J) * deviator(b_bar);*/

    Tensor<2, dim> F_bar = pow(J, -1.0/3.0) * F;

    // Use internal variable from previous time step to compute frozen viscous state
    SymmetricTensor<2, dim> C_B_inv = invert(symmetrize(transpose(F_B) * F_B));

    // Trial elastic strain state
    b_A_bar_trial = symmetrize(F_bar * C_B_inv * transpose(F_bar));

    // Trial elastic stress state
    SymmetricTensor<2, dim> tau_d_trial = mu * deviator(b_A_bar_trial);

    if (tau_d_trial.norm() == 0) {

        tau_d = 0.0;
        tau_d_norm = 0.0;

    } else {
        // n_trial is the direction of the trial elastic state
        n_trial = tau_d_trial / tau_d_trial.norm();
        // tau_d_norm_trial is the magnitude of the trial elastic state
        tau_d_norm_trial = tau_d_trial.norm();

        // x is the amount by which the norm of the trial deviatoric stress has to
        // be decreased to get the norm of the actual deviatoric stress state. It
        // is calculated in this implementation of viscoelasticity using Newton
        // Raphson iterations. The use of Newton-Raphson iterations in possible in
        // this case due to the use of a flow rule that allows a return mapping
        // algorithm.
        x = 0;

        double local_NR_function_initial    = NR_function(x, tau_d_norm_trial);
        double local_NR_function_derivative = NR_function_derivative(x, tau_d_norm_trial);
        double local_NR_function = local_NR_function_initial;

        unsigned int local_iterations = 0;
        unsigned int max_local_iterations = 10;

        while (fabs(local_NR_function / local_NR_function_initial) > 1e-9) {

            x -= local_NR_function / local_NR_function_derivative;

            // When this condition was added, x + tau_d_norm_trial was being
            // raised to a fractional power. Therefore, it is not correct to have
            // this sum be negative.
            while (x + tau_d_norm_trial < 0) {
                x *= 0.5;
            } 

            local_NR_function            = NR_function(x, tau_d_norm_trial);
            local_NR_function_derivative = NR_function_derivative(x, tau_d_norm_trial);

            local_iterations++;

            if (local_iterations == max_local_iterations) {
                std::cout
                << "Too many iterations for integrating the constitutive"
                << " equations. Exiting."
                << std::endl;

                exit(0);
            }
        }

        tau_d_norm = tau_d_norm_trial + x;

        tau_d = tau_d_norm * n_trial;

    }

    SymmetricTensor<2, dim> b_A_bar = tau_d / mu + (1.0/3.0) * trace(b_A_bar_trial) * I;

    b_A_bar = pow(determinant(b_A_bar), -2.0/3.0) * b_A_bar;

    cauchy_stress = K * log(J) * I + tau_d / J;

    SymmetricTensor<2, dim> b_A = pow(J, 2.0/3.0) * b_A_bar;

    SymmetricTensor<2, dim> V_A = tensor_square_root(b_A);

    F_A = V_A;

    F_B_old = F_B;

    F_B = invert(F_A) * F;

}

template <int dim>
void Material<dim>::compute_spatial_tangent_modulus() {

    Tensor<2, dim> F = deformation_gradient;

    double J = determinant(F);

    SymmetricTensor<2, dim> C     = symmetrize(transpose(F) * F);
    SymmetricTensor<2, dim> C_inv = invert(C);

    SymmetricTensor<2, dim> C_B     = symmetrize(transpose(F_B_old) * F_B_old);
    SymmetricTensor<2, dim> C_B_inv = invert(C_B);

    SymmetricTensor<4, dim> dC_inv_dC = Physics::Elasticity::StandardTensors<dim>::dC_inv_dC(F);

    SymmetricTensor<2, dim> dtau_d_norm_trial_dC = - (1.0/3.0) * tau_d_norm_trial * C_inv
        + mu * Physics::Transformations::Contravariant::pull_back(symmetrize(multiply_symmetric_tensors(b_A_bar_trial, n_trial)), F);

    SymmetricTensor<2, dim> S_d = Physics::Transformations::Contravariant::pull_back(tau_d, F);

    double f1 = 1.0 + mu * m * gamma_dot_0 * delta_t * pow(tau_d_norm, m - 1.0) * pow(sigma_0, -m);

    SymmetricTensor<2, dim> dtau_d_norm_dC = dtau_d_norm_trial_dC / f1;

    double f2 = mu * gamma_dot_0 * delta_t * (m - 1.0) * pow(tau_d_norm, m - 2.0) * pow(sigma_0, -m);

    SymmetricTensor<2, dim> S1 = f2 * S_d;

    SymmetricTensor<4, dim> L1 = 0.5 * (outer_product(S1, dtau_d_norm_dC) + outer_product(dtau_d_norm_dC, S1));

    double f3 = 1.0 + mu * gamma_dot_0 * delta_t * pow(tau_d_norm, m - 1.0) * pow(sigma_0, -m);

    SymmetricTensor<4, dim> dS_trial_dC = outer_product(C_B_inv, C_inv)
                                        + outer_product(C_inv, C_B_inv)
                                        - (1.0/3.0) * (C_inv * C_B_inv) * outer_product(C_inv, C_inv)
                                        + (C_inv * C_B_inv) * dC_inv_dC;

    dS_trial_dC *= - mu * pow(J, -2.0/3.0) / 3.0;

    SymmetricTensor<4, dim> dS_d_dC = (dS_trial_dC - L1) / f3;

    double f_h     = K * J * log(J);
    double df_h_dJ = K * (J/2.0) * (log(J) + 1.0);

    SymmetricTensor<4, dim> dS_h_dC = f_h * dC_inv_dC
                                    + df_h_dJ * outer_product(C_inv, C_inv);

    SymmetricTensor<4, dim> dS_dC = dS_h_dC + dS_d_dC;

    spatial_tangent_modulus = (1.0/J) * Physics::Transformations::Contravariant::push_forward(2.0 * dS_dC, F);

    // ------------------------------------------------------------------------------
    
    // Commented out code for purely elastic tangent modulus
    SymmetricTensor<2, dim> I = Physics::Elasticity::StandardTensors<dim>::I;

    // Start computing deviatoric tangent modulus
    dS_d_dC = outer_product(I, C_inv) + outer_product(C_inv, I)
            - (1.0/3.0) * trace(C) * outer_product(C_inv, C_inv)
            + trace(C) * dC_inv_dC;

    dS_d_dC *= - mu * pow(J, -2.0/3.0) / 3.0;

    // End computing deviatoric tangent modulus

    dS_dC = dS_d_dC + dS_h_dC;

    spatial_tangent_modulus = (1.0/J) * Physics::Transformations::Contravariant::push_forward(2.0 * dS_dC, F);
}

    // Commented out code for purely elastic tangent modulus
    /*SymmetricTensor<2, dim> I = Physics::Elasticity::StandardTensors<dim>::I;*/
    /**/
    /*Tensor<2, dim> F = deformation_gradient;*/
    /**/
    /*double J = determinant(F);*/
    /**/
    /*SymmetricTensor<2, dim> C     = symmetrize(transpose(F) * F);*/
    /*SymmetricTensor<2, dim> C_inv = invert(C);*/
    /**/
    /*SymmetricTensor<4, dim> dC_inv_dC = Physics::Elasticity::StandardTensors<dim>::dC_inv_dC(F);*/
    /**/
    /*double f_h     = K * J * log(J);*/
    /*double df_h_dJ = K * (J/2.0) * (log(J) + 1.0);*/
    /**/
    /*SymmetricTensor<4, dim> dS_h_dC = f_h * dC_inv_dC*/
    /*                                + df_h_dJ * outer_product(C_inv, C_inv);*/
    /**/
    /*// Start computing deviatoric tangent modulus*/
    /*SymmetricTensor<4, dim> dS_d_dC = outer_product(I, C_inv)*/
    /*                                + outer_product(C_inv, I)*/
    /*                                - (1.0/3.0) * trace(C) * outer_product(C_inv, C_inv)*/
    /*                                + trace(C) * dC_inv_dC;*/
    /**/
    /*dS_d_dC *= - mu * pow(J, -2.0/3.0) / 3.0;*/
    /**/
    /*// End computing deviatoric tangent modulus*/
    /**/
    /*SymmetricTensor<4, dim> dS_dC = dS_d_dC + dS_h_dC;*/
    /**/
    /*spatial_tangent_modulus = (1.0/J) * Physics::Transformations::Contravariant::push_forward(2.0 * dS_dC, F);*/

