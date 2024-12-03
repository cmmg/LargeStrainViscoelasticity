template <int dim>
class Material {

    public:

        Material();

        void load_material_parameters(ParameterHandler &);
        void compute_initial_tangent_modulus();

        void perform_constitutive_update();
        void compute_spatial_tangent_modulus();

        Tensor<2, dim>          deformation_gradient; 
        SymmetricTensor<2, dim> kirchhoff_stress;
        SymmetricTensor<4, dim> spatial_tangent_modulus;

        double delta_t;
        unsigned int integration_point_index;
        unsigned int cell_index;

        std::ofstream *text_output_file;

        Tensor<2, dim> F_A; // Total elastic strain
        Tensor<2, dim> F_B; // Accumulated viscous strain
        SymmetricTensor<2, dim> n_B_trial; // Unit tensor in direction of trial stress
        double x; // Variable necessary for integration point Newton Raphson iterations

    private:

        // Material parameters
        double K; // Bulk modulus
        double mu_0; // Shear modulus
        double gamma_dot_0; // Reference viscous stretching rate
        double sigma_0; // Resistance to viscous stretching
        double m; // Viscous flow rule exponent

        // Internal variable
        SymmetricTensor<4, dim> C_el; // Elastic tangent modulus
        SymmetricTensor<2, dim> tau_d; // deviator(kirchhoff_stress)
        double tau_d_norm; // norm(deviator(kirchhoff_stress))

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
    kirchhoff_stress = 0;

    F_A = I;
    F_B = I;

}

template <int dim>
void Material<dim>::load_material_parameters(ParameterHandler &parameter_handler) {

    K    = parameter_handler.get_double("K");
    mu_0 = parameter_handler.get_double("mu0");

    gamma_dot_0 = parameter_handler.get_double("gammadot0");
    sigma_0     = parameter_handler.get_double("sigma0");
    m           = parameter_handler.get_double("m");

}

template <int dim>
void Material<dim>::compute_initial_tangent_modulus() {

    // Computes the small strain elastic tangent modulus for use in the first
    // iteration of the first time step.

    // Initialize the spatial tangent modulus
    SymmetricTensor<4, dim> S   = Physics::Elasticity::StandardTensors<dim>::S;
    SymmetricTensor<4, dim> IxI = Physics::Elasticity::StandardTensors<dim>::IxI;

    spatial_tangent_modulus = (K - 2.0 * mu_0 / 3.0) * IxI + 2.0 * mu_0 * S;

    C_el = spatial_tangent_modulus;

}

template <int dim>
double Material<dim>::NR_function(double x, double tau_d_norm_trial) {

    double tmp;

    tmp = x
        + (mu_0 * gamma_dot_0 * delta_t)
        * pow(tau_d_norm_trial + x, m)
        * pow(sigma_0, -m);

    return tmp;

}

template <int dim>
double Material<dim>::NR_function_derivative(double x, double tau_d_norm_trial) {

    double tmp;

    tmp = 1.0
        + (mu_0 * m * gamma_dot_0 * delta_t)
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

    SymmetricTensor<2, dim> b_bar = pow(J, -2.0/3.0) * symmetrize(F * transpose(F));

    kirchhoff_stress = K * J * log(J) * I + mu_0 * deviator(b_bar);
    kirchhoff_stress = kirchhoff_stress / J;

    /*Tensor<2, dim> F_bar = pow(J, -2.0/3.0) * F;*/
    /**/
    /*// Use internal variable from previous time step to compute frozen viscous state*/
    /*SymmetricTensor<2, dim> C_B_inv = invert(symmetrize(transpose(F_B) * F_B));*/
    /**/
    /*// Trial elastic strain state*/
    /*SymmetricTensor<2, dim> b_A_bar_trial = symmetrize(F_bar * C_B_inv * transpose(F_bar));*/
    /**/
    /*// Trial elastic stress state*/
    /*SymmetricTensor<2, dim> tau_d_trial = mu_0 * deviator(b_A_bar_trial);*/
    /**/
    /*if (tau_d_trial.norm() == 0) {*/
    /**/
    /*    tau_d = 0;*/
    /**/
    /*} else {*/
    /*    // n_B_trial is the direction of the trial elastic state*/
    /*    n_B_trial = tau_d_trial / tau_d_trial.norm();*/
    /*    // tau_d_norm_trial is the magnitude of the trial elastic state*/
    /*    double tau_d_norm_trial = tau_d_trial.norm();*/
    /**/
    /*    // x is the amount by which the norm of the trial deviatoric stress has to*/
    /*    // be decreased to get the norm of the actual deviatoric stress state. It*/
    /*    // is calculated in this implementation of viscoelasticity using Newton*/
    /*    // Raphson iterations. The use of Newton-Raphson iterations in possible in*/
    /*    // this case due to the use of a flow rule that allows a return mapping*/
    /*    // algorithm.*/
    /*    x = 0;*/
    /**/
    /*    double local_NR_function_initial    = NR_function(x, tau_d_norm_trial);*/
    /*    double local_NR_function_derivative = NR_function_derivative(x, tau_d_norm_trial);*/
    /*    double local_NR_function = local_NR_function_initial;*/
    /**/
    /*    unsigned int local_iterations = 0;*/
    /*    unsigned int max_local_iterations = 10;*/
    /**/
    /*    while (fabs(local_NR_function / local_NR_function_initial) > 1e-9) {*/
    /**/
    /*        x -= local_NR_function / local_NR_function_derivative;*/
    /**/
    /*        // When this condition was added, x + tau_d_norm_trial was being*/
    /*        // raised to a fractional power. Therefore, it is not correct to have*/
    /*        // this sum be negative.*/
    /*        while (x + tau_d_norm_trial < 0) {*/
    /*            x *= 0.5;*/
    /*        } */
    /**/
    /*        local_NR_function            = NR_function(x, tau_d_norm_trial);*/
    /*        local_NR_function_derivative = NR_function_derivative(x, tau_d_norm_trial);*/
    /**/
    /*        local_iterations++;*/
    /**/
    /*        if (local_iterations == max_local_iterations) {*/
    /*            std::cout*/
    /*            << "Too many iterations for integrating the constitutive"*/
    /*            << " equations. Exiting."*/
    /*            << std::endl;*/
    /**/
    /*            exit(0);*/
    /*        }*/
    /*    }*/
    /**/
    /*    tau_d_norm = tau_d_norm_trial + x;*/
    /**/
    /*    tau_d = tau_d_norm * n_B_trial;*/
    /**/
    /*}*/
    /**/
    /*SymmetricTensor<2, dim> b_A_bar = tau_d / mu_0 + (1.0/3.0) * trace(b_A_bar_trial) * I;*/
    /**/
    /*b_A_bar = pow(determinant(b_A_bar), -2.0/3.0) * b_A_bar;*/
    /**/
    /*kirchhoff_stress = K * log(J) * I + mu_0 * deviator(b_A_bar) / J;*/
    /**/
    /*SymmetricTensor<2, dim> b_A = pow(J, 2.0/3.0) * b_A_bar;*/
    /**/
    /*SymmetricTensor<2, dim> V_A = tensor_square_root(b_A);*/
    /**/
    /*F_A = V_A;*/
    /**/
    /*F_B = invert(F_A) * F;*/
    /**/
    /*if (integration_point_index == 1) {*/
    /*    *text_output_file */
    /*    << kirchhoff_stress[2][2] << " "*/
    /*    << F_B.norm() << " "*/
    /*    << x << " "*/
    /*    << std::endl;*/
    /*}*/

}

template <int dim>
void Material<dim>::compute_spatial_tangent_modulus() {
    
    SymmetricTensor<2, dim> I = Physics::Elasticity::StandardTensors<dim>::I;
    
    Tensor<2, dim> F = deformation_gradient;

    double J = determinant(F);

    SymmetricTensor<2, dim> C     = symmetrize(transpose(F) * F);
    SymmetricTensor<2, dim> C_inv = invert(C);

    SymmetricTensor<4, dim> dC_inv_dC = Physics::Elasticity::StandardTensors<dim>::dC_inv_dC(F);

    double f_h     = K * J * log(J);
    double df_h_dJ = K * (J/2.0) * (log(J) + 1.0);

    SymmetricTensor<4, dim> dS_h_dC = f_h * dC_inv_dC
                                    + df_h_dJ * outer_product(C_inv, C_inv);

    // Start computing deviatoric tangent modulus
    SymmetricTensor<4, dim> dS_d_dC = outer_product(I, C_inv)
                                    + outer_product(C_inv, I)
                                    - (1.0/3.0) * trace(C) * outer_product(C_inv, C_inv)
                                    + trace(C) * dC_inv_dC;

    dS_d_dC *= - mu_0 * pow(J, -2.0/3.0) / 3.0;

    /*// Auxiliary tensors for calculating the deviatoric tangent modulus*/
    /*SymmetricTensor<2, dim> C_B_inv = invert(symmetrize(transpose(F_B) * F_B));*/
    /*SymmetricTensor<4, dim> X_1, Y_1, X_2, Y_2;*/
    /**/
    /*X_1 = outer_product(C_B_inv, C_inv) */
    /*    - (1.0/3.0) * (C_B_inv * C_inv) * outer_product(C_inv, C_inv)*/
    /*    + (C_B_inv * C_inv) * dC_inv_dC*/
    /*    + outer_product(C_B_inv, C_inv); */
    /**/
    /*X_1 *= -(1.0/3.0) * mu_0 * pow(J, -2.0/3.0);*/
    /**/
    /*Y_1 = mu_0 * Physics::Elasticity::StandardTensors<dim>::Dev_P(F);	*/
    /**/
    /*double f_d = - gamma_dot_0 * delta_t * pow(tau_d_norm, m - 3.0) * pow(sigma_0, -m);*/
    /**/
    /*SymmetricTensor<2, dim> Sd = Physics::Transformations::Contravariant::pull_back(tau_d, F);	*/
    /**/
    /*Tensor<2, dim> C_Sd = multiply_symmetric_tensors(C, Sd);*/
    /*SymmetricTensor<2, dim> Sd_C_Sd = symmetrize(Sd * C_Sd);*/
    /**/
    /*Tensor<2, dim> Sd_C = multiply_symmetric_tensors(Sd, C);*/
    /*SymmetricTensor<2, dim> C_Sd_C = symmetrize(C * Sd_C);*/
    /**/
    /*X_2 = f_d * (m - 1.0) * outer_product(Sd, Sd_C_Sd);*/
    /**/
    /*Y_2 = f_d * ((m - 1.0) * outer_product(Sd, C_Sd_C) + pow(tau_d_norm, 2.0) * S);*/
    /**/
    /*dS_d_dC = invert(S - Y_1 * Y_2) * (X_1 + Y_1 * X_2);*/

    // End computing deviatoric tangent modulus

    SymmetricTensor<4, dim> dS_dC = dS_d_dC + dS_h_dC;

    spatial_tangent_modulus = (1.0/J) * Physics::Transformations::Contravariant::push_forward(2.0 * dS_dC, F);

}
