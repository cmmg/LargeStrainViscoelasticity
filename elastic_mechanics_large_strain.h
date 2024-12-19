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
        SymmetricTensor<2, dim> tau_d; // deviator(cauchy_stress)
        double tau_d_norm; // norm(deviator(cauchy_stress))

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
    mu_0 = parameter_handler.get_double("Shear Modulus");

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

    cauchy_stress = K * log(J) * I + (mu_0/J) * deviator(b_bar);

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

    // End computing deviatoric tangent modulus

    SymmetricTensor<4, dim> dS_dC = dS_d_dC + dS_h_dC;

    spatial_tangent_modulus = (1.0/J) * Physics::Transformations::Contravariant::push_forward(2.0 * dS_dC, F);

}
