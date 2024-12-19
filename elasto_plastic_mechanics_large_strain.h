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

        // Standard tensors needed for constitutive update and tangent modulus calcs
        SymmetricTensor<2, dim> I   = Physics::Elasticity::StandardTensors<dim>::I;
        SymmetricTensor<4, dim> S   = Physics::Elasticity::StandardTensors<dim>::S;
        SymmetricTensor<4, dim> IxI = Physics::Elasticity::StandardTensors<dim>::IxI;

        // Material parameters
        double K; // Bulk modulus
        double mu; // Shear modulus
        double sigma_y; // Initial yield stress
        double H; // Hardening modulus

        // Internal variables
        Tensor<2, dim> F_e; // Elastic part of the deformation gradient
        Tensor<2, dim> F_p; // Plastic part of the deformation gradient
        SymmetricTensor<2, dim> kirchhoff_stress; 

        double gamma;
        double delta_gamma;
        double alpha;
        double pressure;

        double mu_bar;
        double tau_d_trial_norm;
        SymmetricTensor<2, dim> n_trial; // Unit tensor in direction of trial deviatoric stress

        SymmetricTensor<2, dim> tensor_square_root(SymmetricTensor<2, dim> B);
        Tensor<2, dim> multiply_symmetric_tensors(SymmetricTensor<2, dim> A,
                                                  SymmetricTensor<2, dim> B);

        double dU_dJ(double J);
        double dp_dJ(double J);
        SymmetricTensor<4, dim> compute_spatial_volumetric_tangent_modulus();
};

template <int dim>
void Material<dim>::load_material_parameters(ParameterHandler &parameter_handler) {

    mu      = parameter_handler.get_double("Shear Modulus");
    K       = parameter_handler.get_double("Bulk Modulus");
    sigma_y = parameter_handler.get_double("Yield Stress");
    H       = parameter_handler.get_double("Linear Hardening Modulus");

}

template <int dim>
Material<dim>::Material() {

    deformation_gradient = I;
    kirchhoff_stress = 0;

    F_e = I;
    F_p = I;

    gamma = 0.0;
    delta_gamma = 0.0;
    alpha = 0.0;

}


template <int dim>
void Material<dim>::compute_initial_tangent_modulus() {

    // Computes the small strain elastic tangent modulus for use in the first
    // iteration of the first time step.

    spatial_tangent_modulus = (K - 2.0 * mu / 3.0) * IxI + 2.0 * mu * S;

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
void Material<dim>::perform_constitutive_update() {

    // Total deformation gradient of the unresolved time step. 
    Tensor<2, dim> F = deformation_gradient;

    double J = determinant(F);

    Tensor<2, dim> F_bar = pow(J, -1.0/3.0) * F;

    SymmetricTensor<2, dim> C_p_inv = invert(symmetrize(transpose(F_p) * F_p));

    SymmetricTensor<2, dim> b_e_bar_trial = symmetrize(F_bar * C_p_inv * transpose(F_bar));

    mu_bar = (1.0/3.0) * trace(b_e_bar_trial) * mu;

    SymmetricTensor<2, dim> tau_d_trial = mu * deviator(b_e_bar_trial);

    tau_d_trial_norm = tau_d_trial.norm();

    if (tau_d_trial_norm == 0) {

        delta_gamma = 0.0;

    } else {

        n_trial = tau_d_trial / tau_d_trial.norm();

        double f_trial = tau_d_trial.norm() - sqrt(2.0/3.0) * (H * alpha + sigma_y);

        if (f_trial <= 0) { // Elastic step

            delta_gamma = 0.0;

        } else { // Plastic step

            delta_gamma = (f_trial / 2.0) / (mu_bar + H / 3.0);

        }

    }

    gamma += delta_gamma;
    alpha += sqrt(2.0/3.0) * delta_gamma;

    SymmetricTensor<2, dim> tau_d = tau_d_trial - 2.0 * mu_bar * delta_gamma * n_trial;

    pressure = dU_dJ(J);

    kirchhoff_stress = J * pressure * I + tau_d;

    cauchy_stress = kirchhoff_stress / J;

    SymmetricTensor<2, dim> b_e_bar = tau_d / mu + (1.0/3.0) * trace(b_e_bar_trial) * I;
    SymmetricTensor<2, dim> b_e     = pow(J, 2.0/3.0) * b_e_bar;
    SymmetricTensor<2, dim> V_e     = tensor_square_root(b_e);

    F_e = V_e;
    F_p = invert(F_e) * F;

}

template <int dim>
Tensor<2, dim> Material<dim>::multiply_symmetric_tensors(SymmetricTensor<2, dim> A,
                                                         SymmetricTensor<2, dim> B) {

    Tensor<2, dim> C;

    for(unsigned int i = 0; i < dim; ++i)
        for(unsigned int j = 0; j < dim; ++j)
            for(unsigned int k = 0; k < dim; ++k)
                C[i][j] += A[i][k] * B[k][j];

    return C;

}

template <int dim>
double Material<dim>::dU_dJ(double J) {

    // This function assumes that the internal energy of the material can be
    // divided into a volumetric and a deviatoric part. This function
    // calculates the derivative of the volumetric part of the internal energy
    // with respect to the determinant of the deformation gradient.

    double pressure;

    pressure = K * log(J);

    return pressure;

}

template <int dim>
double Material<dim>::dp_dJ(double J) {

    // This function assumes that the internal energy of the material can be
    // divided into a volumetric and a deviatoric part. This function
    // calculates the derivative of the pressure with respect to the
    // determinant of the deformation gradient.

    double dp_dJ;

    dp_dJ = K / J;

    return dp_dJ;

}

template <int dim>
SymmetricTensor<4, dim> Material<dim>::compute_spatial_volumetric_tangent_modulus() {

    double J = determinant(deformation_gradient);

    return (dU_dJ(J) + J * dp_dJ(J)) * IxI - 2 * dU_dJ(J) * S;

}

template <int dim>
void Material<dim>::compute_spatial_tangent_modulus() {

    if (delta_gamma > 0) {

        double beta_0 = 1.0 + H/(3.0 * mu_bar);

        double beta_1 = 2.0 * mu_bar * delta_gamma / tau_d_trial_norm;

        double beta_2 = (2.0 / 3.0)
                      * (1.0 - 1.0 / beta_1)
                      * (tau_d_trial_norm / mu_bar) * delta_gamma;

        double beta_3 = 1.0 / beta_0 - beta_1 - beta_2;

        double beta_4 = (1.0 / beta_0 - beta_1) * (tau_d_trial_norm / mu_bar);

        SymmetricTensor<4, dim> C_bar = 2.0 * mu_bar * (S - IxI / 3.0)
                                      - (2.0 / 3.0) * tau_d_trial_norm 
                                      * (outer_product(I, n_trial)
                                        +
                                        outer_product(n_trial, I));

        double J = determinant(deformation_gradient);

        double Ud  = dU_dJ(J);
        double Udd = dp_dJ(J);

        SymmetricTensor<4, dim> C = (Ud + J * Udd) * J * IxI - 2 * J * Ud * S + C_bar;

        SymmetricTensor<2, dim> n_squared = symmetrize(multiply_symmetric_tensors(n_trial, n_trial));

        SymmetricTensor<4, dim> D = 0.5 * (outer_product(n_trial, n_squared)
                                           +
                                           outer_product(n_squared, n_trial));

        spatial_tangent_modulus = C + beta_1 * C_bar 
                                - 2 * mu_bar * beta_3 * outer_product(n_trial, n_trial)
                                - 2 * mu_bar * beta_4 * D;

    } else {

        Tensor<2, dim> F = deformation_gradient;

        double J = determinant(F);

        SymmetricTensor<2, dim> C     = symmetrize(transpose(F) * F);
        SymmetricTensor<2, dim> C_inv = invert(C);

        SymmetricTensor<4, dim> dC_inv_dC = Physics::Elasticity::StandardTensors<dim>::dC_inv_dC(F);

        SymmetricTensor<4, dim> spatial_volumetric_tangent_modulus = compute_spatial_volumetric_tangent_modulus();

        // Start computing deviatoric tangent modulus
        SymmetricTensor<4, dim> dS_d_dC = outer_product(I, C_inv)
                                        + outer_product(C_inv, I)
                                        - (1.0/3.0) * trace(C) * outer_product(C_inv, C_inv)
                                        + trace(C) * dC_inv_dC;

        dS_d_dC *= - mu * pow(J, -2.0/3.0) / 3.0;

        // End computing deviatoric tangent modulus

        spatial_tangent_modulus = (1.0/J) * Physics::Transformations::Contravariant::push_forward(2.0 * dS_d_dC, F)
                                + spatial_volumetric_tangent_modulus;
    }

}
