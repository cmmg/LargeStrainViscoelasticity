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
        SymmetricTensor<2, dim> n_trial; // Unit tensor in direction of trial deviatoric stress

        double gamma;
        double delta_gamma;
        double alpha;
        double pressure;

        SymmetricTensor<2, dim> tensor_square_root(SymmetricTensor<2, dim> B);
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

    double I_bar  = (1.0/3.0) * trace(b_e_bar_trial);
    double mu_bar = I_bar * mu;

    SymmetricTensor<2, dim> tau_d_trial = mu * deviator(b_e_bar_trial);

    n_trial = tau_d_trial / tau_d_trial.norm();

    double f_trial = tau_d_trial.norm() - sqrt(2.0/3.0) * (H * alpha + sigma_y);

    if (f_trial <= 0) { // Elastic step

        delta_gamma = 0.0;

    } else { // Plastic step

        delta_gamma = (f_trial / 2.0) / (mu_bar + H / 3.0);

    }

    gamma += delta_gamma;
    alpha += sqrt(2.0/3.0) * delta_gamma;

    SymmetricTensor<2, dim> tau_d = tau_d_trial - 2.0 * mu_bar * delta_gamma * n_trial;

    pressure = (K/2.0) * (J*J - 1.0) / J;

    kirchhoff_stress = J * pressure * I + tau_d;

    cauchy_stress = kirchhoff_stress / J;

    SymmetricTensor<2, dim> b_e_bar = tau_d / mu + (1.0/3.0) * trace(b_e_bar_trial) * I;
    SymmetricTensor<2, dim> b_e     = pow(J, 2.0/3.0) * b_e_bar;
    SymmetricTensor<2, dim> V_e     = tensor_square_root(b_e);

    F_e = V_e;
    F_p = invert(F_e) * F;

}

template <int dim>
void Material<dim>::compute_spatial_tangent_modulus() {

    spatial_tangent_modulus = 0.0;

}
