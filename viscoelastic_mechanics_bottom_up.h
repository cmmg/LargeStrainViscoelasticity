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

        SymmetricTensor<2, dim> epsilon; // Total strain
        SymmetricTensor<2, dim> epsilon_A; // Total elastic strain
        SymmetricTensor<2, dim> epsilon_B; // Accumulated viscous strain
        SymmetricTensor<2, dim> N_B_trial; // Unit tensor in direction of trial stress
        double x; // Variable necessary for integration point Newton Raphson iterations

    private:

        // Material parameters
        double K; // Bulk modulus
        double mu_0; // Shear modulus
        double gamma_dot_0; // Reference viscous stretching rate
        double sigma_0; // Resistance to viscous stretching
        double n; // Viscous flow rule exponent

        // Internal variable
        SymmetricTensor<4, dim> C_el; // Elastic tangent modulus
        SymmetricTensor<2, dim> sigma_d; // deviator(cauchy_stress)
        double sigma_d_norm; // norm(deviator(cauchy_stress))

        // Required functions
        double NR_function(double x, double sigma_d_norm_trial);
        double NR_function_derivative(double x, double sigma_d_norm_trial);

};

template <int dim>
Material<dim>::Material() {

    SymmetricTensor<2, dim> I = Physics::Elasticity::StandardTensors<dim>::I;

    deformation_gradient = I;
    cauchy_stress = 0;

}

template <int dim>
void Material<dim>::load_material_parameters(ParameterHandler &parameter_handler) {

    K    = parameter_handler.get_double("K");
    mu_0 = parameter_handler.get_double("mu0");

    gamma_dot_0 = parameter_handler.get_double("gammadot0");
    sigma_0     = parameter_handler.get_double("sigma0");
    n           = parameter_handler.get_double("n");

}

template <int dim>
void Material<dim>::compute_initial_tangent_modulus() {

    // Computes the small strain elastic tangent modulus for use in the first
    // iteration of the first time step.

    // Initialize the spatial tangent modulus
    SymmetricTensor<4, dim> S   = Physics::Elasticity::StandardTensors<dim>::S;
    SymmetricTensor<4, dim> IxI = Physics::Elasticity::StandardTensors<dim>::IxI;

    spatial_tangent_modulus = (K - 2.0 * mu_0 / 3.0) * IxI + 2 * mu_0 * S;

    C_el = spatial_tangent_modulus;

}

template <int dim>
double Material<dim>::NR_function(double x, double sigma_d_norm_trial) {

    double tmp;

    tmp = x
        + (2.0 * mu_0 * gamma_dot_0 * delta_t)
        * pow(sigma_d_norm_trial + x, n)
        * pow(sigma_0, -n);

    return tmp;

}

template <int dim>
double Material<dim>::NR_function_derivative(double x, double sigma_d_norm_trial) {

    double tmp;

    tmp = 1.0
        + (2.0 * mu_0 * n * gamma_dot_0 * delta_t)
        * pow(sigma_d_norm_trial + x, n - 1.0)
        * pow(sigma_0, -n);

    return tmp;
}

template <int dim>
void Material<dim>::perform_constitutive_update() {

    SymmetricTensor<2, dim> I = Physics::Elasticity::StandardTensors<dim>::I;

    Tensor<2, dim> F = deformation_gradient;

    // Total strain of the current, unresolved time step
    epsilon = 0.5 * symmetrize(F + transpose(F)) - I;

    // Calculate trial elastic strain state
    SymmetricTensor<2, dim> epsilon_A_trial = epsilon - epsilon_B;
    SymmetricTensor<2, dim> sigma_d_trial = 2 * mu_0 * deviator(epsilon_A_trial);

    if (sigma_d_trial.norm() == 0) {

        sigma_d = 0;

    } else {
        // N_B_trial is the direction of the trial elastic state
        N_B_trial = sigma_d_trial / sigma_d_trial.norm();
        // sigma_d_norm_trial is the magnitude of the trial elastic state
        double sigma_d_norm_trial = sigma_d_trial.norm();

        // x is the amount by which the norm of the trial deviatoric stress has to
        // be decreased to get the norm of the actual deviatoric stress state. It
        // is calculated in this implementation of viscoelasticity using Newton
        // Raphson iterations. The use of Newton-Raphson iterations in possible in
        // this case due to the use of a flow rule that allows a return mapping
        // algorithm.
        x = 0;

        double local_NR_function_initial    = NR_function(x, sigma_d_norm_trial);
        double local_NR_function_derivative = NR_function_derivative(x, sigma_d_norm_trial);
        double local_NR_function = local_NR_function_initial;

        unsigned int local_iterations = 0;
        unsigned int max_local_iterations = 10;

        while (fabs(local_NR_function / local_NR_function_initial) > 1e-9) {

            x -= local_NR_function / local_NR_function_derivative;

            // When this condition was added, x + sigma_d_norm_trial was being
            // raised to a fractional power. Therefore, it is not correct to have
            // this sum be negative.
            while (x + sigma_d_norm_trial < 0) {
                x *= 0.5;
            } 

            local_NR_function            = NR_function(x, sigma_d_norm_trial);
            local_NR_function_derivative = NR_function_derivative(x, sigma_d_norm_trial);

            local_iterations++;

            if (local_iterations == max_local_iterations) {
                std::cout
                << "Too many iterations for integrating the constitutive"
                << " equations. Exiting."
                << std::endl;

                exit(0);
            }
        }

        sigma_d_norm = sigma_d_norm_trial + x;

        sigma_d = sigma_d_norm * N_B_trial;
    }

    cauchy_stress = sigma_d + K * trace(epsilon) * I;

    epsilon_A = sigma_d / (2.0 * mu_0) + (1.0/3.0) * trace(epsilon) * I;

    epsilon_B = epsilon - epsilon_A;

}

template <int dim>
void Material<dim>::compute_spatial_tangent_modulus() {

    SymmetricTensor<4, dim> S   = Physics::Elasticity::StandardTensors<dim>::S;
    SymmetricTensor<4, dim> IxI = Physics::Elasticity::StandardTensors<dim>::IxI;
    SymmetricTensor<4, dim> P   = S - (1.0/3.0) * IxI;

    SymmetricTensor<4, dim> L_3 = pow(sigma_d_norm, n - 1.0) * S
                                + (n - 1.0) * pow(sigma_d_norm, n - 3.0)
                                * outer_product(sigma_d, sigma_d);

    SymmetricTensor<4, dim> L_2 = gamma_dot_0 * pow(sigma_0, -n) * L_3 * P;

    spatial_tangent_modulus = invert(S + C_el * L_2 * delta_t) * C_el;

}
