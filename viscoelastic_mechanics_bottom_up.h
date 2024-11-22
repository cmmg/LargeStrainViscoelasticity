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

        std::ofstream *text_output_file;

    private:

        // Material parameters
        double K; // Bulk modulus
        double f_1; // Volume fraction of moisture in brain tissue
        double mu_0; // Shear modulus
        double lambda_L; // Maximum stretch

        // Internal variables
        SymmetricTensor<2, dim> epsilon_A; // Total elastic strain
        SymmetricTensor<2, dim> epsilon_B; // Accumulated viscous strain

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

}

template <int dim>
void Material<dim>::compute_initial_tangent_modulus() {

    // Computes the small strain elastic tangent modulus for use in the first
    // iteration of the first time step.

    // Initialize the spatial tangent modulus
    SymmetricTensor<4, dim> S   = Physics::Elasticity::StandardTensors<dim>::S;
    SymmetricTensor<4, dim> IxI = Physics::Elasticity::StandardTensors<dim>::IxI;

    spatial_tangent_modulus = (K - 2.0 * mu_0 / 3.0) * IxI + 2 * mu_0 * S;

}

template <int dim>
void Material<dim>::perform_constitutive_update() {

    SymmetricTensor<2, dim> I = Physics::Elasticity::StandardTensors<dim>::I;

    Tensor<2, dim> F = deformation_gradient;

    SymmetricTensor<2, dim> epsilon = 0.5 * symmetrize(F + transpose(F)) - I;

    /*std::cout << "epsilon = " << epsilon << std::endl;*/

    cauchy_stress = (K - 2.0 * mu_0 / 3.0) * trace(epsilon) * I
                  + 2 * mu_0 * epsilon;

}

template <int dim>
void Material<dim>::compute_spatial_tangent_modulus() {
    compute_initial_tangent_modulus();
}
