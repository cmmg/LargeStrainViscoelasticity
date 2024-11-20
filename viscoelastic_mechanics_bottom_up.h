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

        std::ofstream *text_output_file;

    private:

        // Material parameters
        double K; // Bulk modulus
        double f_1; // Volume fraction of moisture in brain tissue
        double mu_0; // Shear modulus
        double lambda_L; // Maximum stretch

};

template <int dim>
Material<dim>::Material() {

    SymmetricTensor<2, dim> I = Physics::Elasticity::StandardTensors<dim>::I;

    deformation_gradient = I;
    kirchhoff_stress = 0;

}

template <int dim>
void Material<dim>::load_material_parameters(ParameterHandler &parameter_handler) {

    K           = parameter_handler.get_double("K");
    mu_0        = parameter_handler.get_double("mu0");
        
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

    double J = determinant(F);

    SymmetricTensor<2, dim> T_h = K * log(J) * I;

    SymmetricTensor<2, dim> B_bar = pow(J, -2.0/3.0)
                                  * symmetrize(F * transpose(F));
    
    SymmetricTensor<2, dim> T_d = mu_0 * deviator(B_bar);

    kirchhoff_stress = T_h + T_d;

}

template <int dim>
void Material<dim>::compute_spatial_tangent_modulus() {
    compute_initial_tangent_modulus();
}
