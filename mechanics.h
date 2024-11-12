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

    private:

        // Material parameters
        double K; // Bulk modulus
        double f_1; // Volume fraction of moisture in brain tissue
        double mu_0; // Shear modulus
        double lambda_L; // Maximum stretch

    private:

        double inverse_Langevin(double y);
        double d_inverse_Langevin_dy(double y);

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
    f_1         = parameter_handler.get_double("f1");
    mu_0        = parameter_handler.get_double("mu0");
    lambda_L    = parameter_handler.get_double("lambdaL");
        
}

template <int dim>
void Material<dim>::compute_initial_tangent_modulus() {

    // Computes the small strain elastic tangent modulus for use in the first
    // iteration of the first time step.

    // Initialize the spatial tangent modulus
    SymmetricTensor<4, dim> S   = Physics::Elasticity::StandardTensors<dim>::S;
    SymmetricTensor<4, dim> IxI = Physics::Elasticity::StandardTensors<dim>::IxI;

    double K_s  = K / (1.0 - f_1);
    double mu_s = mu_0 * lambda_L * inverse_Langevin(1/lambda_L);

    spatial_tangent_modulus = (K_s - 2.0 * mu_s / 3.0) * IxI + 2 * mu_s * S;

}

template <int dim>
double Material<dim>::inverse_Langevin(double y) {

    double a = 36.0/35.0;
    double b = 33.0/35.0;

    return y * (3.0 - a * pow(y, 2.0)) / (1.0 - b * pow(y, 2.0));

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
void Material<dim>::perform_constitutive_update() {

    SymmetricTensor<2, dim> I = Physics::Elasticity::StandardTensors<dim>::I;

    Tensor<2, dim> F = deformation_gradient;

    // Compute Volumetric stress
    double J = determinant(F);

    SymmetricTensor<2, dim> T_h = K * log((J - f_1)/(1 - f_1)) * I;

    // Compute deviatoric stress
    SymmetricTensor<2, dim> B = Physics::Elasticity::Kinematics::b(F);

    SymmetricTensor<2, dim> B_bar = pow(J, -2.0/3.0) * B;

    double lambda = sqrt(trace(B_bar) / 3.0);

    SymmetricTensor<2, dim> T_d = (mu_0 / J)
                                * (lambda_L / lambda)
                                * inverse_Langevin(lambda / lambda_L)
                                * deviator(B_bar); 

    kirchhoff_stress = J * (T_h + T_d);

}

template <int dim>
void Material<dim>::compute_spatial_tangent_modulus() {

    SymmetricTensor<2, dim> I = Physics::Elasticity::StandardTensors<dim>::I;

    Tensor<2, dim> F = deformation_gradient;

    double J = determinant(F);

    SymmetricTensor<2, dim> C     = Physics::Elasticity::Kinematics::C(F);
    SymmetricTensor<2, dim> C_inv = invert(C);

    SymmetricTensor<2, dim> B     = Physics::Elasticity::Kinematics::b(F);
    SymmetricTensor<2, dim> B_bar = pow(J, -2.0/3.0) * B;

    SymmetricTensor<2, dim> ddet_F_dC = Physics::Elasticity::StandardTensors<dim>::ddet_F_dC(F);
    SymmetricTensor<4, dim> dC_inv_dC = Physics::Elasticity::StandardTensors<dim>::dC_inv_dC(F);
    SymmetricTensor<4, dim> dC_bar_dC = Physics::Elasticity::StandardTensors<dim>::Dev_P(F);

    double H    = J * log((J - f_1)/(1 - f_1));
    double dHdJ = log((J - f_1)/(1 - f_1)) + J/(J - f_1);

    // Derivative of the hydrostatic part of the stress
    SymmetricTensor<4, dim> dS_h_dC;

    dS_h_dC = K * dHdJ * outer_product(C_inv, ddet_F_dC)
            + K * H * dC_inv_dC;

    double lambda = sqrt(trace(B_bar) / 3.0);

    double y = lambda / lambda_L;

    SymmetricTensor<2, dim> dlambda_dC = (I * dC_bar_dC) / (6 * lambda);

    // Derivative of the deviatoric part of the stress
    SymmetricTensor<4, dim> dS_d_dC;
    SymmetricTensor<4, dim> dS_d_dC_1;
    SymmetricTensor<4, dim> dS_d_dC_2;
    SymmetricTensor<4, dim> dS_d_dC_3;
    SymmetricTensor<4, dim> dS_d_dC_4;

    dS_d_dC_1 = (mu_0 / lambda_L)
              * (d_inverse_Langevin_dy(y)/y - inverse_Langevin(y)/(y*y))
              * outer_product(
                pow(J, -2.0/3.0) * I - pow(lambda, 2.0) * C_inv
                ,
                dlambda_dC);

    dS_d_dC_2 = - (2.0 / 3.0) * pow(J, -2.0/3.0) * outer_product(I, ddet_F_dC);

    dS_d_dC_3 = - 2.0 * lambda * outer_product(C_inv, dlambda_dC);

    dS_d_dC_4 = - pow(lambda, 2.0) * dC_inv_dC;

    dS_d_dC = dS_d_dC_1
            + (mu_0 * inverse_Langevin(y) / y) 
            * (dS_d_dC_2 + dS_d_dC_3 + dS_d_dC_4);

    SymmetricTensor<4, dim> dS_dC = dS_h_dC + dS_d_dC;

    spatial_tangent_modulus = J * Physics::Transformations::Contravariant::push_forward(2.0 * dS_dC, F);

}
