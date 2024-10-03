double inverse_Langevin(double y) {

    double a = 36/35;
    double b = 33/35;

    return y * (3 - a * pow(y, 2)) / (1 - b * pow(y, 2));

}

double d_inverse_Langevin_dy(double y) {
    // Derivative of the approximation of the inverse Langevin function with
    // respect to its argument

    double a = 36/35;
    double b = 33/35;

    double numerator   = a * b * pow(y, 4) + 3 * (b - a) * pow(y, 2) + 3;
    double denominator = pow(b * y * y - 1, 2);

    return numerator / denominator;
}

template <int dim>
SymmetricTensor<2, dim> HenckyStrain(Tensor<2, dim> F) {

    SymmetricTensor<2, dim> B = symmetrize(F * transpose(F));

    auto eigen_solution = eigenvectors(B);

    double eigenvalue_1 = eigen_solution[0].first;
    double eigenvalue_2 = eigen_solution[1].first;
    double eigenvalue_3 = eigen_solution[2].first;

    Tensor<1, dim> eigenvector_1 = eigen_solution[0].second;
    Tensor<1, dim> eigenvector_2 = eigen_solution[1].second;
    Tensor<1, dim> eigenvector_3 = eigen_solution[2].second;

    // Automatically initialized to when created
    SymmetricTensor<2, dim> E; 

    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            E[i][j] += 
                log(sqrt(eigenvalue_1)) * eigenvector_1[i] * eigenvector_1[j]
              + log(sqrt(eigenvalue_2)) * eigenvector_2[i] * eigenvector_2[j]
              + log(sqrt(eigenvalue_3)) * eigenvector_3[i] * eigenvector_3[j];
        }
    }

    return E;

}

template <int dim>
SymmetricTensor<4, dim> compute_tangent_modulus(
    Tensor<2, dim> F_A // Elastic part of the deformation gradient
) {

    SymmetricTensor<2, dim> I = Physics::Elasticity::StandardTensors<dim>::I;

    SymmetricTensor<4, dim> 
    C,     // Tangent modulus in reference coordinates
    C_vol, // Volumetric part of the referential tangent modulus
    C_dev, // Deviatoric part of the referential tangent modulus
    Jc;    // Tangent modulus in spatial coordinates

    // Calculation for C_dev starts

    double J = determinant(F_A);
    SymmetricTensor<2, dim> C_A = symmetrize(transpose(F_A) * F_A);
    SymmetricTensor<2, dim> C_bar_A = pow(J, -2/3) * C_A;

    SymmetricTensor<2, dim> dJ_dC     = Physics::Elasticity::StandardTensors<dim>::ddet_F_dC(F_A);
    SymmetricTensor<4, dim> dC_inv_dC = Physics::Elasticity::StandardTensors<dim>::dC_inv_dC(F_A);
    SymmetricTensor<4, dim> dC_bar_dC = Physics::Elasticity::StandardTensors<dim>::Dev_P(F_A);

    double lambda = sqrt(trace(C_bar_A)/3);

    SymmetricTensor<2, dim> dlambda_dC = (dC_bar_dC * I) / (6 * lambda);

    double y = lambda / lambda_L;
    double H = d_inverse_Langevin_dy(y) / y - inverse_Langevin(y) / (y*y);

    // Pieces that add up to C_dev
    SymmetricTensor<4, dim> C_dev_1, C_dev_2, C_dev_21, C_dev_22, C_dev_23;

    C_dev_1  = (mu_0 / lambda_L) * H 
             * outer_product(
               pow(J, -2/3) * I - pow(lambda, 2) * invert(C_A)
               ,
               dlambda_dC);

    C_dev_21 = outer_product(I, -(2/3) * pow(J, -5/3) * dJ_dC);
    C_dev_22 = outer_product(-2 * lambda * invert(C_A), dlambda_dC);
    C_dev_23 = -pow(lambda, 2) * dC_inv_dC;
    C_dev_2  = (mu_0 * inverse_Langevin(y) / y)
             * (C_dev_21 + C_dev_22 + C_dev_23);

    C_dev = C_dev_1 + C_dev_2;

    // Calculation for C_dev ends
    
    // Calculation for C_vol starts

    double G     = J * log((J - f_1)/(1 - f_1));
    double dG_dJ = log((J - f_1)/(1 - f_1)) + J / (J - f_1);

    SymmetricTensor<4, dim> C_vol_1, C_vol_2;

    C_vol_1 = K * dG_dJ * outer_product(invert(C_A), dJ_dC);

    C_vol_2 = K * G * dC_inv_dC;

    C_vol = C_vol_1 + C_vol_2;

    // Calculation for C_vol ends

    C = 2 * (C_vol + C_dev); // Material tangent = 2 * (dS/dC)

    Jc = J * Physics::Transformations::Contravariant::push_forward(C, F_A);

    return Jc;

}
