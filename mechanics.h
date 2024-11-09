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

    auto eigen_solution = eigenvectors(B,
                                       SymmetricTensorEigenvectorMethod::jacobi);

    double eigenvalue_1 = eigen_solution[0].first;
    double eigenvalue_2 = eigen_solution[1].first;
    double eigenvalue_3 = eigen_solution[2].first;

    Tensor<1, dim> eigenvector_1 = eigen_solution[0].second;
    Tensor<1, dim> eigenvector_2 = eigen_solution[1].second;
    Tensor<1, dim> eigenvector_3 = eigen_solution[2].second;

    // Automatically initialized to zero when created
    SymmetricTensor<2, dim> E; 

    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            E[i][j] = 
                log(sqrt(eigenvalue_1)) * eigenvector_1[i] * eigenvector_1[j]
              + log(sqrt(eigenvalue_2)) * eigenvector_2[i] * eigenvector_2[j]
              + log(sqrt(eigenvalue_3)) * eigenvector_3[i] * eigenvector_3[j];
        }
    }

    return E;

}

template <int dim>
SymmetricTensor<4, dim> compute_dS_A_dC_A(
    Tensor<2, dim> F_A // Elastic part of the deformation gradient
) {

    // This function calculates the derivative of the 2nd Piola Kirchhoff
    // stress S_A with respect to the elastic strain metric C_A.
    //
    // S_A = J * F_A^-1 * T_A * F_A^-T
    // C_A = F_A^T * F_A
    //
    // Here, T_A is the Cauchy stress and F_A is the elastic part of the
    // deformation gradient

    SymmetricTensor<2, dim> I = Physics::Elasticity::StandardTensors<dim>::I;

    // Derivative of the 2nd PK stress with respect to the elasticity metric
    SymmetricTensor<4, dim> dS_A_dC_A; 

    SymmetricTensor<4, dim> dS_A_dC_A_vol; // Volumetric part of dS_A_dC_A
    SymmetricTensor<4, dim> dS_A_dC_A_dev; // Deviatoric part of dS_A_dC_A

    // Calculation for dS_A_dC_A_dev starts

    SymmetricTensor<2, dim> dJ_dC     = Physics::Elasticity::StandardTensors<dim>::ddet_F_dC(F_A);
    SymmetricTensor<4, dim> dC_inv_dC = Physics::Elasticity::StandardTensors<dim>::dC_inv_dC(F_A);
    SymmetricTensor<4, dim> dC_bar_dC = Physics::Elasticity::StandardTensors<dim>::Dev_P_T(F_A);

    double J = determinant(F_A);
    SymmetricTensor<2, dim> C_A = symmetrize(transpose(F_A) * F_A);
    SymmetricTensor<2, dim> C_bar_A = pow(J, -2/3) * C_A;

    double lambda = sqrt(trace(C_bar_A)/3);

    SymmetricTensor<2, dim> dlambda_dC = (dC_bar_dC * I) / (6 * lambda);

    double y = lambda / lambda_L;
    double H = d_inverse_Langevin_dy(y) / y - inverse_Langevin(y) / (y*y);

    // Pieces that add up to dS_A_dC_A_dev
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

    dS_A_dC_A_dev = C_dev_1 + C_dev_2;

    // Calculation for dS_A_dC_A_dev ends
    
    // Calculation for dS_A_dC_A_vol starts

    double G     = J * log((J - f_1)/(1 - f_1));
    double dG_dJ = log((J - f_1)/(1 - f_1)) + J / (J - f_1);

    // Pieces that add up to dS_A_dC_A_vol
    SymmetricTensor<4, dim> C_vol_1, C_vol_2;

    C_vol_1 = K * dG_dJ * outer_product(invert(C_A), dJ_dC);

    C_vol_2 = K * G * dC_inv_dC;

    dS_A_dC_A_vol = C_vol_1 + C_vol_2;

    // Calculation for dS_A_dC_A_vol ends

    dS_A_dC_A = dS_A_dC_A_vol + dS_A_dC_A_dev;

    return dS_A_dC_A;
}


template <int dim>
SymmetricTensor<4, dim> compute_tangent_modulus(
    Tensor<2, dim> F,  // Total deformation gradient
    Tensor<2, dim> F_A, // Elastic part of the deformation gradient
    Tensor<2, dim> F_B, // Viscous part of the deformation gradient
    SymmetricTensor<2, dim> D_B, // Symmetric part of viscous velocity gradient
    double delta_t, // Length of the current time step
    double f_R,
    SymmetricTensor<2, dim> S_B,
    SymmetricTensor<2, dim> S_B_dev
) {

    SymmetricTensor<2, dim> I = Physics::Elasticity::StandardTensors<dim>::I;

    SymmetricTensor<4, dim> I4;
    for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
            for (unsigned int k = 0; k < dim; ++k)
                for (unsigned int l = 0; l < dim; ++l)
                    I4[i][j][k][l] = 0.5 * (
                                     I[i][k] * I[j][l]
                                     +
                                     I[i][l] * I[j][k]
                                    );

    SymmetricTensor<2, dim> C   = symmetrize(transpose(F) * F);
    SymmetricTensor<2, dim> C_A = symmetrize(transpose(F_A) * F_A);
    Tensor<2, dim> F_B_inv      = invert(F_B);

    SymmetricTensor<4, dim> dC_A_inv_dC_A = 
        Physics::Elasticity::StandardTensors<dim>::Dev_P(F_A);

    // 4th rank tensors needed to compute the tangent modulus in reference coordinates
    SymmetricTensor<4, dim> A;
    SymmetricTensor<4, dim> B;
    SymmetricTensor<4, dim> M;
    SymmetricTensor<4, dim> N;
    SymmetricTensor<4, dim> P;

    Tensor<2, dim> F_B_invT_C = transpose(F_B_inv) * C;
    Tensor<2, dim> D_B_F_B_inv = D_B * F_B_inv;

    for (unsigned int i = 0; i < dim; ++i) {
        for (unsigned int j = 0; j < dim; ++j) {
            for (unsigned int k = 0; k < dim; ++k) {
                for (unsigned int l = 0; l < dim; ++l) {

                    A[i][j][k][l] = F_B_inv[k][i] * F_B_inv[l][j]
                                  + F_B_inv[l][i] * F_B_inv[k][j]
                                  - (
                                    F_B_inv[k][i] * D_B_F_B_inv[l][j]
                                  + F_B_inv[l][i] * D_B_F_B_inv[k][j]
                                  + F_B_inv[k][j] * D_B_F_B_inv[l][i]
                                  + F_B_inv[l][j] * D_B_F_B_inv[k][i]
                                  ) * delta_t;

                    B[i][j][k][l] = - (
                        F_B_invT_C[i][k] * F_B_inv[l][j] +
                        F_B_invT_C[j][k] * F_B_inv[l][i]
                        ) * delta_t;

                }
            }
        }
    }

    M = (gamma_dot_0 * f_R / pow(sqrt(2) * sigma_0, np)) 
        * (
          2 * pow(S_B_dev * S_B_dev, (3-2*np)/(2*np-2))
        * outer_product(S_B_dev, S_B_dev)
        + pow(S_B_dev * S_B_dev, (np - 1)/2)
        * I4);

    N = I4 - (1/3) * outer_product(C_A, invert(C_A));

    P = -(1/3) * (
          (invert(C_A) * S_B) * I4
        + outer_product(C_A, S_B)
        * dC_A_inv_dC_A);

    // Derivative of the 2nd PK stress with respect to the total strain metric
    SymmetricTensor<4, dim> dS_A_dC_A = compute_dS_A_dC_A(F_A);

    // Compute the tangent modulus in reference coordinates
    SymmetricTensor<4, dim> dS_A_dC; // Tangent modulus in reference coordinates

    dS_A_dC = dS_A_dC_A * invert(I4 - B * M * P) * (A + B * M * N);

    // Push the tangent modulus in reference coordinates to the spatial configuration
    SymmetricTensor<4, dim> Jc; // Tangent modulus in spatial coordinates

    /*Jc = (K / (1-f_1)) * outer_product(I, I);*/
    Jc = determinant(F) 
       /** Physics::Transformations::Contravariant::push_forward(2*dS_A_dC_A, F_A);*/
       * Physics::Transformations::Contravariant::push_forward(2*dS_A_dC, F);

    return Jc;

}

template <int dim>
class PointHistory {
    public:
        PointHistory() {

            SymmetricTensor<2, dim> I = Physics::Elasticity::StandardTensors<dim>::I;

            kirchhoff_stress = 0;
            deformation_gradient = I;

            // Initialize viscoelasticity variables
            F_B = I;
            F_D = I;

            tangent_modulus = 
                compute_tangent_modulus(
                    deformation_gradient, 
                    deformation_gradient * invert(F_B), 
                    F_B,
                    SymmetricTensor<2, dim>(),
                    0,
                    1,
                    SymmetricTensor<2, dim>(),
                    SymmetricTensor<2, dim>());

        }

        virtual ~PointHistory() = default;
        Tensor<2, dim> deformation_gradient;
        SymmetricTensor<2, dim> kirchhoff_stress;
        SymmetricTensor<4, dim> tangent_modulus;

        // Accumulated viscous strains for the viscoelastic model
        Tensor<2, 3> F_B;
        Tensor<2, 3> F_D;

};

