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

