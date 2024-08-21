// Mesh generation
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>

// Store data at quadrature points
#include <deal.II/base/quadrature_point_data.h>

// For calculating strain tensors at quadrature points
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

// For pull back and push forward operations
#include <deal.II/physics/transformations.h>

// Linear algebra
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/identity_matrix.h>

// Boundary conditions
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

// Graphical output
#include <deal.II/numerics/data_out.h>

#include <iostream>

#include "parameters.h"

using namespace dealii;

template <int dim>
class PointHistory {
    public:
        PointHistory() {

            second_pk_stress = 0;
            deformation_gradient = Physics::Elasticity::StandardTensors<dim>::I;

            // Compute the Lamè parameters. Y and nu are in the parameters.h file.
            double lambda = (Y * nu) / ((1 + nu) * (1 - 2 * nu));
            double mu     = Y / (2 * (1 + nu));

            // Calculate the tangent modulus for hyperelasticity
            for (unsigned int i = 0; i < dim; ++i)
                for (unsigned int j = 0; j < dim; ++j)
                    for (unsigned int k = 0; k < dim; ++k)
                        for (unsigned int l = 0; l < dim; ++l)
                            tangent_modulus[i][j][k][l] = 
                                  (((i == k) && (j == l) ? mu : 0.0) +
                                   ((i == l) && (j == k) ? mu : 0.0) +
                                   ((i == j) && (k == l) ? lambda : 0.0));

        }

        virtual ~PointHistory() = default;
        Tensor<2, dim> deformation_gradient;
        SymmetricTensor<2, dim> second_pk_stress;
        SymmetricTensor<4, dim> tangent_modulus;
};

template <int dim>
class VelocityBoundaryCondition : public Function<dim> {
    public:
    VelocityBoundaryCondition(
        const double current_time,
        const double speed);

    virtual void vector_value(
                    const Point<dim> &p,
                    Vector<double> &values) 
                    const override;

    private:
    const double current_time;
    const double speed;
};

// The variables current_time and speed will be passed to the constructor for
// the velocity boundary conditions class by the top level class for the
// problem at the moment the interpolate_boundary_conditions function is called
template <int dim>
VelocityBoundaryCondition<dim>::VelocityBoundaryCondition(
    const double current_time,
    const double speed) : Function<dim>(dim)
    , current_time(current_time)
    , speed(speed)
{}

template <int dim>
void VelocityBoundaryCondition<dim>::vector_value(const Point<dim> &/*p*/,
                                             Vector<double> &values) const {
    // The variable name p has been commented out to avoid compiler warnings
    // about unused variables
    values = 0;
    values(2) = - speed * current_time;
}

template <int dim>
class Problem {
    public:
        Problem();
        void run();

    private: // functions
        void setup_system();
        void generate_boundary_conditions();
        void assemble_linear_system();
        void calculate_residual_norm();
        void solve_linear_system();
        void update_quadrature_point_histories();
        void output_results();

    private: // variables
        // Meshing
        Triangulation<dim> triangulation;
        DoFHandler<dim> dof_handler;
        FESystem<dim> fe;
        QGauss<dim> quadrature_formula;
        FEValues<dim> fe_values;

        // History data at quadrature points
        CellDataStorage<typename Triangulation<dim>::cell_iterator, PointHistory<dim>>
        quadrature_point_history;

        // Time stepping
        double residual_norm;
        double current_time;
        double delta_t;
        double total_time;
        int step_number;
        int iterations;

        // Linear algebra
        Vector<double> system_rhs;
        Vector<double> solution, delta_solution;
        Vector<double> residual;
        SparsityPattern sparsity_pattern;
        SparseMatrix<double> system_matrix;
        AffineConstraints<double> non_homogenous_constraints,
                                      homogenous_constraints;

        // Output
        std::ofstream text_output_file;
        DataOut<dim> data_out;
};

template <int dim>
Problem<dim>::Problem () : 
    dof_handler(triangulation),
    fe(FE_Q<dim>(1)^dim),
    quadrature_formula(fe.degree + 1),
    fe_values(
            fe,
            quadrature_formula,
            update_values |
            update_quadrature_points |
            update_gradients |
            update_JxW_values),
    current_time(0),
    delta_t(1e-3),
    total_time(2e-3),
    step_number(0),
    text_output_file("text_output_file.txt")
    {}

template <int dim>
void Problem<dim>::run () {

    setup_system();

    current_time += delta_t; 
    step_number++;

    std::cout << "\n"
              << "Step number : " << step_number
              << " "
              << "Current time : " << current_time
              << "\n";

    iterations = 0;

    // Generate boundary conditions for the current increment
    generate_boundary_conditions();

    // Solve the current, nonlinear increment
    while (true) {
        std::cout << "\n";
        iterations++;
        assemble_linear_system();
        calculate_residual_norm();
        std::cout << "Residual norm : " << residual_norm << "\n";
        /*std::cout << "System RHS : " << system_rhs << "\n";*/
        solve_linear_system();
        /*std::cout << "Solution : " << solution << "\n";*/
        update_quadrature_point_histories();

        if (iterations == 4) {
            std::cout << "Max iterations reached.\n";
            exit(0);
        }
    }

    /*output_results();*/

}

template <int dim>
void Problem<dim>::setup_system () {
    std::cout << "-- Setting up\n" << std::endl;

    // Generate mesh
    GridGenerator::hyper_cube(triangulation);
    /*triangulation.refine_global(1);*/
    dof_handler.distribute_dofs(fe);

    // Make space for all the history variables of the system
    quadrature_point_history.initialize(triangulation.begin_active(),
                                        triangulation.end(),
                                        quadrature_formula.size());

    // Generate linear algebra objets
    delta_solution.reinit(dof_handler.n_dofs());
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(
                                dof_handler, 
                                dsp, 
                                non_homogenous_constraints, 
                                false);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);

    // Set boundary indices. The following boundary indexing assumes that the
    // domain is a cube of edge length 1 with sides parallel to and on the 3
    // coordinate planes.

    for (const auto &cell : triangulation.active_cell_iterators()) {
        for (const auto &face : cell->face_iterators()) {
            if (face->at_boundary()) {
                const Point<dim> face_center = face->center();

                // Face on the yz plane
                if(face_center[0] == 0) face->set_boundary_id(0);

                // Face opposite the yz plane
                if(face_center[0] == 1) face->set_boundary_id(1);

                // Face on the xz plane
                if(face_center[1] == 0) face->set_boundary_id(2);

                // Face opposite the xz plane
                if(face_center[1] == 1) face->set_boundary_id(3);

                // Face on the xy plane
                if(face_center[2] == 0) face->set_boundary_id(4);

                // Face opposite the xy plane
                if(face_center[2] == 1) face->set_boundary_id(5);

            }
        }
    }

    std::cout << "No of cells : " << triangulation.n_active_cells() << std::endl;
    std::cout << "No of vertices : " << triangulation.n_vertices() << std::endl;
    std::cout << "No of dofs : " << dof_handler.n_dofs() << std::endl;
    std::cout << "FE system dealii name : " << fe.get_name() << std::endl;

}

template <int dim>
void Problem<dim>::generate_boundary_conditions () {

    // The following are three arrays of boolean values that tell the
    // interpolate_boundary_values function which component to apply the
    // boundary values to.
    const FEValuesExtractors::Scalar x_component(0);
    const FEValuesExtractors::Scalar y_component(1);
    const FEValuesExtractors::Scalar z_component(2);

    non_homogenous_constraints.clear();

    // The face on the yz plane has boundary indicator of 0 and must be kept
    // from moving in the x direction
    VectorTools::interpolate_boundary_values(
                            dof_handler,
                            0,
                            Functions::ZeroFunction<dim>(dim),
                            non_homogenous_constraints,
                            fe.component_mask(x_component));

    // The face on the xz plane has boundary indicator of 2 and must be kept
    // from moving in the y direction
    //
    VectorTools::interpolate_boundary_values(
                            dof_handler,
                            2,
                            Functions::ZeroFunction<dim>(dim),
                            non_homogenous_constraints,
                            fe.component_mask(y_component));

    // The face on the xy plane has boundary indicator of 4 and must be kept
    // from moving in the z direction
    VectorTools::interpolate_boundary_values(
                            dof_handler,
                            4,
                            Functions::ZeroFunction<dim>(dim),
                            non_homogenous_constraints,
                            fe.component_mask(z_component));

    // The face opposite the xy plane has boundary indicator of 5. This is
    // where the velocity boundary condition must be applied.
    VectorTools::interpolate_boundary_values(
                            dof_handler,
                            5,
                            VelocityBoundaryCondition<dim>(current_time, z1_speed),
                            non_homogenous_constraints,
                            fe.component_mask(z_component));

    non_homogenous_constraints.close();


    homogenous_constraints.clear();

    // The face on the yz plane has boundary indicator of 0 and must be kept
    // from moving in the x direction
    VectorTools::interpolate_boundary_values(
                            dof_handler,
                            0,
                            Functions::ZeroFunction<dim>(dim),
                            homogenous_constraints,
                            fe.component_mask(x_component));

    // The face on the xz plane has boundary indicator of 2 and must be kept
    // from moving in the y direction
    VectorTools::interpolate_boundary_values(
                            dof_handler,
                            2,
                            Functions::ZeroFunction<dim>(dim),
                            homogenous_constraints,
                            fe.component_mask(y_component));

    // The face on the xy plane has boundary indicator of 4 and must be kept
    // from moving in the z direction
    VectorTools::interpolate_boundary_values(
                            dof_handler,
                            4,
                            Functions::ZeroFunction<dim>(dim),
                            homogenous_constraints,
                            fe.component_mask(z_component));

    // The face opposite the xy plane has boundary indicator of 5. This is
    // where the velocity boundary condition must be applied.
    VectorTools::interpolate_boundary_values(
                            dof_handler,
                            5,
                            Functions::ZeroFunction<dim>(dim),
                            homogenous_constraints,
                            fe.component_mask(z_component));

    homogenous_constraints.close();

}

template <int dim>
void Problem<dim>::assemble_linear_system () {

    system_matrix = 0.0;
    system_rhs = 0.0;

    const unsigned int dofs_per_cell       = fe.n_dofs_per_cell();
    const unsigned int n_quadrature_points = fe_values.n_quadrature_points;

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    // types::global_dof_index is an unsigned int of 32 bits on most systems.
    // So the following is an array of integers.
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Object for temporarily storing the quadrature point data while
    // performing quadrature over the current cell
    std::vector<std::shared_ptr<PointHistory<dim>>> quadrature_point_history_data;

    Tensor<2, dim> F;    // Deformation gradient
    Tensor<2, dim> Finv; // Inverse of the deformation gradient

    SymmetricTensor<2, dim> S; // Second Piola-Kirchhoff stress
    SymmetricTensor<4, dim> C; // Tangent modulus in the reference configuration

    SymmetricTensor<2, dim> Js; // Kirchhoff stress
    SymmetricTensor<4, dim> Jc; // Spatial tangent modulus * determinant(F)

    SymmetricTensor<2, dim> delta = Physics::Elasticity::StandardTensors<dim>::I;

    // Create rank 1 tensors to hold the gradients of the shape functions with
    // respect to the spatial coordinates
    Tensor<1, dim> dphidx_i, dphidx_j;

    // Loop over all the cells of the triangulation
    for (const auto &cell : dof_handler.active_cell_iterators()) {

        // Initialize the fe_values object with values relevant to the current cell
        fe_values.reinit(cell);

        // Initialize the cell matrix and the right hand side vector with zeros
        cell_matrix = 0.0; 
        cell_rhs = 0.0;

        // Temporary structure for holding the quadrature point data
        quadrature_point_history_data = quadrature_point_history.get_data(cell);

        // Quadrature loop for current cell and degrees of freedom i, j
        for (unsigned int q = 0; q < n_quadrature_points; q++) {

            S = quadrature_point_history_data[q]->second_pk_stress;
            F = quadrature_point_history_data[q]->deformation_gradient;
            C = quadrature_point_history_data[q]->tangent_modulus;

            Js = Physics::Transformations::Contravariant::push_forward(S, F);
            Jc = Physics::Transformations::Contravariant::push_forward(C, F);

            Finv = invert(F);

            /*if (iterations == 2) {*/
            /*    std::cout << "Quadrature point : " << q + 1 << "\n";*/
            /*    std::cout << "2nd PK : " << S << "\n";*/
            /*    std::cout << "Cauchy stress : " << s / determinant(F) << "\n";*/
            /*    std::cout << "Deformation gradient : " << F << "\n";*/
            /*    std::cout */
            /*        << "Green Lagrange strain : " */
            /*        << Physics::Elasticity::Kinematics::E(F) */
            /*        << "\n";*/
            /*}*/

            for (unsigned int i = 0; i < dofs_per_cell; i++) {

                const unsigned int ci = fe_values
                                        .get_fe()
                                        .system_to_component_index(i)
                                        .first;

                for (unsigned int j = 0; j < dofs_per_cell; j++) {

                    const unsigned int cj = fe_values
                                            .get_fe()
                                            .system_to_component_index(j)
                                            .first;

                    dphidx_i = 0.0; 
                    dphidx_j = 0.0; 

                    // Transform the gradients of the shape functions returned
                    // by dealii to the current configuration
                    for (unsigned int m = 0; m < dim; m++) {
                        for (unsigned int n = 0; n < dim; n++) {
                            dphidx_i[m] += fe_values.shape_grad(i, q)[n] * Finv[n][m];
                            dphidx_j[m] += fe_values.shape_grad(j, q)[n] * Finv[n][m];
                        }
                    }

                    for (unsigned int di = 0; di < dim; di++) {
                        cell_rhs(i) +=
                             -dphidx_i[di] * Js[ci][di] * fe_values.JxW(q);
                        for (unsigned int dj = 0; dj < dim; dj++) {
                            cell_matrix(i, j) +=
                                dphidx_i[di] *
                                Jc[ci][di][cj][dj] *
                                dphidx_j[dj] *
                                fe_values.JxW(q)
                                +
                                dphidx_i[di] *
                                delta[ci][cj] * Js[di][dj] *
                                dphidx_j[dj] *
                                fe_values.JxW(q);
                        }
                    }
                } // End of j loop
            } // End of i loop
        } // End of quadrature loop

        std::cout << "Iterations : " << iterations << "\n";
        std::cout << "cell rhs : " << cell_rhs << "\n";

        // Distribute local contributions to global system
        cell->get_dof_indices(local_dof_indices);

        if (iterations == 1) {
            // Apply non-homogenous boundary conditions only in the first
            // iteration of the increment.
            non_homogenous_constraints.distribute_local_to_global(
                        cell_matrix,
                        cell_rhs,
                        local_dof_indices,
                        system_matrix,
                        system_rhs);
        } else {
            // Non-homogenous boundary conditions will be satisfied in the
            // first iteration of the increment. No need to change the
            // components of the solution corresponding to unconstrained dofs
            // in the remaining iterations of the increment.
            homogenous_constraints.distribute_local_to_global(
                        cell_matrix,
                        cell_rhs,
                        local_dof_indices,
                        system_matrix,
                        system_rhs);

        }

        std::cout << "system rhs : " << system_rhs << "\n";

        /*if (true) {*/
        /*std::cout << "iterations : " << iterations << "\n";*/
        /*std::cout << "cell_rhs : " << cell_rhs << "\n";*/
        /*std::cout << "system_rhs : " << system_rhs << "\n";*/
        /*}*/

    } // End of loop over all cells

}

template <int dim>
void Problem<dim>::calculate_residual_norm () {

    residual.reinit(dof_handler.n_dofs());

    for (unsigned int i = 0; i < dof_handler.n_dofs(); i++) {
        if (!homogenous_constraints.is_constrained(i))
            residual(i) = system_rhs(i);
    }

    residual_norm = residual.l2_norm();

}

template <int dim>
void Problem<dim>::solve_linear_system () {

    delta_solution = 0.0;

    // The solver will do a maximum of 1000 iterations before giving up
    SolverControl solver_control(1000, 1e-6);
    SolverCG<Vector<double>> solver_cg(solver_control);
    solver_cg.solve(system_matrix,
                    delta_solution,
                    system_rhs,
                    IdentityMatrix(solution.size()));

    if (iterations == 1) {
        non_homogenous_constraints.distribute(delta_solution);
    } else {
        homogenous_constraints.distribute(delta_solution);
    } 

    /*std::cout << "iterations : " << iterations << "\n";*/
    /*std::cout << "delta solution : " << delta_solution << "\n";*/

    solution += delta_solution;

}

template <int dim>
void Problem<dim>::update_quadrature_point_histories () {

    // Vector of vectors for storing the gradients of the displacement field at the
    // integration points of a cell. The outer vector has length equal to the
    // number of quadrature points in a cell. Each of the inner vectors is a
    // list of dim elements of type Tensor<1, dim>.
    std::vector<std::vector<Tensor<1, dim>>> solution_gradients(
                                                quadrature_formula.size(),
                                                std::vector<Tensor<1, dim>>(dim));

    Tensor<2, dim> 
    dUdX, // Gradient of displacement wrt reference coordinates
    F;    // Deformation gradient

    SymmetricTensor<2, dim>
    E, // Green Lagrange strain tensor
    S; // Second Piola Kirchhoff stress

    SymmetricTensor<4, dim> C; // Material tangent modulus
    // Compute the Lamè parameters. Y and nu are Young's modulus and Poisson's
    // ratio respectively and defined in the parameters.h file.
    double lambda = (Y * nu) / ((1 + nu) * (1 - 2 * nu));
    double mu     = Y / (2 * (1 + nu));

    // Calculate the tangent modulus for hyperelasticity
    for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
            for (unsigned int k = 0; k < dim; ++k)
                for (unsigned int l = 0; l < dim; ++l)
                    C[i][j][k][l] = 
                          (((i == k) && (j == l) ? mu : 0.0) +
                           ((i == l) && (j == k) ? mu : 0.0) +
                           ((i == j) && (k == l) ? lambda : 0.0));

    // Temporary structure for holding the quadrature point data
    std::vector<std::shared_ptr<PointHistory<dim>>> quadrature_point_history_data;

    for (auto &cell : dof_handler.active_cell_iterators()) {

        // Initialize the fe_values object with values relevant to the current cell
        fe_values.reinit(cell);

        // Get displacement gradients at all integration points of the cell
        // from dealii
        fe_values.get_function_gradients(solution, solution_gradients);

        quadrature_point_history_data = quadrature_point_history.get_data(cell);

        for (unsigned int q = 0; q < quadrature_formula.size(); q++) {

            for (unsigned int i = 0; i < dim; i++)
                for (unsigned int j = 0; j < dim; j++)
                    dUdX[i][j] = solution_gradients[q][i][j];

            F = Physics::Elasticity::Kinematics::F(dUdX);
            E = Physics::Elasticity::Kinematics::E(F);

            S = C * E;

            /*if (q == 0 and iterations == 1) {*/
            /*    std::cout << "S11 = " << S[0][0] << "\n";*/
            /*    std::cout << "C1111 * E11 = " << C[0][0][0][0] << " * " << E[0][0] << " = " << C[0][0][0][0] * E[0][0] << "\n";*/
            /*    std::cout << "C1112 * E12 = " << C[0][0][0][1] << " * " << E[0][1] << " = " << C[0][0][0][1] * E[0][1] << "\n";*/
            /*    std::cout << "C1113 * E13 = " << C[0][0][0][2] << " * " << E[0][2] << " = " << C[0][0][0][2] * E[0][2] << "\n";*/
            /*    std::cout << "C1121 * E21 = " << C[0][0][1][0] << " * " << E[1][0] << " = " << C[0][0][1][0] * E[1][0] << "\n";*/
            /*    std::cout << "C1122 * E22 = " << C[0][0][1][1] << " * " << E[1][1] << " = " << C[0][0][1][1] * E[1][1] << "\n";*/
            /*    std::cout << "C1123 * E23 = " << C[0][0][1][2] << " * " << E[1][2] << " = " << C[0][0][1][2] * E[1][2] << "\n";*/
            /*    std::cout << "C1131 * E31 = " << C[0][0][2][0] << " * " << E[2][0] << " = " << C[0][0][2][0] * E[2][0] << "\n";*/
            /*    std::cout << "C1132 * E32 = " << C[0][0][2][1] << " * " << E[2][1] << " = " << C[0][0][2][1] * E[2][1] << "\n";*/
            /*    std::cout << "C1133 * E33 = " << C[0][0][2][2] << " * " << E[2][2] << " = " << C[0][0][2][2] * E[2][2] << "\n";*/
            /*}*/

            quadrature_point_history_data[q]->second_pk_stress = S;
            quadrature_point_history_data[q]->deformation_gradient = F;

        }
    }
}    

template <int dim>
void Problem<dim>::output_results () {
    
    std::ofstream output_file("solution.vtu");
    
    data_out.attach_dof_handler(dof_handler);

    std::vector<std::string> solution_names;

    solution_names.emplace_back("x_displacement");
    solution_names.emplace_back("y_displacement");
    solution_names.emplace_back("z_displacement");

    data_out.add_data_vector(solution, solution_names);

    data_out.build_patches();;
    data_out.write_vtu (output_file);

    std::cout << "\nResults written" << std::endl;
}

int main() {

    std::cout << "\n---- Simulation started\n" << std::endl;

    // Create the problem
    Problem<3> problem;

    // Solve the problem
    problem.run();

    std::cout << "\n---- Simulation ended" << std::endl;

    return 0;

}
