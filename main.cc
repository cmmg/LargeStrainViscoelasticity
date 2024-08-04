// Mesh generation
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>

// Linear algebra
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

// Boundary conditions
#include <deal.II/numerics/vector_tools.h>

// Graphical output
#include <deal.II/numerics/data_out.h>

#include <iostream>

#include "parameters.h"

using namespace dealii;

template <int dim>
class VelocityBoundaryCondition : public Function<dim> {
    public:
    VelocityBoundaryCondition(
        const double current_time,
        const double speed);

    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &values) const override;

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
        void assemble_system();
        void solve_linear_system();
        void output_results();

    private: // variables
        // Meshing
        Triangulation<dim> triangulation;
        DoFHandler<dim> dof_handler;
        FESystem<dim> fe;
        QGauss<dim> quadrature_formula;

        // Time stepping
        double current_time;
        double delta_t;
        double total_time;
        double step_no;

        // Linear algebra
        Vector<double> system_rhs;
        Vector<double> solution;
        SparsityPattern sparsity_pattern;
        SparseMatrix<double> system_matrix;
        AffineConstraints<double> constraints;

        // Graphical output
        DataOut<dim> data_out;
};

template <int dim>
Problem<dim>::Problem () : 
    dof_handler(triangulation),
    fe(FE_Q<dim>(1)^dim),
    quadrature_formula(fe.degree + 1)
    {}

template <int dim>
void Problem<dim>::run () {
    setup_system();

    current_time = 0.1;
    assemble_system();
    solve_linear_system();
    output_results();
}

template <int dim>
void Problem<dim>::setup_system () {
    std::cout << "-- Setting up\n" << std::endl;

    // Generate mesh
    GridGenerator::hyper_cube(triangulation);
    /*triangulation.refine_global(1);*/
    dof_handler.distribute_dofs(fe);

    // Generate linear algebra objets
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
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

    std::cout << "\n-- Set up complete" << std::endl;
}

template <int dim>
void Problem<dim>::assemble_system () {

    std::cout << "\n-- Assembling system\n" << std::endl;

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values |
                            update_gradients |
                            update_quadrature_points |
                            update_JxW_values);

    unsigned int n_quadrature_points = fe_values.n_quadrature_points;

    // Constitutive model computation begin -------------
    
    // Declare and compute the Kronecker delta tensor
    Tensor<2, dim> kronecker_delta; 
    kronecker_delta = 0;
    kronecker_delta[0][0] = 1;
    kronecker_delta[1][1] = 1;
    kronecker_delta[2][2] = 1;

    // Compute the Lamè parameters. E and nu are in the parameters.h file.
    double lambda = (E * nu) / ((1 + nu) * (1 - 2 * nu));
    double mu     = E / (2 * (1 + nu));

    // Calculate the tangent modulus for hyperelasticity
    Tensor<4, dim> tangent_modulus;
    for(unsigned int i = 0; i < dim; i++) {
        for(unsigned int j = 0; j < dim; j++) {
            for(unsigned int k = 0; k < dim; k++) {
                for(unsigned int l = 0; l < dim; l++) {
                    tangent_modulus[i][j][k][l] = 
                        lambda * kronecker_delta[i][j] * kronecker_delta[k][l]
                        +
                        mu * (kronecker_delta[i][k] * kronecker_delta[j][l] +
                              kronecker_delta[i][l] * kronecker_delta[j][k]);
                }
            }
        }
    }

    // Constitutive model computation end --------------

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    // types::global_dof_index is an unsigned int of 32 bits in most cases. So
    // the following is an array of integers
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Loop over all the cells of the triangulation
    for (const auto &cell : dof_handler.active_cell_iterators()) {

        // Initialize the fe_values object with values relevant to the current cell
        fe_values.reinit(cell);

        // Initialize the cell matrix and the right hand side vector with zeros
        cell_matrix = 0; 
        cell_rhs = 0;

        // Quadrature loop for current cell
        for (unsigned int q = 0; q < n_quadrature_points; q++) {

            for (unsigned int i = 0; i < dofs_per_cell; i++) {

                const unsigned int ci = fe_values
                                        .get_fe()
                                        .system_to_component_index(i)
                                        .first;

                for (unsigned int j = 0; j < dofs_per_cell; j++) {
                    const unsigned int cj = fe_values
                                            .get_fe()
                                            .system_to_component_index(i)
                                            .first;

                    for (unsigned int di; di < dim; di++) {
                        for (unsigned int dj; dj < dim; dj++) {
                            cell_matrix(i, j) +=
                                fe_values.shape_grad(i, q)[di] *
                                tangent_modulus[ci][di][cj][dj] *
                                fe_values.shape_grad(j, q)[dj] *
                                fe_values.JxW(q);
                        }
                    }
                }
            }
        }

        // Distribute local contributions to global system
        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(
                    cell_matrix,
                    cell_rhs,
                    local_dof_indices,
                    system_matrix,
                    system_rhs);

    } // End of loop over all cells

    // Start applying boundary conditions

    // Create an object that will hold the values of the boundary conditions
    std::map<types::global_dof_index, double> boundary_values;

    // The following are three arrays of boolean values that tell the
    // interpolate_boundary_values function which component to apply the
    // boundary values to.
    const FEValuesExtractors::Scalar x_component(0);
    const FEValuesExtractors::Scalar y_component(1);
    const FEValuesExtractors::Scalar z_component(2);

    // The face on the yz plane has boundary indicator of 0 and must be kept
    // from moving in the x direction
    VectorTools::interpolate_boundary_values(
                            dof_handler,
                            0,
                            Functions::ZeroFunction<dim>(dim),
                            boundary_values,
                            fe.component_mask(x_component));

    // The face on the xz plane has boundary indicator of 2 and must be kept
    // from moving in the y direction
    VectorTools::interpolate_boundary_values(
                            dof_handler,
                            2,
                            Functions::ZeroFunction<dim>(dim),
                            boundary_values,
                            fe.component_mask(y_component));

    // The face on the xy plane has boundary indicator of 4 and must be kept
    // from moving in the z direction
    VectorTools::interpolate_boundary_values(
                            dof_handler,
                            4,
                            Functions::ZeroFunction<dim>(dim),
                            boundary_values,
                            fe.component_mask(z_component));

    std::cout << "Current time : " << current_time << std::endl;

    // The face opposite the xy plane has boundary indicator of 5. This is
    // where the velocity boundary condition must be applied.
    VectorTools::interpolate_boundary_values(
                            dof_handler,
                            5,
                            VelocityBoundaryCondition<dim>(current_time, z1_speed),
                            boundary_values,
                            fe.component_mask(z_component));

    std::cout << "\n-- Assembly complete" << std::endl;
}

template <int dim>
void Problem<dim>::solve_linear_system () {
    // The solver will do a maximum of 1000 iterations before giving up
    SolverControl solver_control(1000, 1e-12);
    SolverCG<Vector<double>> cg(solver_control);

    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    std::cout << "Solution before solving" << std::endl;
    solution.print(std::cout);
    std::cout << "System rhs before solving" << std::endl;
    system_rhs.print(std::cout);

    cg.solve(system_matrix, solution, system_rhs, preconditioner);

    std::cout << "Solution after solving" << std::endl;
    solution.print(std::cout);
    std::cout << "System rhs after solving" << std::endl;
    system_rhs.print(std::cout);

    std::cout << "\n-- " << solver_control.last_step()
        << " iterations needed to obtain convergence."
        << std::endl;
}

template <int dim>
void Problem<dim>::output_results () {
    
    std::ofstream output_file("solution.vtu");
    
    data_out.attach_dof_handler(dof_handler);
    data_out.build_patches();
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
