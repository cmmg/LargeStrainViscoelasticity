// Mesh generation
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe.h>
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

// Graphical output
#include <deal.II/numerics/data_out.h>
#include <deal.II/fe/mapping_q_eulerian.h>

// Reading input parameters
#include <deal.II/base/parameter_handler.h>

using namespace dealii;

#include <iostream>

/*#include "mechanics.h"*/
/*#include "viscoelastic_mechanics.h"*/
#include "viscoelastic_mechanics_bottom_up.h"
/*#include "st_venant_elasticity.h"*/

template <int dim>
class Problem {
    public:
        Problem();
        void run();

    private: // functions
        void declare_parameters();
        void setup_system();
        void generate_boundary_conditions();
        void assemble_linear_system();
        void calculate_residual_norm();
        void solve_linear_system();
        void update_all_history_data();
        void perform_L2_projections();
        void output_results();

    private: // variables

        // Meshing
        Triangulation<dim> triangulation;
        DoFHandler<dim> dof_handler;
        FESystem<dim> fe;
        QGauss<dim> quadrature_formula;
        FEValues<dim> fe_values;

        // L2 projection
        DoFHandler<dim> dof_handler_L2;
        FESystem<dim> fe_L2;
        FEValues<dim> fe_values_L2;
        AffineConstraints<double> constraints_L2;
        SparsityPattern sparsity_pattern_L2;
        SparseMatrix<double> mass_matrix_L2;
        std::vector<Vector<double>> nodal_output_L2;

        // History data at quadrature points
        CellDataStorage<typename Triangulation<dim>::cell_iterator, Material<dim>>
        quadrature_point_history;

        // Time stepping
        double initial_residual_norm;
        double residual_norm;
        double relative_tolerance;
        double absolute_tolerance;
        double current_time;
        double delta_t;
        double total_time;
        int step_number;
        int iterations;
        int max_no_of_NR_iterations;

        // Linear algebra objects for solving the problem
        Vector<double> system_rhs;
        Vector<double> solution;
        Vector<double> delta_solution;
        Vector<double> residual;
        SparsityPattern sparsity_pattern;
        SparseMatrix<double> system_matrix;
        AffineConstraints<double> non_homogenous_constraints,
                                      homogenous_constraints;

        // Reading parameters from an input file
        ParameterHandler parameter_handler;
        std::ifstream parameters_file;

        // Output
        DataOut<dim> data_out;
        std::ofstream text_output_file;

};

template <int dim>
Problem<dim>::Problem () : 
    dof_handler(triangulation),
    fe(FE_Q<dim>(1), dim),
    quadrature_formula(fe.degree + 1),
    fe_values(
            fe,
            quadrature_formula,
            update_values |
            update_quadrature_points |
            update_gradients |
            update_JxW_values),
    dof_handler_L2(triangulation),
    fe_L2(FE_Q<dim>(1), 1),
    fe_values_L2(
            fe_L2,
            quadrature_formula,
            update_values |
            update_JxW_values),
    max_no_of_NR_iterations(10),
    parameters_file("parameters.json"),
    text_output_file("text_output_file.txt")
    {}

template <int dim>
void Problem<dim>::run () {

    declare_parameters();
    setup_system();

    parameter_handler.enter_subsection("Time Stepping and Iteration Control");

    relative_tolerance = parameter_handler.get_double("relative tolerance");
    absolute_tolerance = parameter_handler.get_double("absolute tolerance");
    delta_t            = parameter_handler.get_double("time step length");
    total_time         = parameter_handler.get_double("total simulation time");

    parameter_handler.leave_subsection();

    step_number = 0;
    current_time = 0.0;
    solution = 0.0;

    perform_L2_projections();
    output_results(); // Output the initial state of the system to the output file

    while (current_time < total_time) {

        if (current_time + delta_t > total_time) {

            delta_t = total_time - current_time;
            current_time = total_time;

        } else {

            current_time += delta_t; 

        }

        step_number++;

        std::cout << "\n"
                  << "Step number : " << step_number
                  << " "
                  << "Current time : " << current_time
                  << "\n";

        // Reset the number of iterations for every time step
        iterations = 0;

        // Generate homogenous and non homogenous boundary conditions for the
        // current increment

        generate_boundary_conditions();

        // Calculations of the zeroth iteration of the increment are used to set
        // the initial norm of the residual to be used for the convergence
        // criterion.

        delta_solution = 0.0;

        non_homogenous_constraints.distribute(delta_solution);

        solution += delta_solution;

        /*std::cout << "delta_solution = " << delta_solution << std::endl;*/
        /*std::cout << "Printing from run function" << std::endl;*/
        /*std::cout << "solution       = " << solution << std::endl;*/

        update_all_history_data();
        assemble_linear_system();
        calculate_residual_norm();
        initial_residual_norm = residual_norm;
        std::cout << "Initial norm : " << residual_norm << "\n";

        solve_linear_system();
        update_all_history_data();
        iterations++;

        // Solve the current, nonlinear increment
        while (true) {

            assemble_linear_system();
            calculate_residual_norm();

            std::cout 
                << "Iteration : " << iterations << " "
                << "Relative force norm : " << residual_norm / initial_residual_norm << " "
                << std::endl;

            if (iterations == max_no_of_NR_iterations) {
                std::cout << "\nMax Newton-Raphson iterations reached. Exiting program.\n";
                exit(0);
            }

            if (initial_residual_norm == 0 
                or
                residual_norm / initial_residual_norm < 1e-9) {

                std::cout 
                    << "Step converged in " 
                    << iterations 
                    << " iteration(s)." 
                    << std::endl;

                break;
            }

            /*if (iterations == 3) {*/
            /*    std::cout << "Exiting." << std::endl; */
            /*    exit(0);*/
            /*}*/

            solve_linear_system();

            update_all_history_data();
            iterations++;

        } // Nonlinear time step converged. Time to write to result files.

        perform_L2_projections();

        output_results();

    }

} // End of run function

template <int dim>
void Problem<dim>::declare_parameters () {

    parameter_handler.enter_subsection("Domain Geometry");

    parameter_handler.declare_entry("length", "1.0", Patterns::Double());
    parameter_handler.declare_entry("width", "1.0", Patterns::Double());
    parameter_handler.declare_entry("height", "1.0", Patterns::Double());

    parameter_handler.leave_subsection();

    parameter_handler.enter_subsection("Loading Condition");

    parameter_handler.declare_entry("nominal strain rate", "1", Patterns::Double());

    parameter_handler.leave_subsection();

    parameter_handler.enter_subsection("Time Stepping and Iteration Control");

    parameter_handler.declare_entry("relative tolerance", "1e-12", Patterns::Double());
    parameter_handler.declare_entry("absolute tolerance", "1e-12", Patterns::Double());
    parameter_handler.declare_entry("time step length", "1e-3", Patterns::Double());
    parameter_handler.declare_entry("total simulation time", "0.5", Patterns::Double());

    parameter_handler.leave_subsection();

    parameter_handler.enter_subsection("Viscoelastic Material Parameters");

    parameter_handler.declare_entry("K", "800.0", Patterns::Double());
    parameter_handler.declare_entry("f1", "0.0", Patterns::Double());
    parameter_handler.declare_entry("mu0", "20.0", Patterns::Double());
    parameter_handler.declare_entry("lambdaL", "1.09", Patterns::Double());
    parameter_handler.declare_entry("sigma0", "25", Patterns::Double());
    parameter_handler.declare_entry("n", "3", Patterns::Double());
    parameter_handler.declare_entry("G0", "4500", Patterns::Double());
    parameter_handler.declare_entry("Ginfinity", "600", Patterns::Double());
    parameter_handler.declare_entry("eta", "60000", Patterns::Double());
    parameter_handler.declare_entry("gammadot0", "1e-4", Patterns::Double());
    parameter_handler.declare_entry("alpha", "0.005", Patterns::Double());

    parameter_handler.leave_subsection();

    parameter_handler.parse_input_from_json(parameters_file);
}

template <int dim>
void Problem<dim>::setup_system () {

    std::cout << "-- Setting up\n" << std::endl;

    parameter_handler.enter_subsection("Domain Geometry");

    double length = parameter_handler.get_double("length");
    double width  = parameter_handler.get_double("width");
    double height = parameter_handler.get_double("height");

    parameter_handler.leave_subsection();

    // Generate mesh
    GridGenerator::hyper_rectangle(triangulation, 
                                   Point<dim>(0, 0, 0),
                                   Point<dim>(length, width, height));

    /*triangulation.refine_global(1);*/

    // Make space for all the history variables of the system
    quadrature_point_history.initialize(triangulation.begin_active(),
                                        triangulation.end(),
                                        quadrature_formula.size());

    // Set boundary indices. The following boundary indexing assumes that the
    // domain is a cuboidal with sides parallel to and on the 3 coordinate
    // planes. This loop also reads material parameters from the input file and
    // assigns the same to all materials.

    // Object for temporarily storing the quadrature point data while loading
    // material parameters from the input file.
    std::vector<std::shared_ptr<Material<dim>>> quadrature_point_history_data;

    for (const auto &cell : triangulation.active_cell_iterators()) {
        for (const auto &face : cell->face_iterators()) {
            if (face->at_boundary()) {
                const Point<dim> face_center = face->center();

                // Face on the yz plane
                if(face_center[0] == 0) face->set_boundary_id(0);

                // Face opposite the yz plane
                if(face_center[0] == length) face->set_boundary_id(1);

                // Face on the xz plane
                if(face_center[1] == 0) face->set_boundary_id(2);

                // Face opposite the xz plane
                if(face_center[1] == width) face->set_boundary_id(3);

                // Face on the xy plane
                if(face_center[2] == 0) face->set_boundary_id(4);

                // Face opposite the xy plane
                if(face_center[2] == height) face->set_boundary_id(5);

            }
        }

        // Load material parameters from input file for all integration points
        // of all cells.
        fe_values.reinit(cell);

        quadrature_point_history_data = quadrature_point_history.get_data(cell);

        parameter_handler.enter_subsection("Viscoelastic Material Parameters");

        // Quadrature loop for current cell
        for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q) {
            quadrature_point_history_data[q]->load_material_parameters(parameter_handler);
            quadrature_point_history_data[q]->compute_initial_tangent_modulus();
            quadrature_point_history_data[q]->integration_point_index = q + 1;
            quadrature_point_history_data[q]->text_output_file = &text_output_file;
        }

        parameter_handler.leave_subsection();

    }

    dof_handler.distribute_dofs(fe);

    // Set the sizes of the linear algebra objects
    residual.reinit(dof_handler.n_dofs());
    delta_solution.reinit(dof_handler.n_dofs());
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    // Make space in memory for the system matrix
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(
                                dof_handler, 
                                dsp, 
                                non_homogenous_constraints, 
                                false);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);

    // Make space in memory for the mass matrix used for L2 projection
    dof_handler_L2.distribute_dofs(fe_L2);
    DynamicSparsityPattern dsp_L2(dof_handler_L2.n_dofs());
    DoFTools::make_sparsity_pattern(
                                dof_handler_L2, 
                                dsp_L2, 
                                constraints_L2, 
                                false);
    sparsity_pattern_L2.copy_from(dsp_L2);
    mass_matrix_L2.reinit(sparsity_pattern_L2);

    // Set up complete. Output details to screen.
    std::cout << "No of cells : " << triangulation.n_active_cells() << std::endl;
    std::cout << "No of vertices : " << triangulation.n_vertices() << std::endl;
    std::cout << "No of dofs : " << dof_handler.n_dofs() << std::endl;
    std::cout << "FE system dealii name : " << fe.get_name() << std::endl;

}

template <int dim>
void Problem<dim>::generate_boundary_conditions () {

    /*

    The following are descriptions of the boundary conditions applied by this
    function. In the following, assume that the boundary conditions are being
    applied to a cube of side 1 with three of its faces lying on the coordinate
    planes.

    - constrained_shear_no_lateral_displacement
        One face is not allowed any displacement at all. The opposite face
    moves parallel to the fixed face. The points on the moving face are NOT
    allowed to move perpendicular to the direction of shear.

    - constrained_shear_with_lateral_displacement
        One face is not allowed any displacement at all. The opposite face
    moves parallel to the fixed face. The points on the moving face are allowed
    to move perpendicular to the direction of shear.

    - pure_shear
        One face is not allowed any displacement at all. A simple velocity
    boundary condition is applied to the opposite face with no restraint on any
    other displacement component.

    - uniaxial_compression
        The three faces of the cube lying on the coordinate planes are not
    allowed to move perpendicular to the coordinate planes they are lying on,
    but their motion parallel to their respective planes is not restricted. One
    face not on any of the coordinate planes is moved towards its opposite
    face. The whole cube experiences a spatially uniform deformation.

    */

    bool constrained_shear_no_lateral_displacement   = false;
    bool constrained_shear_with_lateral_displacement = false;
    bool pure_shear                                  = false;
    bool uniaxial_compression                        = false;

    /*constrained_shear_no_lateral_displacement   = true;*/
    /*constrained_shear_with_lateral_displacement = true;*/
    /*pure_shear                                  = true;*/
    uniaxial_compression                        = true;

    parameter_handler.enter_subsection("Domain Geometry");
    double height = parameter_handler.get_double("height");
    parameter_handler.leave_subsection();

    double nominal_strain_rate = 1;
    double top_surface_speed   = height * nominal_strain_rate;

    // Check that exactly one of the above set of boundary conditions is
    // applied to the domain.
    if (!(
        constrained_shear_no_lateral_displacement +
        constrained_shear_with_lateral_displacement +
        pure_shear +
        uniaxial_compression
        == 1)) {

        std::cout 
        << "Exactly one of the set of boundary conditions must be applied at point of time."
        << "Exiting program."
        << std::endl;
        exit(0);

    }

    // The following are three arrays of boolean values that tell the
    // interpolate_boundary_values function which component to apply the
    // boundary values to.
    const FEValuesExtractors::Scalar x_component(0);
    const FEValuesExtractors::Scalar y_component(1);
    const FEValuesExtractors::Scalar z_component(2);

    if (constrained_shear_no_lateral_displacement) {

        non_homogenous_constraints.clear();

        // The face on the xy plane has boundary indicator of 4 and must be kept
        // from moving in the x, y or z directions. The following three boundary
        // conditions ensure this. They can probably be rolled into one function
        // call.
        VectorTools::interpolate_boundary_values(
                                dof_handler,
                                4,
                                Functions::ZeroFunction<dim>(dim),
                                non_homogenous_constraints,
                                fe.component_mask(x_component));

        VectorTools::interpolate_boundary_values(
                                dof_handler,
                                4,
                                Functions::ZeroFunction<dim>(dim),
                                non_homogenous_constraints,
                                fe.component_mask(y_component));

        VectorTools::interpolate_boundary_values(
                                dof_handler,
                                4,
                                Functions::ZeroFunction<dim>(dim),
                                non_homogenous_constraints,
                                fe.component_mask(z_component));

        // The face opposite the xy plane has boundary indicator of 5. This must be
        // made to move parallel to the xy plane and must not change height
        VectorTools::interpolate_boundary_values(
                                dof_handler,
                                5,
                                Functions::ZeroFunction<dim>(dim),
                                non_homogenous_constraints,
                                fe.component_mask(x_component));

        VectorTools::interpolate_boundary_values(
                    dof_handler,
                    5,
                    Functions::ConstantFunction<dim>(delta_t*top_surface_speed, dim),
                    non_homogenous_constraints,
                    fe.component_mask(y_component));

        VectorTools::interpolate_boundary_values(
                                dof_handler,
                                5,
                                Functions::ZeroFunction<dim>(dim),
                                non_homogenous_constraints,
                                fe.component_mask(z_component));

        non_homogenous_constraints.close();


        homogenous_constraints.clear();

        VectorTools::interpolate_boundary_values(
                                dof_handler,
                                4,
                                Functions::ZeroFunction<dim>(dim),
                                homogenous_constraints,
                                fe.component_mask(x_component));

        VectorTools::interpolate_boundary_values(
                                dof_handler,
                                4,
                                Functions::ZeroFunction<dim>(dim),
                                homogenous_constraints,
                                fe.component_mask(y_component));

        VectorTools::interpolate_boundary_values(
                                dof_handler,
                                4,
                                Functions::ZeroFunction<dim>(dim),
                                homogenous_constraints,
                                fe.component_mask(z_component));

        VectorTools::interpolate_boundary_values(
                                dof_handler,
                                5,
                                Functions::ZeroFunction<dim>(dim),
                                homogenous_constraints,
                                fe.component_mask(x_component));

        VectorTools::interpolate_boundary_values(
                                dof_handler,
                                5,
                                Functions::ZeroFunction<dim>(dim),
                                homogenous_constraints,
                                fe.component_mask(y_component));

        VectorTools::interpolate_boundary_values(
                                dof_handler,
                                5,
                                Functions::ZeroFunction<dim>(dim),
                                homogenous_constraints,
                                fe.component_mask(z_component));

        homogenous_constraints.close();

        constraints_L2.clear();
        constraints_L2.close();

    } // End of conditional statement for constrained_shear_no_lateral_displacement

    if (constrained_shear_with_lateral_displacement) {

        non_homogenous_constraints.clear();

        // The face on the xy plane has boundary indicator of 4 and must be kept
        // from moving in the x, y or z directions. The following three boundary
        // conditions ensure this. They can probably be rolled into one function
        // call.
        VectorTools::interpolate_boundary_values(
                                dof_handler,
                                4,
                                Functions::ZeroFunction<dim>(dim),
                                non_homogenous_constraints,
                                fe.component_mask(x_component));

        VectorTools::interpolate_boundary_values(
                                dof_handler,
                                4,
                                Functions::ZeroFunction<dim>(dim),
                                non_homogenous_constraints,
                                fe.component_mask(y_component));

        VectorTools::interpolate_boundary_values(
                                dof_handler,
                                4,
                                Functions::ZeroFunction<dim>(dim),
                                non_homogenous_constraints,
                                fe.component_mask(z_component));

        // The face opposite the xy plane has boundary indicator of 5. This must be
        // made to move parallel to the xy plane and must not change height
        VectorTools::interpolate_boundary_values(
                    dof_handler,
                    5,
                    Functions::ConstantFunction<dim>(delta_t*top_surface_speed, dim),
                    non_homogenous_constraints,
                    fe.component_mask(y_component));

        VectorTools::interpolate_boundary_values(
                                dof_handler,
                                5,
                                Functions::ZeroFunction<dim>(dim),
                                non_homogenous_constraints,
                                fe.component_mask(z_component));

        non_homogenous_constraints.close();


        homogenous_constraints.clear();

        VectorTools::interpolate_boundary_values(
                                dof_handler,
                                4,
                                Functions::ZeroFunction<dim>(dim),
                                homogenous_constraints,
                                fe.component_mask(x_component));

        VectorTools::interpolate_boundary_values(
                                dof_handler,
                                4,
                                Functions::ZeroFunction<dim>(dim),
                                homogenous_constraints,
                                fe.component_mask(y_component));

        VectorTools::interpolate_boundary_values(
                                dof_handler,
                                4,
                                Functions::ZeroFunction<dim>(dim),
                                homogenous_constraints,
                                fe.component_mask(z_component));

        VectorTools::interpolate_boundary_values(
                                dof_handler,
                                5,
                                Functions::ZeroFunction<dim>(dim),
                                homogenous_constraints,
                                fe.component_mask(y_component));

        VectorTools::interpolate_boundary_values(
                                dof_handler,
                                5,
                                Functions::ZeroFunction<dim>(dim),
                                homogenous_constraints,
                                fe.component_mask(z_component));

        homogenous_constraints.close();

        constraints_L2.clear();
        constraints_L2.close();

    } // End of conditional statement for constrained_shear_with_lateral_displacement

    if (pure_shear) {

        non_homogenous_constraints.clear();

        // The face on the xy plane has boundary indicator of 4 and must be kept
        // from moving in the x, y or z directions. The following three boundary
        // conditions ensure this. They can probably be rolled into one function
        // call.
        VectorTools::interpolate_boundary_values(
                                dof_handler,
                                4,
                                Functions::ZeroFunction<dim>(dim),
                                non_homogenous_constraints,
                                fe.component_mask(x_component));

        VectorTools::interpolate_boundary_values(
                                dof_handler,
                                4,
                                Functions::ZeroFunction<dim>(dim),
                                non_homogenous_constraints,
                                fe.component_mask(y_component));

        VectorTools::interpolate_boundary_values(
                                dof_handler,
                                4,
                                Functions::ZeroFunction<dim>(dim),
                                non_homogenous_constraints,
                                fe.component_mask(z_component));

        // The face opposite the xy plane has boundary indicator of 5. This must be
        // made to move parallel to the xy plane and must not change height
        VectorTools::interpolate_boundary_values(
                    dof_handler,
                    5,
                    Functions::ConstantFunction<dim>(delta_t*top_surface_speed, dim),
                    non_homogenous_constraints,
                    fe.component_mask(y_component));

        non_homogenous_constraints.close();


        homogenous_constraints.clear();

        VectorTools::interpolate_boundary_values(
                                dof_handler,
                                4,
                                Functions::ZeroFunction<dim>(dim),
                                homogenous_constraints,
                                fe.component_mask(x_component));

        VectorTools::interpolate_boundary_values(
                                dof_handler,
                                4,
                                Functions::ZeroFunction<dim>(dim),
                                homogenous_constraints,
                                fe.component_mask(y_component));

        VectorTools::interpolate_boundary_values(
                                dof_handler,
                                4,
                                Functions::ZeroFunction<dim>(dim),
                                homogenous_constraints,
                                fe.component_mask(z_component));

        VectorTools::interpolate_boundary_values(
                                dof_handler,
                                5,
                                Functions::ZeroFunction<dim>(dim),
                                homogenous_constraints,
                                fe.component_mask(y_component));

        homogenous_constraints.close();

        constraints_L2.clear();
        constraints_L2.close();

    } // End of pure_shear

    if (uniaxial_compression) {

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
                        Functions::ConstantFunction<dim>(-delta_t*top_surface_speed, dim),
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

        constraints_L2.clear();
        constraints_L2.close();

    } // End of uniaxial_compression

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
    std::vector<std::shared_ptr<Material<dim>>> quadrature_point_history_data;

    Tensor<2, dim> F;    // Deformation gradient
    Tensor<2, dim> Finv; // Inverse of the deformation gradient

    double J; // Determinant(F)

    SymmetricTensor<2, dim> s; // Kirchhoff stress
    SymmetricTensor<4, dim> c; // Spatial tangent modulus * determinant(F)

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
        for (unsigned int q = 0; q < n_quadrature_points; ++q) {

            F = quadrature_point_history_data[q]->deformation_gradient;
            s = quadrature_point_history_data[q]->cauchy_stress;
            c = quadrature_point_history_data[q]->spatial_tangent_modulus;
            J = determinant(F);

            Finv = invert(F);

            for (unsigned int i = 0; i < dofs_per_cell; ++i) {

                const unsigned int ci = fe_values
                                        .get_fe()
                                        .system_to_component_index(i)
                                        .first;

                dphidx_i = 0.0; 

                // Transform the gradients of the shape functions returned
                // by dealii to the current configuration
                for (unsigned int m = 0; m < dim; ++m)
                    for (unsigned int n = 0; n < dim; ++n)
                        dphidx_i[m] += fe_values.shape_grad(i, q)[n] * Finv[n][m];

                for (unsigned int di = 0; di < dim; ++di)
                    cell_rhs(i) += -dphidx_i[di] * s[ci][di] * J * fe_values.JxW(q);

                for (unsigned int j = 0; j < dofs_per_cell; ++j) {

                    const unsigned int cj = fe_values
                                            .get_fe()
                                            .system_to_component_index(j)
                                            .first;

                    dphidx_j = 0.0; 

                    // Transform the gradients of the shape functions returned
                    // by dealii to the current configuration
                    for (unsigned int m = 0; m < dim; ++m)
                        for (unsigned int n = 0; n < dim; ++n)
                            dphidx_j[m] += fe_values.shape_grad(j, q)[n] * Finv[n][m];

                    for (unsigned int di = 0; di < dim; ++di) {
                        for (unsigned int dj = 0; dj < dim; ++dj) {
                            cell_matrix(i, j) +=
                                dphidx_i[di] *
                                c[ci][di][cj][dj] *
                                dphidx_j[dj] *
                                J * fe_values.JxW(q)
                                +
                                dphidx_i[di] *
                                delta[ci][cj] * s[di][dj] *
                                dphidx_j[dj] *
                                J * fe_values.JxW(q);
                        }
                    }
                } // End of j loop
            } // End of i loop
        } // End of quadrature loop

        // Distribute local contributions to global system
        cell->get_dof_indices(local_dof_indices);

        if (iterations == 0) {

            /*system_matrix.print(text_output_file, false, false);*/
            /*cell_matrix.print_formatted(text_output_file, 3, false, 0, "0");*/
            /*text_output_file << "cell_rhs     = " << cell_rhs << std::endl;;*/
            /*text_output_file << "delta_solution = " << delta_solution << std::endl;*/
            /*text_output_file << "solution       = " << solution << std::endl;*/
            /*text_output_file << std::endl;*/

            // Apply non-homogenous boundary conditions only in the first
            // iteration of the increment.
            non_homogenous_constraints.distribute_local_to_global(
                        cell_matrix,
                        cell_rhs,
                        local_dof_indices,
                        system_matrix,
                        system_rhs);

            /*system_matrix.print(text_output_file, false, false);*/
            /*cell_matrix.print_formatted(text_output_file, 3, false, 0, "0");*/
            /*text_output_file << "system_rhs     = " << system_rhs << std::endl;;*/
            /*text_output_file << "delta_solution = " << delta_solution << std::endl;*/
            /*text_output_file << "solution       = " << solution << std::endl;*/
            /*text_output_file << std::endl;*/

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

    } // End of loop over all cells

}

template <int dim>
void Problem<dim>::calculate_residual_norm () {

    residual = 0.0;

    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i) {
        if (!homogenous_constraints.is_constrained(i))
            residual(i) = system_rhs(i);
    }

    residual_norm = residual.l2_norm();

}

template <int dim>
void Problem<dim>::solve_linear_system () {

    delta_solution = 0.0;

    // The solver will do a maximum of 1000 iterations before giving up
    SolverControl solver_control(1000, 1e-50);
    SolverCG<Vector<double>> solver_cg(solver_control);
    solver_cg.solve(system_matrix,
                    delta_solution,
                    system_rhs,
                    IdentityMatrix(solution.size()));

    /*if (iterations == 0) */
    /*    non_homogenous_constraints.distribute(delta_solution);*/

    solution += delta_solution;

}

template <int dim>
void Problem<dim>::update_all_history_data () {

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

    // Temporary structure for holding the quadrature point data
    std::vector<std::shared_ptr<Material<dim>>> quadrature_point_history_data;

    // Variables for retrieving history variables and Cauchy stress from the
    // integration point level calculations.
    Tensor<2, dim> F_B;
    Tensor<2, dim> F_D;
    SymmetricTensor<2, dim> T_A;
    SymmetricTensor<4, dim> Jc;

    /*std::cout << "Printing from update function" << std::endl;*/

    for (auto &cell : dof_handler.active_cell_iterators()) {

        // Initialize the fe_values object with values relevant to the current cell
        fe_values.reinit(cell);

        // Get displacement gradients at all integration points of the cell
        // from dealii
        fe_values.get_function_gradients(solution, solution_gradients);

        quadrature_point_history_data = quadrature_point_history.get_data(cell);

        for (unsigned int q = 0; q < quadrature_formula.size(); ++q) {

            for (unsigned int i = 0; i < dim; ++i)
                for (unsigned int j = 0; j < dim; ++j)
                    dUdX[i][j] = solution_gradients[q][i][j];

            F = Physics::Elasticity::Kinematics::F(dUdX);

            // Set deformation gradient of quadrature point to new deformation
            // gradient. For time t = 0, this is calculated as the identity
            // matrix.
            quadrature_point_history_data[q]->deformation_gradient = F;

            // For history dependent materials, the time step length is
            // required as input for the material variables and the tangent
            // modulus
            quadrature_point_history_data[q]->delta_t = delta_t;
            quadrature_point_history_data[q]->perform_constitutive_update();
            quadrature_point_history_data[q]->compute_spatial_tangent_modulus();

            /*if (q == 0) {*/
            /*    std::cout << "F = " << F << std::endl;*/
            /*    std::cout << "sigma = " << quadrature_point_history_data[q]->cauchy_stress << std::endl;*/
            /*}*/

        } // End of loop over quadrature points
    }
}    

template <int dim>
void Problem<dim>::perform_L2_projections () {

    // This function projects quadrature point data to the nodes of the
    // triangulation. The vectors containing the projected data are written by
    // the output_results function to the output vtu file for making
    // temperature plots.

    mass_matrix_L2 = 0.0;
    nodal_output_L2.clear();

    unsigned int no_of_dofs_L2 = dof_handler_L2.n_dofs();

    const unsigned int dofs_per_cell       = fe_L2.n_dofs_per_cell();
    const unsigned int n_quadrature_points = fe_values_L2.n_quadrature_points;

    // Contributions from this cell level matrix are used to buld the global
    // mass matrix
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

    // Variables named *rhs_L2 are built from quadrature point data
    // Variables named *projection_L2 contain the projected, nodal data
    // Variables named *cell_rhs are used to build variables named *rhs_L2

    Vector<double> sigma_xx_rhs_L2(no_of_dofs_L2);
    Vector<double> sigma_xx_projection_L2(no_of_dofs_L2);
    Vector<double> sigma_xx_cell_rhs(dofs_per_cell);

    Vector<double> sigma_xy_rhs_L2(no_of_dofs_L2);
    Vector<double> sigma_xy_projection_L2(no_of_dofs_L2);
    Vector<double> sigma_xy_cell_rhs(dofs_per_cell);

    Vector<double> sigma_xz_rhs_L2(no_of_dofs_L2);
    Vector<double> sigma_xz_projection_L2(no_of_dofs_L2);
    Vector<double> sigma_xz_cell_rhs(dofs_per_cell);

    Vector<double> sigma_yy_rhs_L2(no_of_dofs_L2);
    Vector<double> sigma_yy_projection_L2(no_of_dofs_L2);
    Vector<double> sigma_yy_cell_rhs(dofs_per_cell);

    Vector<double> sigma_yz_rhs_L2(no_of_dofs_L2);
    Vector<double> sigma_yz_projection_L2(no_of_dofs_L2);
    Vector<double> sigma_yz_cell_rhs(dofs_per_cell);

    Vector<double> sigma_zz_rhs_L2(no_of_dofs_L2);
    Vector<double> sigma_zz_projection_L2(no_of_dofs_L2);
    Vector<double> sigma_zz_cell_rhs(dofs_per_cell);

    // types::global_dof_index is an unsigned int of 32 bits on most systems.
    // So the following is an array of integers.
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Object for temporarily storing the quadrature point data while
    // performing quadrature over the current cell
    std::vector<std::shared_ptr<Material<dim>>> quadrature_point_history_data;

    Tensor<2, dim> F;    // Deformation gradient

    for (const auto &cell : dof_handler_L2.active_cell_iterators()) {

        fe_values_L2.reinit(cell);

        cell_matrix = 0.0;

        sigma_xx_cell_rhs = 0.0;
        sigma_xy_cell_rhs = 0.0;
        sigma_xz_cell_rhs = 0.0;
        sigma_yy_cell_rhs = 0.0;
        sigma_yz_cell_rhs = 0.0;
        sigma_zz_cell_rhs = 0.0;

        quadrature_point_history_data = quadrature_point_history.get_data(cell);

        for (unsigned int q = 0; q < n_quadrature_points; ++q) {

            F = quadrature_point_history_data[q]->deformation_gradient;
            
            double J = determinant(F);

            for (unsigned int i = 0; i < dofs_per_cell; ++i) {

                sigma_xx_cell_rhs(i) +=
                    fe_values_L2.shape_value(i, q) * 
                    quadrature_point_history_data[q]->cauchy_stress[0][0] *
                    fe_values_L2.JxW(q); 

                sigma_xy_cell_rhs(i) +=
                    fe_values_L2.shape_value(i, q) * 
                    quadrature_point_history_data[q]->cauchy_stress[0][1] *
                    fe_values_L2.JxW(q); 

                sigma_xz_cell_rhs(i) +=
                    fe_values_L2.shape_value(i, q) * 
                    quadrature_point_history_data[q]->cauchy_stress[0][2] *
                    fe_values_L2.JxW(q); 

                sigma_yy_cell_rhs(i) +=
                    fe_values_L2.shape_value(i, q) * 
                    quadrature_point_history_data[q]->cauchy_stress[1][1] *
                    fe_values_L2.JxW(q); 

                sigma_yz_cell_rhs(i) +=
                    fe_values_L2.shape_value(i, q) * 
                    quadrature_point_history_data[q]->cauchy_stress[1][2] *
                    fe_values_L2.JxW(q); 

                sigma_zz_cell_rhs(i) +=
                    fe_values_L2.shape_value(i, q) * 
                    quadrature_point_history_data[q]->cauchy_stress[2][2] *
                    fe_values_L2.JxW(q); 

                for (unsigned int j = 0; j < dofs_per_cell; ++j) {

                    cell_matrix(i, j) +=
                        fe_values_L2.shape_value(i, q) * 
                        fe_values_L2.shape_value(j, q) *
                        J * fe_values_L2.JxW(q); 

                } // End of j loop
            } // End of i loop
        } // Quadrature loop

        cell->get_dof_indices(local_dof_indices);

        constraints_L2.distribute_local_to_global(
                    cell_matrix,
                    local_dof_indices,
                    mass_matrix_L2);

        constraints_L2.distribute_local_to_global(
                    sigma_xx_cell_rhs,
                    local_dof_indices,
                    sigma_xx_rhs_L2);

        constraints_L2.distribute_local_to_global(
                    sigma_xy_cell_rhs,
                    local_dof_indices,
                    sigma_xy_rhs_L2);

        constraints_L2.distribute_local_to_global(
                    sigma_xz_cell_rhs,
                    local_dof_indices,
                    sigma_xz_rhs_L2);

        constraints_L2.distribute_local_to_global(
                    sigma_yy_cell_rhs,
                    local_dof_indices,
                    sigma_yy_rhs_L2);

        constraints_L2.distribute_local_to_global(
                    sigma_yz_cell_rhs,
                    local_dof_indices,
                    sigma_yz_rhs_L2);

        constraints_L2.distribute_local_to_global(
                    sigma_zz_cell_rhs,
                    local_dof_indices,
                    sigma_zz_rhs_L2);

    } // End of loop over all cells to make the mass matrix

    // The solver will do a maximum of 1000 iterations before giving up
    SolverControl solver_control(1000, 1e-12);
    SolverCG<Vector<double>> solver_cg(solver_control);
    solver_cg.solve(mass_matrix_L2,
                    sigma_xx_projection_L2,
                    sigma_xx_rhs_L2,
                    IdentityMatrix(no_of_dofs_L2));

    solver_cg.solve(mass_matrix_L2,
                    sigma_xy_projection_L2,
                    sigma_xy_rhs_L2,
                    IdentityMatrix(no_of_dofs_L2));

    solver_cg.solve(mass_matrix_L2,
                    sigma_xz_projection_L2,
                    sigma_xz_rhs_L2,
                    IdentityMatrix(no_of_dofs_L2));

    solver_cg.solve(mass_matrix_L2,
                    sigma_yy_projection_L2,
                    sigma_yy_rhs_L2,
                    IdentityMatrix(no_of_dofs_L2));

    solver_cg.solve(mass_matrix_L2,
                    sigma_yz_projection_L2,
                    sigma_yz_rhs_L2,
                    IdentityMatrix(no_of_dofs_L2));

    solver_cg.solve(mass_matrix_L2,
                    sigma_zz_projection_L2,
                    sigma_zz_rhs_L2,
                    IdentityMatrix(no_of_dofs_L2));

    nodal_output_L2.push_back(sigma_xx_projection_L2);
    nodal_output_L2.push_back(sigma_xy_projection_L2);
    nodal_output_L2.push_back(sigma_xz_projection_L2);
    nodal_output_L2.push_back(sigma_yy_projection_L2);
    nodal_output_L2.push_back(sigma_yz_projection_L2);
    nodal_output_L2.push_back(sigma_zz_projection_L2);

}

template <int dim>
void Problem<dim>::output_results () {

    // Refresh data_out object to write data from current time step.
    data_out.clear_data_vectors();

    // -------------------------------------------------------------------------

    // Output data directly from DOFs
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(
    dim, DataComponentInterpretation::component_is_part_of_vector);

    std::vector<std::string> solution_name(dim, "displacement");

    data_out.add_data_vector(
                        dof_handler,
                        solution, 
                        solution_name,
                        data_component_interpretation);

    // -------------------------------------------------------------------------

    // Output data from L2 projection
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation_L2;

    data_component_interpretation_L2
        .push_back(DataComponentInterpretation::component_is_scalar);

    data_out.add_data_vector(
                        dof_handler_L2,
                        nodal_output_L2[0], 
                        "sigma_xx",
                        data_component_interpretation_L2);

    data_out.add_data_vector(
                        dof_handler_L2,
                        nodal_output_L2[1], 
                        "sigma_xy",
                        data_component_interpretation_L2);

    data_out.add_data_vector(
                        dof_handler_L2,
                        nodal_output_L2[2], 
                        "sigma_xz",
                        data_component_interpretation_L2);

    data_out.add_data_vector(
                        dof_handler_L2,
                        nodal_output_L2[3], 
                        "sigma_yy",
                        data_component_interpretation_L2);

    data_out.add_data_vector(
                        dof_handler_L2,
                        nodal_output_L2[4], 
                        "sigma_yz",
                        data_component_interpretation_L2);

    data_out.add_data_vector(
                        dof_handler_L2,
                        nodal_output_L2[5], 
                        "sigma_zz",
                        data_component_interpretation_L2);

    // -------------------------------------------------------------------------

    // All data added to data_out object. Now send to vtu file. The following
    // mapping object allows one to view the deformed shape of the domain.
    const MappingQEulerian<dim> q_mapping(fe.degree, dof_handler, solution);

    data_out.build_patches(q_mapping, fe.degree);;

    std::string output_file_name = 
            "solution/solution-" 
            + std::to_string(step_number)
            + ".vtu";

    std::ofstream output_file(output_file_name);

    data_out.write_vtu(output_file);

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
