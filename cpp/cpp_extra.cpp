#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <sstream>
#include <iostream>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/png/writePNG.h>
#include <igl/png/readPNG.h>
#include <igl/marching_cubes.h>
#include "pcg_solver.h"

// Eigen::MatrixXd V;
// Eigen::MatrixXi F;

// int main(int argc, char *argv[])
// {
//   // Load a mesh in OFF format
//   igl::readOFF(TUTORIAL_SHARED_PATH "/bunny.off", V, F);

//   // Plot the mesh
//   igl::opengl::glfw::Viewer viewer;
//   viewer.data().set_mesh(V, F);
//   viewer.launch();
// }
const int MODE_FREESURFACE = 0;
const int MODE_TAGGED = 1;
struct Viewer
{
    igl::opengl::glfw::Viewer inner;
    int particle_count;
    Eigen::MatrixXd P;
    Eigen::MatrixXd C;
    Eigen::MatrixXd V;
    Eigen::MatrixXd mV;
    Eigen::MatrixXi mF;
    std::vector<double> mV_data;
    std::vector<int> mF_data;
    std::vector<float> pos_buf;
    std::vector<float> vel_buf;
    std::vector<int> tags;
    bool updated = false;
    bool mesh_loaded = false;
    int capture_cnt = 0;
    bool record_screen = false;
    Viewer(int particle_count) : particle_count(particle_count)
    {
        P.resize(particle_count, 3);
        C.resize(particle_count, 3);
        V.resize(particle_count, 3);
        pos_buf.resize(particle_count * 3);
        vel_buf.resize(particle_count * 3);
        tags.resize(particle_count, 0);
    }
};
extern "C"
{
    void *create_viewer(size_t particle_count)
    {
        auto viewer = new Viewer(particle_count);
        viewer->inner.callback_key_pressed = [=](igl::opengl::glfw::Viewer &v, unsigned char key, int modifier) -> bool
        {
            if (key == '1' || key == '2')
            {
                // Allocate temporary buffers
                Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> R(1280, 800);
                Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> G(1280, 800);
                Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> B(1280, 800);
                Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> A(1280, 800);

                // Draw the scene in the buffers
                v.core().draw_buffer(
                    v.data(), false, R, G, B, A);
                std::ostringstream os;
                os << "output/out" << viewer->capture_cnt++ << ".png";
                auto file = os.str();
                // Save it to a PNG
                if (key == '2')
                {
                    A.fill(255);
                }
                igl::png::writePNG(R, G, B, A, file);
            }
            if (key == '3')
            {
                viewer->record_screen = !viewer->record_screen;
            }
            return false;
        };
        viewer->inner.callback_post_draw = [=](igl::opengl::glfw::Viewer &) -> bool
        {
            if (viewer->updated)
            {
                viewer->updated = false;
                auto &P = viewer->P;
                auto &C = viewer->C;
                auto &V = viewer->V;
                auto &tags = viewer->tags;
                auto &points = viewer->pos_buf;
                auto &vel = viewer->vel_buf;
                for (size_t i = 0; i < viewer->particle_count; i++)
                {

                    P.row(i) = Eigen::RowVector3d(points[3 * i + 0], points[3 * i + 1], points[3 * i + 2]);
                    Eigen::RowVector3d v = Eigen::RowVector3d(vel[3 * i + 0], vel[3 * i + 1], vel[3 * i + 2]);
                    V.row(i) = v * 0.3;
                    Eigen::RowVector3d default_color = Eigen::RowVector3d(0.1, 0.1, 0.6) + Eigen::RowVector3d(1, 1, 1) * v.norm() / 3.0;
                    if (tags[i] == 0)
                    {
                        C.row(i) = default_color;
                    }
                    else if (tags[i] == 1)
                    {
                        C.row(i) = Eigen::RowVector3d(0.3, 0.1, 0.1) + Eigen::RowVector3d(1, 0, 0) * v.norm() / 3.0;
                    }
                    else if (tags[i] == 2)
                    {
                        C.row(i) = Eigen::RowVector3d(0.1, 0.1, 0.3) + Eigen::RowVector3d(0, 0, 1) * v.norm() / 3.0;
                    }
                }
                viewer->inner.data().point_size = 4;
                viewer->inner.data().set_points(P, C);
                // viewer->inner.data().set_edges_from_vector_field(P, V, Eigen::RowVector3d(1,0,0));
                if (viewer->record_screen)
                {
                    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> R(1280, 800);
                    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> G(1280, 800);
                    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> B(1280, 800);
                    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> A(1280, 800);

                    // Draw the scene in the buffers
                    viewer->inner.core().draw_buffer(
                        viewer->inner.data(), false, R, G, B, A);
                    std::ostringstream os;
                    os << "output/out" << viewer->capture_cnt++ << ".png";
                    auto file = os.str();
                    // Save it to a PNG
                    A.fill(255);
                    igl::png::writePNG(R, G, B, A, file);
                }
            }

            // if(viewer->mesh_loaded) {
            //     viewer->inner.data().set_mesh(viewer->mV, viewer->mF);
            // }
            return false;
        };

        return viewer;
    }
    void launch_viewer(void *viewer_)
    {
        auto viewer = (Viewer *)viewer_;
        viewer->inner.core().is_animating = true;
        viewer->inner.core().background_color = Eigen::Vector4f(0.1, 0.1, 0.1, 1.0);
        viewer->inner.launch();
    }
    bool viewer_load_mesh(void *viewer_, const char *path, const float *translate_scale, int *nV, int *nF)
    {
        auto viewer = (Viewer *)viewer_;
        if (!igl::readOBJ(path, viewer->mV, viewer->mF))
        {
            return false;
        }
        Eigen::RowVector3d translate(translate_scale[0], translate_scale[1], translate_scale[2]);
        // Eigen::RowVector3d scale(translate_scale[3], translate_scale[4],translate_scale[5]);
        Eigen::Matrix3d scale;
        scale << translate_scale[3], 0, 0, 0, translate_scale[4], 0, 0, 0, translate_scale[5];
        for (int i = 0; i < viewer->mV.rows(); i++)
        {
            Eigen::RowVector3d row = viewer->mV.row(i);
            viewer->mV.row(i) = (scale * row.transpose()).transpose() + translate;
            row = viewer->mV.row(i);
            viewer->mV_data.push_back(row[0]);
            viewer->mV_data.push_back(row[1]);
            viewer->mV_data.push_back(row[2]);
        }
        for (int i = 0; i < viewer->mF.rows(); i++)
        {
            Eigen::RowVector3i row = viewer->mF.row(i);
            viewer->mF_data.push_back(row[0]);
            viewer->mF_data.push_back(row[1]);
            viewer->mF_data.push_back(row[2]);
        }
        viewer->mesh_loaded = true;
        *nV = viewer->mV.rows();
        *nF = viewer->mF.rows();
        printf("loaded %d vertices %d faces\n", *nV, *nF);
        viewer->inner.data().set_mesh(viewer->mV, viewer->mF);
        return true;
    }
    const double *viewer_mesh_vertices(void *viewer_)
    {
        auto viewer = (Viewer *)viewer_;
        return viewer->mV_data.data();
    }
    const int *viewer_mesh_faces(void *viewer_)
    {
        auto viewer = (Viewer *)viewer_;
        return viewer->mF_data.data();
    }
    void viewer_set_tags(void *viewer_, const int *tags)
    {
        auto viewer = (Viewer *)viewer_;
        std::memcpy(viewer->tags.data(), tags, sizeof(int) * viewer->particle_count);
    }
    void viewer_set_points(void *viewer_, const float *points, const float *velocity)
    {
        auto viewer = (Viewer *)viewer_;
        viewer->updated = true;
        std::memcpy(viewer->pos_buf.data(), points, sizeof(float) * viewer->particle_count * 3);
        std::memcpy(viewer->vel_buf.data(), velocity, sizeof(float) * viewer->particle_count * 3);
        // auto &P = viewer->P;
        // for (int i = 0; i < viewer->particle_count; i++)
        // {
        //     P.row(i) = Eigen::RowVector3d(points[3 * i + 0], points[3 * i + 1], points[3 * i + 2]);
        // }
        // viewer->inner.data().set_points(P, Eigen::RowVector3d(1, 1, 1));
    }
    void destroy_viewer(void *viewer_)
    {
        auto viewer = (Viewer *)viewer_;
        delete viewer;
    }
    // solve N x N linear system with a given stencil
    void bridson_pcg_solve(int nx, int ny, int nz, const float *stencil, const int *offsets, int noffsets, const float *b, float *out)
    {
        auto N = nx * ny * nz;
        robertbridson::PCGSolver<scalar> solver;
        robertbridson::SparseMatrix<scalar> matrix(N);
        std::vector<float> rhs(N), pressure(N);
        std::memcpy(rhs.data(), b, sizeof(float) * N);
        for (int z = 0; z < nz; ++z)
        {
            for (int y = 0; y < ny; ++y)
            {
                for (int x = 0; x < nx; ++x)
                {
                    int i = x + y * nx + z * nx * ny;
                    // printf("%f\n", stencil[0]);
                    // triplets.push_back(Eigen::Triplet<float>(i, i, stencil[i * noffsets]));
                    matrix.add_to_element(i, i, stencil[i * noffsets]);
                    for (int j = 1; j < noffsets; ++j)
                    {
                        auto off_x = offsets[j * 3 + 0];
                        auto off_y = offsets[j * 3 + 1];
                        auto off_z = offsets[j * 3 + 2];
                        // fprintf(stderr, "off_x: %d, off_y: %d, off_z: %d\n", off_x, off_y, off_z);
                        int idx = i + off_x + off_y * nx + off_z * nx * ny;
                        if (x + off_x >= 0 && x + off_x < nx &&
                            y + off_y >= 0 && y + off_y < ny &&
                            z + off_z >= 0 && z + off_z < nz)
                        {
                            // triplets.push_back(Eigen::Triplet<float>(i, idx, stencil[i * noffsets + j]));
                            matrix.add_to_element(i, idx, stencil[i * noffsets + j]);
                        }
                    }
                }
            }
        }
        scalar residual;
        solver.set_solver_parameters(1e-5, 1024, 0.97, 0.25);
        int iterations;
        bool success = solver.solve(matrix, rhs, pressure, residual, iterations);
        if (!success)
        {
            std::cout << "WARNING: Pressure solve failed! residual = " << residual << ", iters = " << iterations << std::endl;
        }
        std::cout << "solve finished in " << iterations << " iterations" << std::endl;
        std::memcpy(out, pressure.data(), sizeof(float) * N);
    }
    void compute_G(const float *A_, uint32_t N, float h, float *G_, float *G_norm)
    {
        Eigen::Map<const Eigen::Matrix3f> A(A_);
        // std::cout <<A << std::endl << std::endl;
        Eigen::Map<Eigen::Matrix3f> G(G_);
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3f R = svd.matrixU();
        Eigen::Vector3f sigmas = svd.singularValues();
        const auto kr = 4.0;
        const auto ks = 1400.0;
        const auto kn = 0.5;
        const int Ne = 25;
        float sigma1 = sigmas[0];
        for (int i = 1; i < 3; i++) {
            sigmas[i] = std::fmax(sigmas[i], sigma1 / kr);
        }
        Eigen::DiagonalMatrix<float, 3> Sigma;
        if (N < Ne) {
            Sigma = kn * Eigen::Vector3f::Ones().asDiagonal();
        } else {
            Sigma = ks * sigmas.asDiagonal();
        }
        G = 1.0 / h * R * Sigma.inverse() * R.transpose();
        *G_norm = G.norm();
    }
    void save_reconstructed_mesh(uint32_t nx, uint32_t ny, uint32_t nz, const float *S_, const float *GV_, float iso, const char *path)
    {
        Eigen::Map<const Eigen::VectorXf> S(S_, nx * ny * nz);
        Eigen::Map<const Eigen::MatrixXf> GV_T(GV_, 3, nx * ny * nz);
        Eigen::MatrixXf GV = GV_T.transpose();
        Eigen::MatrixXf V;
        Eigen::MatrixXi F;
        igl::marching_cubes(S, GV, nx, ny, nz, iso, V, F);
        igl::writeOBJ(path, V, F);
    }
    // solve N x N linear system with a given stencil
    void eigen_pcg_solve(int nx, int ny, int nz, const float *stencil, const int *offsets, int noffsets, const float *b, float *out)
    {
        std::vector<Eigen::Triplet<float>> triplets;
        auto N = nx * ny * nz;
        // printf("%d %d %d\n",nx,ny,nz);
        // std::unordered_map<
        for (int z = 0; z < nz; ++z)
        {
            for (int y = 0; y < ny; ++y)
            {
                for (int x = 0; x < nx; ++x)
                {
                    int i = x + y * nx + z * nx * ny;
                    // printf("%f\n", stencil[0]);
                    triplets.push_back(Eigen::Triplet<float>(i, i, stencil[i * noffsets]));
                    for (int j = 1; j < noffsets; ++j)
                    {
                        auto off_x = offsets[j * 3 + 0];
                        auto off_y = offsets[j * 3 + 1];
                        auto off_z = offsets[j * 3 + 2];
                        // fprintf(stderr, "off_x: %d, off_y: %d, off_z: %d\n", off_x, off_y, off_z);
                        int idx = i + off_x + off_y * nx + off_z * nx * ny;
                        if (x + off_x >= 0 && x + off_x < nx &&
                            y + off_y >= 0 && y + off_y < ny &&
                            z + off_z >= 0 && z + off_z < nz)
                        {
                            triplets.push_back(Eigen::Triplet<float>(i, idx, stencil[i * noffsets + j]));
                        }
                    }
                }
            }
        }
        Eigen::SparseMatrix<float> A(N, N);
        A.setFromTriplets(triplets.begin(), triplets.end());
        // Eigen::SimplicialLLT<Eigen::SparseMatrix<float>> lltOfA(A); // compute the Cholesky decomposition of A
        // if(lltOfA.info() == Eigen::NumericalIssue)
        // {
        //     fprintf(stderr, "Possibly non semi-positive definitie matrix!\n");
        //     // std::cerr << A << std::endl;
        //     // abort();
        // }
        Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower | Eigen::Upper> cg;
        cg.setMaxIterations(2000);
        cg.compute(A);
        Eigen::Map<const Eigen::VectorXf> bmap(b, N);
        Eigen::Map<Eigen::VectorXf> outmap(out, N);
        outmap = cg.solve(bmap);

        // std::cout << Eigen::MatrixXf(A) << std::endl;
        // Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower|Eigen::Upper> solver2;
        // solver2.compute(A);
        // Eigen::SimplicialLLT<Eigen::SparseMatrix<float>> solver;
        // solver.compute(A);
        // Eigen::Map<const Eigen::VectorXf> bmap(b, N);
        // Eigen::Map<Eigen::VectorXf> outmap(out, N);
        // std::cout <<  bmap << std::endl;
        // std::cout << "solution:" << std::endl;
        // outmap = solver.solve(bmap);
        //  std::cout <<  outmap << std::endl;

        // Eigen::VectorXf out2 = solver2.solve(bmap);
        // std::cout << "solution2:" << std::endl;
        //  std::cout <<  out2 << std::endl;
        // Eigen::VectorXf residual = (A* outmap) - bmap;
        // std::cout << "residual:" << std::endl;
        // std::cout <<  residual << std::endl;

        //  Eigen::VectorXf residual2 = (A* out2) - bmap;
        // std::cout << "residual2:" << std::endl;
        // std::cout <<  residual2 << std::endl;
        printf("iterations = %ld\n", cg.iterations());
    }
}