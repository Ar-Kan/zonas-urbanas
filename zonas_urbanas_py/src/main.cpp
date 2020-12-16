#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <locale>

#include <Eigen/Dense>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/Eigen.h>
namespace py = pybind11;

#define ARMA_DONT_USE_WRAPPER
#define ARMA_USE_BLAS
//#define ARMA_DONT_KUSE_LAPACK
#define ARMA_USE_LAPACK
//#define ARMA_USE_HDF5
#include <armadillo>
#include <armadillo>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

arma::sp_mat open_matrix_market_file(const char* filename, long long unsigned int mat_dim, int ignore_lines_up_to = 0) {
    std::vector<long long unsigned int> location_u;
    std::vector<long long unsigned int> location_m;
    std::vector<double> values;

    std::ifstream file(filename);
    int a, b, c;

    int ignore = 0;
    std::string linha, campo;
    for (size_t i = 0; i < ignore_lines_up_to; i++) {
        std::cout << "ignorando linha " << i << std::endl;
        getline(file, linha);
    }
    while (file >> a >> b >> c) {
        /*
        * In Armadillo matrix indices start at 0 (due to C++ conventions),
        * while in the matrix market file they start at 1
        */
        location_u.push_back(a - 1);
        location_m.push_back(b - 1);
        values.push_back(c);
    }

    arma::umat lu(location_u);
    arma::umat lm(location_m);
    arma::umat location(join_rows(lu, lm).t());

    arma::sp_mat V(location, arma::vec(values), mat_dim, mat_dim);
    return V;
}


arma::mat open_csv_file(const std::string& filename, const std::string& delimeter = ",")
{
    std::ifstream csv(filename);
    std::vector<std::vector<double>> datas;

    for (std::string line; std::getline(csv, line); ) {

        std::vector<double> data;

        // split string by delimeter
        auto start = 0U;
        auto end = line.find(delimeter);
        while (end != std::string::npos) {
            data.push_back(std::stod(line.substr(start, end - start)));
            start = end + delimeter.length();
            end = line.find(delimeter, start);
        }
        data.push_back(std::stod(line.substr(start, end)));
        datas.push_back(data);
    }

    arma::mat data_mat = arma::zeros<arma::mat>(datas.size(), datas[0].size());

    for (int i = 0; i < datas.size(); i++) {
        arma::mat r(datas[i]);
        data_mat.row(i) = r.t();
    }

    return data_mat;
}


Eigen::MatrixXd converter_matriz(arma::Mat<double>& arma_A) {
    Eigen::MatrixXd eigen_B = Eigen::Map<Eigen::MatrixXd>(
        arma_A.memptr(),
        arma_A.n_rows,
        arma_A.n_cols
        );
    return eigen_B;
}

arma::Mat<double> converter_matriz(Eigen::Ref<Eigen::MatrixXd> eigen_A) {
    arma::Mat<double> arma_B = arma::Mat<double>(
        eigen_A.data(), eigen_A.rows(), eigen_A.cols(),
        true, false
        );
    return arma_B;
}

arma::Mat<double> converter_matriz(py::array_t<double>& array_A) {
    Eigen::MatrixXd arma_B = array_A.cast<Eigen::MatrixXd>();
    return converter_matriz(arma_B);
}

arma::Mat<double> converter_matriz(arma::vec& arma_A, int nrow, int ncol) {
    arma::mat arma_B(arma_A);
    arma_B.reshape(nrow, ncol);
    return arma_B;
}


bool eh_simetrica(py::array_t<double>& np_array) {
    arma::Mat<double> mat = converter_matriz(np_array);
    return mat.is_symmetric();
}


class EigenClass {
public:
    EigenClass(py::array_t<double>& np_array) {
        std::cout << "Convertendo matriz" << std::endl;
        this->_matriz_densa = converter_matriz(np_array);
        std::cout << "Matriz convertida" << std::endl;
    }

    EigenClass(const char* local_arquivo, long long unsigned int dimensao, int ignorar_ate_linha = 0) {
        arma::sp_mat arquivo_matriz_sparsa = open_matrix_market_file(local_arquivo, dimensao, ignorar_ate_linha);
        this->_matriz_densa = arma::mat(arquivo_matriz_sparsa);
    }

    bool calcular() {
        std::cout << "Calculando..." << std::endl;

        auto t1 = std::chrono::high_resolution_clock::now();
        bool ok = arma::eig_sym(
            this->_auto_valores,
            this->_auto_vetores,
            this->_matriz_densa
        );
        auto t2 = std::chrono::high_resolution_clock::now();

        auto duracao = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1.0e+9;
        std::cout << "O calculo levou: " << duracao << " segundos" << std::endl;
        return ok;
    }

    const Eigen::MatrixXd matriz() {
        return converter_matriz(this->_matriz_densa);
    }

    const Eigen::MatrixXd auto_valores() {
        arma::Mat<double> mat_auto_valores = converter_matriz(
            this->_auto_valores, 1, this->_auto_valores.size()
        );
        return converter_matriz(mat_auto_valores);
    }

    const Eigen::MatrixXd auto_vetores() {
        return converter_matriz(this->_auto_vetores);
    }

private:
    arma::Mat<double> _matriz_densa;
    arma::vec _auto_valores;
    arma::Mat<double> _auto_vetores;
};


class KMeans {
public:
    KMeans(py::array_t<double>& data, int n_clusters, std::string seed_mode = "random_subset", int n_iter = 10, bool verbose = true, int seed = 42) {
        this->_dados = converter_matriz(data);
        this->n_clusters = n_clusters;
        this->seed_mode = seed_mode;
        this->n_iter = n_iter;
        this->print_mode = verbose;
        this->seed = seed;
    }

    const bool calcular() {
        arma::arma_rng::set_seed(this->seed);

        bool ok = arma::kmeans(
            this->means,
            this->_dados,
            this->n_clusters,
            this->retorna_seed_mode(this->seed_mode),
            this->n_iter,
            this->print_mode
        );

        return ok;
    }

    const Eigen::MatrixXd resultado() {
        return converter_matriz(this->means);
    }

private:
    arma::Mat<double> _dados;
    arma::mat means;
    int n_clusters;
    std::string seed_mode;
    int n_iter;
    bool print_mode;
    int seed;

    arma::gmm_seed_mode retorna_seed_mode(std::string seed_mode) {
        if (seed_mode == "keep_existing") {
            return arma::keep_existing;
        }
        else if (seed_mode == "static_subset") {
            return arma::static_subset;
        }
        else if (seed_mode == "random_subset") {
            return arma::random_subset;
        }
        else if (seed_mode == "static_spread") {
            return arma::static_spread;
        }
        else if (seed_mode == "random_spread") {
            return arma::random_spread;
        }
        else {
            throw std::invalid_argument("O valor \"" + seed_mode + "\" do parametro \"seed_mode\" e invalido");
        }
    }
};

PYBIND11_MODULE(zonas_urbanas, m) {
    m.doc() = R"pbdoc(Pybind11, operacoes em matrizes na analise de grafos)pbdoc";

    m.def("eh_simetrica", &eh_simetrica, R"pbdoc(Indica se a matriz e simetrica ou nao)pbdoc", py::arg("np_array"));

    py::class_<EigenClass>(m, "Eigen")
        .def(py::init<py::array_t<double>&>(), py::arg("np_array"))
        .def(
            py::init<const char*, long long unsigned int, int>(),
            py::arg("local_arquivo"), py::arg("dimensao"), py::arg("ignorar_ate_linha") = 0,
            R"pbdoc(Le arquivos .mtx, para casos onde a matriz e muito grande)pbdoc"
        )
        .def("calcular", &EigenClass::calcular, "Executa o calculo dos autovetores e autovalores")
        .def("matriz", &EigenClass::matriz, "Retorna matriz original")
        .def("auto_valores", &EigenClass::auto_valores, "Retorna autovalores")
        .def("auto_vetores", &EigenClass::auto_vetores, "Retorna autovetores");

    py::class_<KMeans>(m, "KMeans")
        .def(
            py::init<py::array_t<double>&, int, std::string, int, bool, int>(),
            py::arg("data"), py::arg("n_clusters"),
            py::arg("seed_mode") = "random_subset", py::arg("n_iter") = 10, py::arg("verbose") = true, py::arg("seed") = 42,
            R"pbdoc(
            Cluster given data into k disjoint sets

            Parameters
            ----------
                data: input data matrix, with each sample stored as a column vector
                n_clusters: number of centroids; the number of samples in the data matrix should be much larger than n_clusters
                seed_mode: specifies how the initial centroids are seeded; it is one of:
                    keep_existing: use the centroids specified in the means matrix as the starting point
                    static_subset: use a subset of the data vectors (repeatable)
                    random_subset (default): use a subset of the data vectors (random)
                    static_spread: use a maximally spread subset of data vectors (repeatable)
                    random_spread: use a maximally spread subset of data vectors (random start)

                    caveat: seeding the initial centroids with static_spread and random_spread can be much more time consuming than with static_subset and random_subset
                n_iter (default = 10): number of clustering iterations; this is data dependent, but about 10 is typically sufficient
                verbose (default = true): parameter is either true or false, indicating whether progress is printed during clustering
                seed (default = 42): integer value for random number generator

            If the clustering fails, the result matrix is reset and a bool set to false is returned
            )pbdoc"
        )
        .def("calcular", &KMeans::calcular, "Executa o calculo dos clusters", R"pbdoc(
        Returns
        -------
        bool
            false if clustering fails
        )pbdoc")
        .def("matriz", &KMeans::resultado, "Retorna resultado das operacoes");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

int main() {

    //const std::string CAMINHO_ARQUIVOS = "../zonas_urbanas_py/tests/data/";

    //arma::sp_mat arquivo_matriz_esparsa;
    //bool lido_com_sucesso = arquivo_matriz_esparsa.load("C:/Users/arqui/Documents/Repositorios/zonas_urbanas/zonas_urbanas_py/teste.coo.npz", arma::pgm_binary);
    //if (!lido_com_sucesso) {
    //    std::cout << "O arquivo nao pode ser aberto" << std::endl;
    //    return 1;
    //}
    //arma::mat arquivo_matriz_densa = arma::mat(arquivo_matriz_esparsa);

    //std::cout << "matriz_esparsa simetria: " << matriz_esparsa.is_symmetric() << std::endl;
    //std::cout << "matriz_densa simetria: " << matriz_densa.is_symmetric() << std::endl;

    //arma::sp_mat arquivo = open_matrix_market_file("C:/Users/arqui/Documents/Repositorios/zonas_urbanas/zonas_urbanas_py/test_io.mtx", ?? , 3);
    //std::cout << "matriz_esparsa simetria: " << arquivo.is_symmetric() << std::endl;
    //arquivo.print("arquivo");

    //arma::mat arquivo_matriz_densa = arma::mat(arquivo);
    //std::cout << "matriz_esparsa simetria: " << arquivo_matriz_densa.is_symmetric() << std::endl;
    //arquivo_matriz_densa.print("arquivo_matriz_densa");

    //arma::mat matriz = { {1, 2, 8},{2, 1, 1},{8, 1, 1} };

    //arma::mat matriz = {
    //    {2, 1, 0, 0, 1, 0},
    //    {1, 0, 1, 0, 1, 0},
    //    {0, 1, 0, 1, 0, 0},
    //    {0, 0, 1, 0, 1, 1},
    //    {1, 1, 0, 1, 0, 0},
    //    {0, 0, 0, 1, 0, 0}
    //};

    //std::cout << "simetria: " << matriz.is_symmetric() << std::endl;
    //std::cout << "hermitiana: " << matriz.is_hermitian() << std::endl;

    //arma::vec auto_valores;
    //arma::mat auto_vetores;
    //bool ok = arma::eig_sym(auto_valores, auto_vetores, matriz, "dc"); // 40 segundos

    arma::fmat matriz = {
        {2, 1, 0, 0, 1, 0},
        {1, 0, 1, 0, 1, 0},
        {0, 1, 0, 1, 0, 0},
        {0, 0, 1, 0, 1, 1},
        {1, 1, 0, 1, 0, 0},
        {0, 0, 0, 1, 0, 0}
    };
    arma::cx_fvec  auto_valores;
    arma::cx_fmat  auto_vetores;

    auto t1 = std::chrono::high_resolution_clock::now();
    bool ok = arma::eig_gen(auto_valores, auto_vetores, matriz);
    auto t2 = std::chrono::high_resolution_clock::now();

    auto duracao = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1.0e+9;
    std::cout << "O calculo levou: " << duracao << " segundos" << std::endl;

    std::cout << "ok: " << ok << std::endl;

    auto_valores.print("auto_valores");
    auto_vetores.print("auto_vetores");

    //std::cout << "auto_valores(0, 0): " << auto_valores(0, 0) << std::endl;
    //std::cout << "auto_vetores(0, 1): " << auto_vetores(0, 1) << std::endl;
    //std::cout << "auto_valores tamaho: " << auto_valores.size() << std::endl;

    //auto_valores.print("auto_valores");

    ////std::cout << auto_valores(1, 0) << std::endl;
    //// converter arma::vec para arma::mat
    //arma::mat auto_valores_mat(auto_valores);
    ////auto_valores_mat.reshape(1, auto_valores.size());
    //std::cout << "auto_valores_convertido(0, 0): " << auto_valores_mat(0, 0) << std::endl;

    //auto_valores_mat.print("auto_valores_mat");

    //Eigen::MatrixXd _mat = converter_matriz(matriz);
    //std::cout << "verifica convercao entre matrizes: " << converter_matriz(_mat)(0, 0) << std::endl;

    //auto_vetores.save(CAMINHO_ARQUIVOS + "berkeley_sparse_matrix_autovetores_TESTE.csv", arma::csv_ascii);
    return 0;
}
