#include <Eigen/Dense>
#include <vector>

namespace nn {

inline Eigen::VectorXd sigmoid(Eigen::VectorXd const& a)
{
    auto f = [](double const v) {return 1. / (1 + std::exp(-v));} ;
    return a.unaryExpr(f);
}

inline Eigen::VectorXd sigmoid_prime(Eigen::VectorXd const& a)
{            
    return sigmoid(a).cwiseProduct(Eigen::VectorXd::Ones(a.size()) - sigmoid(a));
}

template<typename Iter>
inline Iter safe_advance(Iter it, Iter last, std::size_t step)
{
    return (last - it > step) ? it + step : last;
}

struct data_set
{       
    Eigen::VectorXd data;
    Eigen::VectorXd labels;
};

class network
{
    
public:
    network(std::vector<std::size_t> layer_sizes);
    
    Eigen::VectorXd feedforward(Eigen::VectorXd a);
    
    void stochastic_gradient_descent(std::vector<data_set>& training_data,
        std::size_t epochs, std::size_t mini_batch_size, double eta,
        std::vector<data_set> const& test_data);

protected:
    void update_mini_batch(std::size_t begin, std::size_t end,
        std::vector<data_set> const& training_data, double eta);
    
    void backprop(data_set const& training_data,
        std::vector<Eigen::VectorXd>& nabla_b,
        std::vector<Eigen::MatrixXd>& nabla_w);
    
    std::size_t evaluate(std::vector<data_set> const& test_data);
    
    inline Eigen::VectorXd cost_derivative(Eigen::VectorXd const& activations,
        Eigen::VectorXd const& target);
    
    std::vector<std::size_t> sizes;
    std::size_t num_layers;
    
    std::vector<Eigen::VectorXd> biases;
    std::vector<Eigen::MatrixXd> weights;
    
    std::vector<Eigen::VectorXd> nabla_b;        
    std::vector<Eigen::MatrixXd> nabla_w;    
    std::vector<Eigen::VectorXd> delta_nabla_b;;        
    std::vector<Eigen::MatrixXd> delta_nabla_w;
    
    std::vector<Eigen::VectorXd> activations;
};

Eigen::VectorXd split(std::string const &input, std::size_t num_items);

std::vector<data_set> load_data(std::string data_file, std::string label_file,
    std::size_t num_features, std::size_t num_labels);
}