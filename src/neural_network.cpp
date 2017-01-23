#include "neural_network.hpp"

#include <Eigen/Dense>

#include <vector>

#include <algorithm>

#include <iostream>
#include <fstream>

namespace nn {

network::network(std::vector<std::size_t> layer_sizes)
    :   sizes(layer_sizes), num_layers(layer_sizes.size()),
        biases(num_layers), weights(num_layers), nabla_b(num_layers),
        nabla_w(num_layers), delta_nabla_b(num_layers),
        delta_nabla_w(num_layers), activations(num_layers)
{
    for(std::size_t i = 1; i < num_layers; ++i)
    {
        biases[i] = Eigen::VectorXd::Random(sizes[i]);
        weights[i] = Eigen::MatrixXd::Random(sizes[i], sizes[i - 1]);

        nabla_b[i] = Eigen::VectorXd::Zero(sizes[i]);
        nabla_w[i] = Eigen::MatrixXd::Zero(sizes[i], sizes[i - 1]);
        
        delta_nabla_b[i] = Eigen::VectorXd::Zero(sizes[i]);
        delta_nabla_w[i] = Eigen::MatrixXd::Zero(sizes[i], sizes[i - 1]);            
    }      
}

Eigen::VectorXd network::feedforward(Eigen::VectorXd a)
{
    for (std::size_t i = 1; i < num_layers; ++i)
        a = sigmoid(weights[i] * a + biases[i]);
        
    return a;
}

void network::stochastic_gradient_descent(std::vector<data_set>& training_data,
    std::size_t epochs, std::size_t mini_batch_size, double eta,
    std::vector<data_set> const& test_data)
{
    std::size_t num_data = training_data.size();
    
    for (std::size_t epoch = 0; epoch < epochs; ++epoch)
    {
        std::random_shuffle(training_data.begin(), training_data.end());
        
        std::size_t idx_begin = 0;
        std::size_t idx_end =
            safe_advance(idx_begin, training_data.size(), mini_batch_size);

        do
        {              
            update_mini_batch(idx_begin, idx_end,  training_data, eta);
            
            idx_begin =
                safe_advance(idx_begin, training_data.size(),
                                mini_batch_size);
            idx_end =
                safe_advance(idx_end, training_data.size(),
                                mini_batch_size);
        }
        while(idx_begin != idx_end);

        
        if (test_data.size() > 0)
        {
            auto accuracy = evaluate(test_data);
            std::cout   << "Epoch " << epoch << ": " << accuracy 
                        << " / " << test_data.size() << "\n";          

        }
        else
            std::cout << "Epoch " << epoch << " complete\n";            
    }
}

void network::update_mini_batch(std::size_t begin, std::size_t end,
    std::vector<data_set> const& training_data, double eta)
{        
    for(std::size_t i = 1; i < num_layers; ++i)
    {
        nabla_b[i].setZero();
        nabla_w[i].setZero();
    }
    
    for (std::size_t idx = begin; idx < end; ++idx)
    {
        //TODO hpx::parallel::for_each
        backprop(training_data[idx], delta_nabla_b, delta_nabla_w);
      
        for (std::size_t l = 1; l < num_layers; ++l)
        {
            nabla_b[l].noalias() += delta_nabla_b[l];
            nabla_w[l].noalias() += delta_nabla_w[l];
        }
    }        

    for (std::size_t l = 1; l < num_layers; ++l)
    {
        weights[l].noalias() -= nabla_w[l] * (eta / (end-begin)); 
        biases[l].noalias() -= nabla_b[l] * (eta / (end-begin)); 
    }
}

void network::backprop(data_set const& training_data,
    std::vector<Eigen::VectorXd>& nabla_b,
    std::vector<Eigen::MatrixXd>& nabla_w)
{        
    Eigen::VectorXd activation = training_data.data;
            
    activations[0] = activation;
    
    std::vector<Eigen::VectorXd> zs(num_layers);
    
    for (std::size_t i = 1; i < num_layers; ++i)
    {                        
        auto z = weights[i] * activation + biases[i];
        zs[i] = z;
        activation = sigmoid(z);
        activations[i] = activation;
    }
    
    Eigen::VectorXd delta =
        cost_derivative(activations[num_layers - 1], training_data.labels)
            .cwiseProduct(sigmoid_prime(zs[num_layers - 1]));
     
    nabla_b[num_layers - 1] = delta;
    
    nabla_w[num_layers - 1].noalias() =
        delta * activations[num_layers - 2].transpose();

    for (std::size_t l = num_layers - 2; l >= 1; --l)
    {            
        delta.noalias() =
            (weights[l+1].transpose() * delta)
                .cwiseProduct(sigmoid_prime(zs[l]));

        nabla_b[l].noalias() = delta;
        nabla_w[l].noalias() = delta * activations[l - 1].transpose();                        
    }
}

std::size_t network::evaluate(std::vector<data_set> const& test_data)
{
    std::size_t sum = 0;
    
    for (auto const& d : test_data)
    {
        Eigen::VectorXd result = feedforward(d.data);
        
        std::size_t max_index;
        result.maxCoeff(&max_index);
        
        if (max_index == d.labels[0])
            sum++;            
    }
    
    return sum;
}

inline Eigen::VectorXd network::cost_derivative(Eigen::VectorXd const& activations,
    Eigen::VectorXd const& target)
{
    return activations - target;
}

Eigen::VectorXd split(std::string const &input, std::size_t num_items) { 
    std::stringstream ss(input);
    std::string item;

    Eigen::VectorXd ret(num_items);

    std::size_t i = 0;
    while (ss >> item)
        ret[i++] = std::atof(item.c_str());
  
    return ret;
}    

std::vector<data_set> load_data(std::string data_file, std::string label_file, std::size_t num_features, std::size_t num_labels)
{
    std::vector<data_set> data;
    
    std::ifstream dfile(data_file);
    std::ifstream lfile(label_file);
                
    for (std::string dline, lline; getline(dfile, dline) && getline(lfile, lline);)
    {
        data_set d;
        
        d.data = split(dline, num_features);
        d.labels = split(lline, num_labels);
        
        data.emplace_back(d);
    }
    
    return data;
}

}