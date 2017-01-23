#include "include/neural_network.hpp"

int main()
{
    nn::network net({784, 30, 10});
    unsigned epochs = 30;
    unsigned mini_batch_size = 10;
    
    double eta = 3.0;
    
    auto training_data = nn::load_data("../training_images.dat", "../training_labels.dat", 784, 10);
    auto test_data = nn::load_data("../test_images.dat", "../test_labels.dat", 784, 1);
    
    net.stochastic_gradient_descent(training_data, epochs, mini_batch_size, eta, test_data);
            
    return 0;
}