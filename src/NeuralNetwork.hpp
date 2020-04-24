#pragma once
#include <iostream>
#include <vector>

template<class NetworkT, class NetworkDataT>
class NeuralNetwork
{
    public:
    NeuralNetwork(NetworkT&& net)
    {
        std::cout << "NeuralNetwork constructor is called" << std::endl;
        _net = std::move(net);
    }
    /**
     * Get node names for the output layer
     */ 
    virtual std::vector<std::string> getLayerNames() = 0;

    /**
     * Process input and generate network data output
     */ 
    virtual std::vector<NetworkDataT> processInputImg(std::string&& inputImagePath) = 0;

    ~NeuralNetwork()
    {
        std::cout << "NeuralNetwork destructor is called" << std::endl;
    }
    NetworkT _net;//TODO: make it protected
    protected:
    
    private:

};
