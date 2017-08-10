#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include <fstream>

using namespace tensorflow;

int main(int argc, char* argv[]){
    // Initialize a TensorFlow session
    Session* session;
    Status status = NewSession(SessionOptions(), &session);
    if (!status.ok()){
        std::cout << status.ToString() << "\n";
        return 1;
    }

    // Read in the protobuf (.pb) graph
    GraphDef graph_def;
    status = ReadBinaryProto(Env::Default(), argv[1], &graph_def);
    if (!status.ok()){
        std::cout << status.ToString() << "\n";
        return 1;
    }

    // Add the graph to the session
    status = session->Create(graph_def);
    if (!status.ok()){
        std::cout << status.ToString() << "\n";
        return 1;
    }

    std::ifstream fin (argv[2], std::ios::binary);
    std::stringstream image_ss;
    image_ss << fin.rdbuf();
    string image_string = image_ss.str();
    fin.close();

    Tensor image_tensor(DT_STRING, TensorShape());
    image_tensor.scalar<string>()() = image_string;

    // Setup inputs and outputs
    std::vector<std::pair<string, Tensor>> inputs = {
        { "input_image_string", image_tensor}
    };

    // The session will initialize the outputs
    std::vector<Tensor> outputs;

    status = session->Run(inputs, {"top_k:0", "top_k:1"}, {}, &outputs);
    if (!status.ok()){
        std::cout << status.ToString() << "\n";
        return 1;
    }

    Tensor values = outputs[0];
    Tensor indices = outputs[1];    

    TTypes<float>::Flat values_flat = values.flat<float>();
    TTypes<int32>::Flat indices_flat = indices.flat<int32>();
    
    for (int pos = 0; pos < 5; ++pos) {
        const int label_index = indices_flat(pos);
        const float score = values_flat(pos);
        std::cout << (label_index - 1) << "\n" << score << "\n";
    }

    // Free any resources used by the session
    session->Close();
    return 0;
}

