#include <torch/torch.h>
#include <iostream>
#include "mpi.h"

std::map<at::ScalarType, MPI_Datatype> mpiDatatype = {
    {at::kByte, MPI_UNSIGNED_CHAR},
    {at::kChar, MPI_CHAR},
    {at::kDouble, MPI_DOUBLE},
    {at::kFloat, MPI_FLOAT},
    {at::kInt, MPI_INT},
    {at::kLong, MPI_LONG},
    {at::kShort, MPI_SHORT},
};

/*
// Define a completely Linear Module.
struct Model : torch::nn::Module {
    Model()
    {
        // Construct and register Linear submodules.
        fc1 = register_module("fc1", torch::nn::Linear(784, 128));
        fc2 = register_module("fc2", torch::nn::Linear(128, 10));
    }

    // Implement the Net's algorithm.
    torch::Tensor forward(torch::Tensor x)
    {
        // Use one of many tensor manipulation functions.
        x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
        x = torch::relu(fc2->forward(x));
        return x;
    }

    // Use one of many "standard library" modules.
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};
*/

/*
//Stable Convolutional Module
struct Model : torch::nn::Module {
    
    Model()
          : conv1(torch::nn::Conv2dOptions(1, 10, 5)),
            conv2(torch::nn::Conv2dOptions(10, 20, 5)),
            fc1(320, 100),
            fc2(100, 10) {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv2_drop", conv2_drop);      
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }
    
   torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
        x = torch::relu(torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
        x = x.view({-1, 320});
        x = torch::relu(fc1->forward(x));
        x = torch::dropout(x, 0.5, is_training());
        x = fc2->forward(x);
        return torch::log_softmax(x, 1);
    }        

    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::Dropout2d conv2_drop;
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
};
*/

//Experimental Convolutional Module
struct Model : torch::nn::Module {

    Model()
          : conv1(torch::nn::Conv2dOptions(1, 10, 3)),
            conv2(torch::nn::Conv2dOptions(10, 20, 3)),
            fc1(500, 50),
            fc2(50, 10) {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv2_drop", conv2_drop);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }


    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
        x = torch::relu(torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
        x = x.view({-1, 500});
        x = torch::relu(fc1->forward(x));
        x = torch::dropout(x, 0.5, is_training());
        x = fc2->forward(x);
        return torch::log_softmax(x, 1);
    }

    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::Dropout2d conv2_drop;
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
};


int main(int argc, char *argv[])
{
    // parsing runtime args
    int file_write = (int)std::atoi(argv[1]);    
    int thres_type = (int)std::atoi(argv[2]); // 0 for non-adaptive, 1 for adaptive

    float parameter, constant, gamma;
    
    if(thres_type == 1) {
       parameter = (float)std::atof(argv[3]); // adaptive threshold
    } else {
       constant = (float)std::atof(argv[3]); //non-adaptive constant threshold
       gamma = (float)std::atof(argv[4]);
    }

    // history at sender and receiver
    auto sent_history = 2;
    auto recv_history = 2;

    // MPI variables
    int rank, numranks;
    int left, right;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;
    MPI_Request req1, req2;
    MPI_Win win;

    // ring arrangement
    if (rank == 0)
        left = numranks - 1;
    else
        left = rank - 1;

    if (rank == numranks - 1)
        right = 0;
    else
        right = rank + 1;

    // end MPI variables

    // Timer variables
    auto tstart = 0.0;
    auto tend = 0.0;

    // Read dataset
    std::string filename =
        "/afs/crc.nd.edu/user/s/sghosh2/Public/ML/mnist/data";
    auto dataset =
        torch::data::datasets::MNIST(filename)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());

    // Distributed Random Sampler - use sequential for just gradient descent
    auto data_sampler = torch::data::samplers::DistributedSequentialSampler(
        dataset.size().value(), numranks, rank, false);

    auto num_train_samples_per_pe = dataset.size().value() / numranks;

    // Generate dataloader
    auto batch_size = 64; //16; //num_train_samples_per_pe;
    auto data_loader = torch::data::make_data_loader(std::move(dataset),
                                                     data_sampler, batch_size);

    // sets manual seed - randomness in sgd later ?
    torch::manual_seed(0);

    auto model = std::make_shared<Model>();

    auto sz = model->named_parameters().size();
    auto param = model->named_parameters();

    // counting total number of elements in the model
    int num_elem_param = 0;
    for (int i = 0; i < sz; i++) {
        num_elem_param += param[i].value().numel();
    }
    if (rank == 0) {
        std::cout << "Number of parameters - " << sz << std::endl;
        std::cout << "Number of elements - " << num_elem_param << std::endl;
    }

    auto param_elem_size = param[0].value().element_size();

    // create memory window for parameters from left and right (STILL CASTING AS
    // FLOAT TYPE)
    auto win_size = 2 * num_elem_param;
    // auto win_mem = (float*)calloc(win_size, win_size*param_elem_size);
    float *win_mem;
    MPI_Alloc_mem(win_size * param_elem_size, MPI_INFO_NULL, &win_mem);
    MPI_Win_create(win_mem, win_size * param_elem_size, param_elem_size,
                   MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    // initializing RMA window
    for (int i = 0; i < win_size; i++) {
        *(win_mem + i) = 0.0;
    }

    // Threshold
    float thres[sz];

    // history of values at the sender - to estimate the threshold
    float last_sent_values_norm[sz];
    float last_sent_iters[sz];
    float sent_slopes_norm[sz][sent_history];

    // history of values at the receiver - to extrapolate parameters between new
    // messages
    float left_last_recv_values[num_elem_param];
    float left_last_recv_values_norm[sz];
    float left_last_recv_iters[sz];
    float left_recv_slopes[num_elem_param][recv_history];

    float right_last_recv_values[num_elem_param];
    float right_last_recv_values_norm[sz];
    float right_last_recv_iters[sz];
    float right_recv_slopes[num_elem_param][recv_history];

    // norm of current values at the receiver
    float left_recv_norm[sz];
    float right_recv_norm[sz];

    // initializing values
    for (int i = 0; i < sz; i++) {
        last_sent_values_norm[i] = 0.0;
        last_sent_iters[i] = 0.0;

        left_last_recv_values_norm[i] = 0.0;
        right_last_recv_values_norm[i] = 0.0;

        left_last_recv_iters[i] = 0.0;
        right_last_recv_iters[i] = 0.0;

        left_recv_norm[i] = 0.0;
        right_recv_norm[i] = 0.0;

        thres[i] = 0.0;

        for (int j = 0; j < sent_history; j++) {
            sent_slopes_norm[i][j] = 0.0;
        }
    }

    for (int i = 0; i < num_elem_param; i++) {
        left_last_recv_values[i] = 0.0;
        right_last_recv_values[i] = 0.0;

        for (int j = 0; j < recv_history; j++) {
            left_recv_slopes[i][j] = 0.0;
            right_recv_slopes[i][j] = 0.0;
        }
    }

    auto learning_rate = 0.05;

    torch::optim::SGD optimizer(
      model->parameters(), torch::optim::SGDOptions(learning_rate)); //.momentum(0.5));

    // File writing
    char send_name[30], recv_name[30], pe_str[3];

    std::ofstream fps; //file for sending log
    std::ofstream fpr; //file for receiving log

    sprintf(pe_str, "%d", rank);

    strcpy(send_name, "send");
    strcat(send_name, pe_str);
    strcat(send_name, ".txt");

    strcpy(recv_name, "recv");
    strcat(recv_name, pe_str);
    strcat(recv_name, ".txt");

    if (file_write == 1) {
        fps.open(send_name);
        fpr.open(recv_name);
    }
    // end file writing

    // Number of epochs
    auto num_epochs = 10; //250;

    // Previous epochs + pass number through current epoch 
    auto pass_num = 0;

    // Number of epochs where event condition not verified (due to starting
    // oscillations)
    auto initial_comm_passes = 30;

    int num_events = 0;

    // start timer
    tstart = MPI_Wtime();

    for (size_t epoch = 1; epoch <= num_epochs; ++epoch) {
        int num_correct = 0;

        for (auto &batch : *data_loader) {
            pass_num++;

            auto ip = batch.data;
            auto op = batch.target.squeeze();

            // convert to required formats
            ip = ip.to(torch::kF32);
            op = op.to(torch::kLong);

            // Reset gradients
            model->zero_grad();

            // Execute forward pass
            auto prediction = model->forward(ip);

            // Compute cross entropy loss
            // Direct function not yet supported in libtorch

            auto loss = torch::nll_loss(torch::log_softmax(prediction, 1), op);

            /*
            // Print loss
            if (epoch % 1 == 0 && file_write == 1) {
                fps << epoch << ", " << loss.item<float>() << std::endl;
            }
            */

            // Backpropagation
            loss.backward();

            int disp = 0;  // running displacement for RMA window

            // parameter loop
            for (auto i = 0; i < sz; i++) {

                // getting dimensions of tensor
                int num_dim = param[i].value().dim();
                std::vector<int64_t> dim_array;
                for (int j = 0; j < num_dim; j++) {
                    dim_array.push_back(param[i].value().size(j));
                }

                // flattening the tensor and copying it to a 1-D vector
                auto flat = torch::flatten(param[i].value());

                auto temp = (float *)calloc(flat.numel(),
                                            flat.numel() * param_elem_size);
                for (int j = 0; j < flat.numel(); j++) {
                    *(temp + j) = flat[j].item<float>();
                }

                // consider norm of current parameter
                auto curr_norm = torch::norm(flat).item<float>();
                auto value_diff =
                    std::fabs(curr_norm - last_sent_values_norm[i]);
                auto iter_diff = pass_num - last_sent_iters[i];

                if (thres_type == 1) {
                    thres[i] = thres[i] * std::pow(parameter, iter_diff);
                } else {
                    thres[i] = constant * std::pow(gamma, epoch);
                }

                // Printing value of norm of current parameter
                if (file_write == 1) {
                   fps << curr_norm << ",  " << thres[i] << ",  ";
                }

                // SENDING OPERATIONS
                // event - based on norm of current parameter
                if (value_diff >= thres[i] ||
                    pass_num < initial_comm_passes) {
                    num_events += 2;  // for both left and right neighbors

                    // PUSH ENTIRE MSG
                    // send to left
                    MPI_Win_lock(MPI_LOCK_SHARED, left, 0,
                                 win);
                    MPI_Put(temp, flat.numel(), MPI_FLOAT, left,
                            (num_elem_param + disp), flat.numel(), MPI_FLOAT,
                            win);
                    // MPI_Win_flush(left, win);
                    MPI_Win_unlock(left, win);

                    // send to right
                    MPI_Win_lock(MPI_LOCK_SHARED, right, 0, win);
                    MPI_Put(temp, flat.numel(), MPI_FLOAT, right, disp,
                            flat.numel(), MPI_FLOAT, win);
                    // MPI_Win_flush(right, win);
                    MPI_Win_unlock(right, win);

                    // Shifting previous slope values
                    auto slope_avg = 0.0;
                    int j = 0;
                    for (j = 0; j < sent_history - 1; j++) {
                        sent_slopes_norm[i][j] = sent_slopes_norm[i][j + 1];
                        slope_avg += sent_slopes_norm[i][j];
                    }

                    // Calculating new slope value
                    sent_slopes_norm[i][j] = value_diff / iter_diff;
                    slope_avg += sent_slopes_norm[i][j];
                    slope_avg = slope_avg / sent_history;

                    // Calculating new threshold if adaptive
                    if(thres_type == 1) {
                        thres[i] = slope_avg;
                    }

                    // update last communicated parameters
                    last_sent_values_norm[i] = curr_norm;
                    last_sent_iters[i] = pass_num;

                    //record that an event was triggered
                    if(file_write == 1) {
                      fps << "1,  ";
                    }
                }
                else {
                    if(file_write == 1) {
                        fps << "0,  ";
                    }
                } //end send

                // RECEIVING OPERATIONS
                // unpack 1-D vector from corresponding displacement and form
                // tensor

                // Left neighbor
                auto left_recv = (float *)calloc(
                    flat.numel(), flat.numel() * param_elem_size);
                float left_temp = 0.0;
                for (int j = 0; j < flat.numel(); j++) {
                    *(left_recv + j) = *(win_mem + disp + j);
                    left_temp += std::pow(*(left_recv + j), 2);
                }
                left_temp = std::sqrt(left_temp / flat.numel());
                left_recv_norm[i] = left_temp;
                auto left_recv_diff = std::fabs(left_recv_norm[i] - left_last_recv_values_norm[i]);

                if (left_recv_diff > 0) {
                    // new value from left received

                    // shift old values
                    int j, k;
                    for (j = 0; j < flat.numel(); j++) {
                        // shifting old slopes
                        for (k = 0; k < recv_history - 1; k++) {
                            left_recv_slopes[disp + j][k] =
                                left_recv_slopes[disp + j][k + 1];
                        }
                        // calculating new slope
                        left_recv_slopes[disp + j][k] =
                            (left_recv[j] - left_last_recv_values[disp + j]) /
                            (pass_num - left_last_recv_iters[i]);
                        left_last_recv_values[disp + j] = left_recv[j];
                    }

                    left_last_recv_values_norm[i] = left_recv_norm[i];
                    left_last_recv_iters[i] = pass_num;

                    // Record that new value is received
                    
                    if (file_write == 1) {
                       fpr << "1,  ";
                    }
                    
                } else {
                    left_temp = 0;
                    
                    for (int j = 0; j < flat.numel(); j++) {
                        auto slope_avg = 0.0;
                        for (int k = 0; k < recv_history; k++) {
                            slope_avg += left_recv_slopes[disp + j][k];
                        }
                        slope_avg = slope_avg / recv_history;

                        left_recv[j] =
                            left_last_recv_values[disp + j] +
                            slope_avg * (pass_num - left_last_recv_iters[i]);

                        //compute extrapolated norm
                        left_temp += std::pow(*(left_recv + j), 2);
                    }
                    left_temp = std::sqrt(left_temp / flat.numel());

                    // Record that extrapolation is done
                    if (file_write == 1) {
                       fpr << "0,  ";
                    }
                } //end left recv

                // Writing value received
                if (file_write == 1) {
                   fpr << left_temp << ",  "; // << left_recv_diff << ",  ";
                }

                // forming left tensor - either new message or extrapolated
                torch::Tensor left_tensor =
                    torch::from_blob(left_recv, dim_array, torch::kFloat)
                        .clone();

                // RIGHT NEIGHBOR
                auto right_recv = (float *)calloc(
                    flat.numel(), flat.numel() * param_elem_size);
                float right_temp = 0.0;
                for (int j = 0; j < flat.numel(); j++) {
                    *(right_recv + j) = *(win_mem + num_elem_param + disp + j);
                    right_temp += std::pow(*(right_recv + j), 2);
                }
                right_temp = std::sqrt(right_temp / flat.numel());
                right_recv_norm[i] = right_temp;
                auto right_recv_diff = std::fabs(right_recv_norm[i] - right_last_recv_values_norm[i]);

                if (right_recv_diff >
                    0) {
                    // new value from right received

                    // shift old values
                    int j, k;
                    for (j = 0; j < flat.numel(); j++) {
                        for (k = 0; k < recv_history - 1; k++) {
                            right_recv_slopes[disp + j][k] =
                                right_recv_slopes[disp + j][k + 1];
                        }
                        right_recv_slopes[disp + j][k] =
                            (right_recv[j] - right_last_recv_values[disp + j]) /
                            (pass_num - right_last_recv_iters[i]);
                        right_last_recv_values[disp + j] = right_recv[j];
                    }

                    right_last_recv_values_norm[i] = right_recv_norm[i];
                    right_last_recv_iters[i] = pass_num;

                    //record that new value is received
                    
                    if (file_write == 1) {
                       fpr << "1,  ";
                    }
                    
                } else {
                    right_temp = 0;

                    for (int j = 0; j < flat.numel(); j++) {
                        auto slope_avg = 0.0;
                        for (int k = 0; k < recv_history; k++) {
                            slope_avg += right_recv_slopes[disp + j][k];
                        }
                        slope_avg = slope_avg / recv_history;

                        right_recv[j] =
                            right_last_recv_values[disp + j] +
                            slope_avg * (pass_num - right_last_recv_iters[i]);

                        // extrapolated norm
                        right_temp += std::pow(*(right_recv + j), 2);
                    }
                    right_temp = std::sqrt(right_temp / flat.numel());   

                    //record that extrpolation is done
                    if (file_write == 1) {
                       fpr << "0,  ";
                    }
                } //end right recv

                //writing value received
                if (file_write == 1) {
                   fpr << right_temp << ",  "; // << right_recv_diff << ",  ";
                }

                // forming right tensor - new message or extrapolated
                torch::Tensor right_tensor =
                    torch::from_blob(right_recv, dim_array, torch::kFloat)
                        .clone();

                // averaging with neighbors
                param[i].value().data().add_(left_tensor.data());
                param[i].value().data().add_(right_tensor.data());
                param[i].value().data().div_(3);

                // updating displacement
                disp = disp + flat.numel();

                // freeing temp arrays
                free(temp);
                free(left_recv);
                free(right_recv);
            } //end parameter loop

            if (file_write == 1) {
               fps << std::endl;
               fpr << std::endl;
            }

            // Update parameters
            optimizer.step();

            // Accuracy
            auto guess = prediction.argmax(1);
            num_correct += torch::sum(guess.eq_(op)).item<int64_t>();

        }  // end batch loader

        auto accuracy = 100.0 * num_correct / num_train_samples_per_pe;

        std::cout << epoch << ", " << accuracy << std::endl;

    }  // end epochs

    // End timer
    tend = MPI_Wtime();
    if (rank == 0) {
        std::cout << "Training time - " << (tend - tstart) << std::endl;
    }

    // Print event stats
    std::cout << "No of events in rank " << rank << " - " << num_events
              << std::endl;

    if (file_write == 1) {
        fps.close();
        fpr.close();
    }

    // Averaging learnt model - relevant only for rank 0
    for (int i = 0; i < sz; i++) {
        MPI_Allreduce(MPI_IN_PLACE, param[i].value().data_ptr(),
                      param[i].value().numel(),
                      mpiDatatype.at(param[i].value().scalar_type()), MPI_SUM,
                      MPI_COMM_WORLD);
        if (rank == 0) {
            param[i].value().data() = param[i].value().data() / numranks;
        }
    }

    // Accumulating number of events in all PEs
    MPI_Allreduce(MPI_IN_PLACE, &num_events, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "Total number of events - " << num_events << std::endl;
    }

    // Testing only in rank 0
    if (rank == 0) {
        auto test_dataset =
            torch::data::datasets::MNIST(
                filename, torch::data::datasets::MNIST::Mode::kTest)
                .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                .map(torch::data::transforms::Stack<>());

        auto num_test_samples = test_dataset.size().value();
        auto test_loader = torch::data::make_data_loader(
            std::move(test_dataset), num_test_samples);

        model->eval(); //enabling evaluation mode to prevent backpropagation

        int num_correct = 0;

        for (auto &batch : *test_loader) {
            auto ip = batch.data;
            auto op = batch.target.squeeze();

            // convert to required format
            ip = ip.to(torch::kF32);
            op = op.to(torch::kLong);

            auto prediction = model->forward(ip);

            auto loss = torch::nll_loss(torch::log_softmax(prediction, 1), op);

            std::cout << "Test loss - " << loss.item<float>() << " "
                      << std::endl;

            auto guess = prediction.argmax(1);

            // std::cout << "Prediction: " << std::endl << prediction <<
            // std::endl;

            /*
            std::cout << "Output  Guess" << std::endl;
            for(auto i = 0; i < num_test_samples; i++)
            {
                std::cout << op[i].item<int64_t>() << "  " <<
            guess[i].item<int64_t>() << std::endl;
            }
            */

            num_correct += torch::sum(guess.eq_(op)).item<int64_t>();

        }  // end test loader

        std::cout << "Num correct - " << num_correct << std::endl;
        std::cout << "Test Accuracy - "
                  << 100.0 * num_correct / num_test_samples << std::endl;
    }  // end rank 0

    MPI_Finalize();
}
