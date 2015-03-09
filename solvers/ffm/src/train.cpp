#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <omp.h>

#include "common.h"
#include "timer.h"

namespace {

struct Option
{
    Option() 
        : eta(0.1f), lambda(0.00002f), iter(15), nr_factor(4), 
          nr_threads(1), do_prediction(true) {}
    std::string Tr_path, Va_path;
    float eta, lambda;
    uint32_t iter, nr_factor, nr_threads;
    bool do_prediction;
};

std::string train_help()
{
    return std::string(
"usage: fm [<options>] <validation_path> <train_path>\n"
"\n"
"<validation_path>.out will be automatically generated at the end of training\n"
"\n"
"options:\n"
"-l <lambda>: set the regularization penalty\n"
"-k <factor>: set the number of latent factors, which must be a multiple of 4\n"
"-t <iteration>: set the number of iterations\n"
"-r <eta>: set the learning rate\n"
"-s <nr_threads>: set the number of threads\n"
"-q: if it is set, then there is no output file\n");
}

Option parse_option(std::vector<std::string> const &args)
{
    uint32_t const argc = static_cast<uint32_t>(args.size());

    if(argc == 0)
        throw std::invalid_argument(train_help());

    Option opt; 

    uint32_t i = 0;
    for(; i < argc; ++i)
    {
        if(args[i].compare("-t") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command\n");
            opt.iter = std::stoi(args[++i]);
        }
        else if(args[i].compare("-k") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command\n");
            opt.nr_factor = std::stoi(args[++i]);
            if(opt.nr_factor%4 != 0)
                throw std::invalid_argument("k should be a multiple of 4\n");
        }
        else if(args[i].compare("-r") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command\n");
            opt.eta = std::stof(args[++i]);
        }
        else if(args[i].compare("-l") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command\n");
            opt.lambda = std::stof(args[++i]);
        }
        else if(args[i].compare("-s") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command\n");
            opt.nr_threads = std::stoi(args[++i]);
        }
        else if(args[i].compare("-q") == 0)
        {
            opt.do_prediction = false;
        }
        else
        {
            break;
        }
    }

    if(i >= argc-1)
        throw std::invalid_argument("training or test set not specified\n");

    opt.Va_path = args[i++];
    opt.Tr_path = args[i++];

    return opt;
}

void init_model(Model &model)
{
    uint32_t const nr_factor = model.nr_factor;
    float const coef = 
        static_cast<float>(0.5/sqrt(static_cast<double>(nr_factor)));

    float * w = model.W.data();
    for(uint32_t j = 0; j < model.nr_feature; ++j)
    {
        for(uint32_t f = 0; f < model.nr_field; ++f)
        {
            for(uint32_t d = 0; d < nr_factor; ++d, ++w)
                *w = coef*static_cast<float>(drand48());
            for(uint32_t d = nr_factor; d < nr_factor; ++d, ++w)
                *w = 0;
            for(uint32_t d = nr_factor; d < 2*nr_factor; ++d, ++w)
                *w = 1;
        }
    }
}

void train(Problem const &Tr, Problem const &Va, Model &model, Option const &opt)
{
    std::vector<uint32_t> order(Tr.Y.size());
    for(uint32_t i = 0; i < Tr.Y.size(); ++i)
        order[i] = i;

    Timer timer;
    printf("iter     time    tr_loss    va_loss\n");
    for(uint32_t iter = 0; iter < opt.iter; ++iter)
    {
        timer.tic();

        double Tr_loss = 0;
        std::random_shuffle(order.begin(), order.end());
#pragma omp parallel for schedule(static)
        for(uint32_t i_ = 0; i_ < order.size(); ++i_)
        {
            uint32_t const i = order[i_];

            float const y = Tr.Y[i];
            
            float const t = wTx(Tr, model, i);

            float const expnyt = static_cast<float>(exp(-y*t));

            Tr_loss += log(1+expnyt);
               
            float const kappa = -y*expnyt/(1+expnyt);

            wTx(Tr, model, i, kappa, opt.eta, opt.lambda, true);
        }
        Tr_loss /= static_cast<double>(Tr.Y.size());

        double const Va_loss = predict(Va, model);

        printf("%4d %8.1f %10.5f %10.5f\n", 
               iter, timer.toc(), Tr_loss, Va_loss);
        fflush(stdout);
    }
}

} //unnamed namespace

int main(int const argc, char const * const * const argv)
{
    Option opt;
    try
    {
        opt = parse_option(argv_to_args(argc, argv));
    }
    catch(std::invalid_argument const &e)
    {
        std::cout << e.what();
        return EXIT_FAILURE;
    }

    std::cout << "reading data..." << std::flush;
    Problem const Va = read_problem(opt.Va_path);
    Problem const Tr = read_problem(opt.Tr_path);
    std::cout << "done\n" << std::flush;

    std::cout << "initializing model..." << std::flush;
    Model model(Tr.nr_feature, opt.nr_factor, Tr.nr_field);
    init_model(model);
    std::cout << "done\n" << std::flush;

	omp_set_num_threads(static_cast<int>(opt.nr_threads));

    train(Tr, Va, model, opt);

	omp_set_num_threads(1);

    if(opt.do_prediction)
        predict(Va, model, opt.Va_path+".out");

    return EXIT_SUCCESS;
}
