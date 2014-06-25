#include <iostream>
#include <omp.h>

#include "common.h"
#include "timer.h"
#include "gbdt.h"

namespace {

struct Option
{
    Option() : nr_tree(30), nr_thread(1) {}
    std::string Tr_path, TrS_path, Va_path, VaS_path, Va_out_path, Tr_out_path;
    uint32_t nr_tree, nr_thread;
};

std::string train_help()
{
    return std::string(
"usage: gbdt [<options>] <dense_validation_path> <sparse_validation_path> <dense_train_path> <sparse_train_path> <validation_output_path> <train_output_path>\n"
"\n"
"options:\n"
"-d <depth>: set the maximum depth of a tree\n"
"-s <nr_thread>: set the maximum number of threads\n"
"-t <nr_tree>: set the number of trees\n");
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
        if(args[i].compare("-d") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            CART::max_depth = std::stoi(args[++i]);
        }
        else if(args[i].compare("-t") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.nr_tree = std::stoi(args[++i]);
        }
        else if(args[i].compare("-s") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.nr_thread = std::stoi(args[++i]);
        }
        else
        {
            break;
        }
    }

    if(i != argc-6)
        throw std::invalid_argument("invalid command");

    opt.Va_path = args[i++];
    opt.VaS_path = args[i++];
    opt.Tr_path = args[i++];
    opt.TrS_path = args[i++];
    opt.Va_out_path = args[i++];
    opt.Tr_out_path = args[i++];

    return opt;
}

void write(Problem const &prob, GBDT const &gbdt, std::string const &path)
{
    FILE *f = open_c_file(path, "w");

    for(uint32_t i = 0; i < prob.nr_instance; ++i)
    {
        std::vector<float> x = construct_instance(prob, i);
        std::vector<uint32_t> indices = gbdt.get_indices(x.data());

        fprintf(f, "%d", static_cast<int>(prob.Y[i]));
        for(uint32_t t = 0; t < indices.size(); ++t)
            fprintf(f, " %d", indices[t]);
        fprintf(f, "\n");
    }

    fclose(f);
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
    Problem const Tr = read_data(opt.Tr_path, opt.TrS_path);
    Problem const Va = read_data(opt.Va_path, opt.VaS_path);
    std::cout << "done\n" << std::flush;

	omp_set_num_threads(static_cast<int>(opt.nr_thread));

    GBDT gbdt(opt.nr_tree);
    gbdt.fit(Tr, Va);

    write(Tr, gbdt, opt.Tr_out_path);
    write(Va, gbdt, opt.Va_out_path);

    return EXIT_SUCCESS;
}
