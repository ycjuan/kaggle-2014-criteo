#include <stdexcept>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <omp.h>

#include "common.h"

namespace {

uint32_t const kMaxLineSize = 1000000;

uint32_t get_nr_line(std::string const &path)
{
    FILE *f = open_c_file(path.c_str(), "r");
    char line[kMaxLineSize];

    uint32_t nr_line = 0;
    while(fgets(line, kMaxLineSize, f) != nullptr)
        ++nr_line;

    fclose(f);

    return nr_line;
}

uint32_t get_nr_field(std::string const &path)
{
    FILE *f = open_c_file(path.c_str(), "r");
    char line[kMaxLineSize];

    fgets(line, kMaxLineSize, f);
    strtok(line, " \t");

    uint32_t nr_field = 0;
    while(1)
    {
        char *val_char = strtok(nullptr," \t");
        if(val_char == nullptr || *val_char == '\n')
            break;
        ++nr_field;
    }

    fclose(f);

    return nr_field;
}

void read_dense(Problem &prob, std::string const &path)
{
    char line[kMaxLineSize];

    FILE *f = open_c_file(path.c_str(), "r");
    for(uint32_t i = 0; fgets(line, kMaxLineSize, f) != nullptr; ++i)
    {
        char *p = strtok(line, " \t");
        prob.Y[i] = (atoi(p)>0)? 1.0f : -1.0f;
        for(uint32_t j = 0; j < prob.nr_field; ++j)
        {
            char *val_char = strtok(nullptr," \t");

            float const val = static_cast<float>(atof(val_char));

            prob.X[j][i] = Node(i, val);
        }
    }

    fclose(f);
}

void sort_problem(Problem &prob)
{
    struct sort_by_v
    {
        bool operator() (Node const lhs, Node const rhs)
        {
            return lhs.v < rhs.v;
        }
    };

    #pragma omp parallel for schedule(static)
    for(uint32_t j = 0; j < prob.nr_field; ++j)
    {
        std::vector<Node> &X1 = prob.X[j];
        std::vector<Node> &Z1 = prob.Z[j];
        std::sort(X1.begin(), X1.end(), sort_by_v());
        for(uint32_t i = 0; i < prob.nr_instance; ++i)
            Z1[X1[i].i] = Node(i, X1[i].v);
    }
}

void read_sparse(Problem &prob, std::string const &path)
{
    char line[kMaxLineSize];

    FILE *f = open_c_file(path.c_str(), "r");

    std::vector<std::vector<uint32_t>> buffer;

    uint64_t nnz = 0; 
    uint32_t nr_instance = 0;
    prob.SJP.push_back(0);
    for(; fgets(line, kMaxLineSize, f) != nullptr; ++nr_instance)
    {
        strtok(line, " \t");
        for( ; ; ++nnz)
        {
            char *idx_char = strtok(nullptr," \t");
            if(idx_char == nullptr || *idx_char == '\n')
                break;

            uint32_t const idx = atoi(idx_char);
            if(idx > buffer.size())
                buffer.resize(idx);
            buffer[idx-1].push_back(nr_instance);
            prob.SJ.push_back(idx-1);
        }
        prob.SJP.push_back(prob.SJ.size());
    }
    prob.SJ.shrink_to_fit();
    prob.SJP.shrink_to_fit();

    prob.nr_sparse_field = static_cast<uint32_t>(buffer.size());
    prob.SI.resize(nnz);
    prob.SIP.resize(prob.nr_sparse_field+1);
    prob.SIP[0] = 0;

    uint64_t p = 0;
    for(uint32_t j = 0; j < prob.nr_sparse_field; ++j)
    {
        for(auto i : buffer[j]) 
            prob.SI[p++] = i;
        prob.SIP[j+1] = p;
    }

    fclose(f);

    sort_problem(prob);
}

} //unamed namespace

Problem read_data(std::string const &dense_path, std::string const &sparse_path)
{
    Problem prob(get_nr_line(dense_path), get_nr_field(dense_path));

    read_dense(prob, dense_path);

    read_sparse(prob, sparse_path);

    return prob;
}

FILE *open_c_file(std::string const &path, std::string const &mode)
{
    FILE *f = fopen(path.c_str(), mode.c_str());
    if(!f)
        throw std::runtime_error(std::string("cannot open ")+path);
    return f;
}

std::vector<std::string> 
argv_to_args(int const argc, char const * const * const argv)
{
    std::vector<std::string> args;
    for(int i = 1; i < argc; ++i)
        args.emplace_back(argv[i]);
    return args;
}
