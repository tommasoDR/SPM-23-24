#include <ff/ff.hpp>
#include <cstring>
#include <vector>
#include <set>
#include <string>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <algorithm>
#include <atomic>
// g++ -std=c++17 -I./fastflow -O3 -o Word-Count-FF-par2 Word-Count-FF-par2.cpp

using namespace ff;

using umap = std::unordered_map<std::string, uint64_t>;
using pair = std::pair<std::string, uint64_t>;

struct Comp {
    bool operator()(const pair& p1, const pair& p2) const {
        return p1.second > p2.second;
    }
};

using ranking = std::multiset<pair, Comp>;

// ------ globals --------
std::atomic_int total_words{0};
volatile uint64_t extraworkXline{0};
// ----------------------

void tokenize_line(const std::string& line, umap& local_UM) {
    char *tmpstr;
    char *token = strtok_r(const_cast<char*>(line.c_str()), " \r\n", &tmpstr);
    while (token) {
        ++local_UM[std::string(token)];
        token = strtok_r(NULL, " \r\n", &tmpstr);
        ++total_words;
    }
    for (volatile uint64_t j{0}; j < extraworkXline; j++);
}

void process_chunk(const std::vector<std::string>& chunk, umap& local_UM) {
    for (const auto& line : chunk) {
        tokenize_line(line, local_UM);
    }
}

struct Worker: ff_node_t<std::vector<std::string>, umap> {
    umap* svc(std::vector<std::string>* chunk) {
        auto local_UM = new umap;
        process_chunk(*chunk, *local_UM);
        delete chunk;
        return local_UM;
    }
};

struct SourceSink: ff_monode_t<umap,std::vector<std::string>> {
    SourceSink(const std::vector<std::string>& files, int chunk_size):files(files), chunk_size(chunk_size) {}

    std::vector<std::string>* svc(umap* local_UM) {

        if(local_UM == nullptr) {
            std::vector<std::string> chunk;
            chunk.reserve(chunk_size);
            
            for (const auto& f : files) {
                std::ifstream file(f, std::ios_base::in);
                if (file.is_open()) {
                    std::string line;

                    while (std::getline(file, line)) {
                        if (!line.empty()) {
                            chunk.push_back(line);
                            if (chunk.size() == chunk_size) {
                                ff_send_out(new std::vector<std::string>(chunk));
                                chunk.clear();
                            }
                        }
                    }

                    file.close();
                }
            }

            if (!chunk.empty()) {
                ff_send_out(new std::vector<std::string>(chunk));
            }

            broadcast_task(EOS);
			return GO_ON;
        }
    
        for (const auto& entry : *local_UM) {
            UM[entry.first] += entry.second;
        }
        delete local_UM;
        return this->GO_ON;
    }

    const std::vector<std::string>& files;
    int chunk_size;
    umap UM;
    const umap& get_UM() const { return UM; }
};

int main(int argc, char *argv[]) {
    auto usage_and_exit = [argv]() {
        std::printf("use: %s filelist.txt [extraworkXline] [topk] [showresults] [nthreads] [chunk_size]\n", argv[0]);
        std::printf("     filelist.txt contains one txt filename per line\n");
        std::printf("     extraworkXline is the extra work done for each line, it is an integer value whose default is 0\n");
        std::printf("     topk is an integer number, its default value is 10 (top 10 words)\n");
        std::printf("     showresults is 0 or 1, if 1 the output is shown on the standard output\n");
        std::printf("     nthreads is the number of threads, its default value is 2\n");
        std::printf("     chunk_size is the maximum number of lines to process in a single task, its default value is 100\n\n");
        exit(-1);
    };

    std::vector<std::string> filenames;
    size_t topk = 10;
    bool showresults = false;
    int nth = 2;
    int chunk_size = 10000;

    if (argc < 2 || argc > 7) {
        usage_and_exit();
    }

    if (argc > 2) {
        try { extraworkXline = std::stoul(argv[2]);
        } catch (std::invalid_argument const& ex) {
            std::printf("%s is an invalid number (%s)\n", argv[2], ex.what());
            return -1;
        }
        if (argc > 3) {
            try { topk = std::stoul(argv[3]);
            } catch (std::invalid_argument const& ex) {
                std::printf("%s is an invalid number (%s)\n", argv[3], ex.what());
                return -1;
            }
            if (topk == 0) {
                std::printf("%s must be a positive integer\n", argv[3]);
                return -1;
            }
            if (argc > 4) {
                int tmp;
                try { tmp = std::stol(argv[4]);
                } catch (std::invalid_argument const& ex) {
                    std::printf("%s is an invalid number (%s)\n", argv[4], ex.what());
                    return -1;
                }
                if (tmp == 1) showresults = true;
            }
            if (argc > 5) {
                try { nth = std::stol(argv[5]);
                } catch (std::invalid_argument const& ex) {
                    std::printf("%s is an invalid number (%s)\n", argv[5], ex.what());
                    return -1;
                }
                if (nth <= 1) {
                    std::printf("%s, numer of threads must be a >1\n", argv[5]);
                    return -1;
                }
            }
            if (argc > 6) {
                try { chunk_size = std::stol(argv[6]);
                } catch (std::invalid_argument const& ex) {
                    std::printf("%s is an invalid number (%s)\n", argv[6], ex.what());
                    return -1;
                }
                if (chunk_size <= 0) {
                    std::printf("%s must be a positive integer\n", argv[6]);
                    return -1;
                }
            }
        }
    }

    if (std::filesystem::is_regular_file(argv[1])) {
        std::ifstream file(argv[1], std::ios_base::in);
        if (file.is_open()) {
            std::string line;
            while (std::getline(file, line)) {
                if (std::filesystem::is_regular_file(line))
                    filenames.push_back(line);
                else
                    std::cout << line << " is not a regular file, skipping it\n";
            }
        } else {
            std::printf("ERROR: opening file %s\n", argv[1]);
            return -1;
        }
        file.close();
    } else {
        std::printf("%s is not a regular file\n", argv[1]);
        usage_and_exit();
    }

    // used for storing results
    umap UM;

    // Create FastFlow nodes
    SourceSink sourceSink(filenames, chunk_size);
    std::vector<std::unique_ptr<ff_node>> workers;
    for (int i = 0; i < nth-1; ++i) {
        workers.push_back(make_unique<Worker>());
    }
    
    // Create the farm
    ff_Farm<> farm(std::move(workers), sourceSink);
    farm.remove_collector();
    farm.wrap_around();
    farm.set_scheduling_ondemand(); 

    // start the time
    auto start = getusec();

    // Run the farm
    if (farm.run_and_wait_end() < 0) {
        error("running farm\n");
        return -1;
    }

    // Collect results from the collector
    UM = sourceSink.get_UM();

    auto stop1 = getusec();

    // sorting in descending order
    ranking rank(UM.begin(), UM.end());

    auto stop2 = getusec();
    std::printf("Compute time (s) %f\nSorting time (s) %f\n",
                (stop1 - start) / 1000000.0, (stop2 - stop1) / 1000000.0);

    if (showresults) {
        // show the results
        std::cout << "Unique words " << rank.size() << "\n";
        std::cout << "Total words  " << total_words << "\n";
        std::cout << "Top " << topk << " words:\n";
        auto top = rank.begin();
        for (size_t i = 0; i < std::clamp(topk, 1ul, rank.size()); ++i)
            std::cout << top->first << '\t' << top++->second << '\n';
    }
}
