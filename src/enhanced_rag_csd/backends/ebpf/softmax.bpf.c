
#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>

struct softmax_args {
    float *input;
    float *output;
    int size;
    float temperature;
};

SEC("csd/softmax")
int compute_softmax(struct softmax_args *args) {
    // Find maximum value for numerical stability
    float max_val = args->input[0];
    for (int i = 1; i < args->size; i++) {
        if (args->input[i] > max_val) {
            max_val = args->input[i];
        }
    }
    
    // Compute exponentials and sum
    float sum = 0.0;
    for (int i = 0; i < args->size; i++) {
        float exp_val = __builtin_expf((args->input[i] - max_val) / args->temperature);
        args->output[i] = exp_val;
        sum += exp_val;
    }
    
    // Normalize
    for (int i = 0; i < args->size; i++) {
        args->output[i] /= sum;
    }
    
    return 0;
}

char _license[] SEC("license") = "GPL";
