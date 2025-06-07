
#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>

struct scale_args {
    float *input;
    float *output;
    int size;
    float factor;
};

SEC("csd/scale")
int vector_scale(struct scale_args *args) {
    for (int i = 0; i < args->size; i++) {
        args->output[i] = args->input[i] * args->factor;
    }
    return 0;
}

char _license[] SEC("license") = "GPL";
