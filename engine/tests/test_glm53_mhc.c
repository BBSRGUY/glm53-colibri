/* Validate glm53.c's mHC path against transformers' DeepseekV4HyperConnection.
 *
 * The fixture is produced by work/mhc_oracle.py, which drives the module that ships
 * in transformers -- so the engine and its oracle do not share an author. This links
 * glm53_mhc.h, the same header glm53.c includes, not a copy of it.
 *
 *   build: make test-glm53-mhc
 *   run:   ./test_glm53_mhc <fixture.bin>
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "../glm53_mhc.h"

static float *rd(FILE *f, size_t n) {
    float *p = malloc(n * sizeof(float));
    if (!p || fread(p, sizeof(float), n, f) != n) {
        fprintf(stderr, "short read (%zu floats)\n", n);
        exit(1);
    }
    return p;
}

/* `tol` is RELATIVE to the tensor's peak magnitude. An absolute bound silently encodes
 * the value range of whatever fixture it was tuned on: this test passed at D=32 and then
 * "failed" at the real D=4096 purely because the same one-bf16-ulp error is larger in
 * absolute terms on larger values (7.8e-03 on peak 3.44, vs 3.7e-03 on peak ~1). */
static int check(const char *what, const float *got, const float *want, size_t n, float tol) {
    double maxe = 0.0, sum = 0.0, peak = 0.0;
    size_t at = 0;
    for (size_t i = 0; i < n; i++) {
        double e = fabs((double)got[i] - (double)want[i]);
        sum += e;
        if (e > maxe) { maxe = e; at = i; }
        if (fabs((double)want[i]) > peak) peak = fabs((double)want[i]);
    }
    if (peak <= 0.0) peak = 1.0;
    int ok = maxe <= tol * peak;
    printf("  %-10s n=%-6zu max|err|=%.3e  mean|err|=%.3e  %s\n",
           what, n, maxe, sum / (double)n, ok ? "OK" : "FAIL");
    if (!ok)
        printf("      worst at [%zu]: got %.9g want %.9g\n", at, got[at], want[at]);
    return ok;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "mhc_fixture.bin";
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 1; }

    int32_t M, D, S, iters;
    float eps, rms_eps;
    if (fread(&M, 4, 1, f) != 1 || fread(&D, 4, 1, f) != 1 ||
        fread(&S, 4, 1, f) != 1 || fread(&iters, 4, 1, f) != 1 ||
        fread(&eps, 4, 1, f) != 1 || fread(&rms_eps, 4, 1, f) != 1) {
        fprintf(stderr, "bad header\n"); return 1;
    }
    printf("fixture: M=%d D=%d S=%d iters=%d eps=%g rms_eps=%g\n", M, D, S, iters, eps, rms_eps);

    int mix = (2 + M) * M;
    float *fn = rd(f, (size_t)mix * M * D);
    float *base = rd(f, mix);
    float *scale = rd(f, 3);
    float *streams = rd(f, (size_t)S * M * D);
    float *blkout = rd(f, (size_t)S * D);
    float *e_post = rd(f, (size_t)S * M);
    float *e_comb = rd(f, (size_t)S * M * M);
    float *e_coll = rd(f, (size_t)S * D);
    float *e_upd = rd(f, (size_t)S * M * D);
    float *e_seed = rd(f, (size_t)S * M * D);
    float *e_mean = rd(f, (size_t)S * D);
    float *embed = rd(f, (size_t)S * D);
    fclose(f);

    float *post = malloc((size_t)S * M * sizeof(float));
    float *comb = malloc((size_t)S * M * M * sizeof(float));
    float *coll = malloc((size_t)S * D * sizeof(float));
    float *upd = malloc((size_t)S * M * D * sizeof(float));
    float *seed = malloc((size_t)S * M * D * sizeof(float));
    float *mean = malloc((size_t)S * D * sizeof(float));

    for (int s = 0; s < S; s++)
        glm53_mhc_pre(streams + (size_t)s * M * D, fn, scale, base, M, D,
                      rms_eps, eps, iters,
                      post + (size_t)s * M, comb + (size_t)s * M * M, coll + (size_t)s * D);
    for (int s = 0; s < S; s++)
        glm53_mhc_post(blkout + (size_t)s * D, streams + (size_t)s * M * D,
                       post + (size_t)s * M, comb + (size_t)s * M * M,
                       M, D, upd + (size_t)s * M * D);
    glm53_hc_seed(embed, S, M, D, seed);
    glm53_hc_mean(e_upd, S, M, D, mean);   /* mean the ORACLE's streams: isolates collapse */

    /* dsv4_mhc rounds its layer input to bf16, so `collapsed` and anything downstream
     * of it carries ~2^-8 relative error against an fp32/fp64 oracle by construction.
     * The gates (post/comb) are not rounded and must match much more tightly. */
    int ok = 1;
    printf("[mHC vs transformers DeepseekV4HyperConnection]\n");
    /* The gates are not bf16-rounded, so they must match far more tightly than the
     * bf16 floor (~4e-3). They are not EXACT either: the Sinkhorn input is an f32 dot
     * product over M*D terms -- 16384 at the real D=4096 -- and random-sign accumulation
     * there runs to ~sqrt(N)*eps = 1.5e-5. 2e-5 covers that while staying two orders
     * below anything that could hide a real disagreement. */
    ok &= check("post", post, e_post, (size_t)S * M, 2e-5f);
    ok &= check("comb", comb, e_comb, (size_t)S * M * M, 2e-5f);
    ok &= check("collapsed", coll, e_coll, (size_t)S * D, 5e-3f);   /* ~2^-8: one bf16 ulp */
    ok &= check("updated", upd, e_upd, (size_t)S * M * D, 5e-3f);   /* ~2^-8: one bf16 ulp */
    printf("[stream seed / collapse]\n");
    ok &= check("seed", seed, e_seed, (size_t)S * M * D, 0.0f);
    ok &= check("mean", mean, e_mean, (size_t)S * D, 2e-6f);

    printf("\n%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
